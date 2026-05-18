"""
Mask inference probe — 不训练, 纯 vLLM 推理对比 mask vs no-mask 的模型行为差异.

用 5 个 CVRP n=20 prompt 跑 2 次 (with mask + without mask), 5-10 分钟出结果.

直接看 5 个核心问题:
  1. completion 长度: with mask 是否全到 context limit (~4071)?
  2. 自然 EOS 数: with mask 是否 0 (= forbid_eos 真生效)?
  3. SECTION_2 anchor 出现率: with mask 是否能写出 "2. **Step-by-step construction**:"?
  4. dup / fullcov: with mask 是否 degenerate?
  5. mask trigger 行为: SECTION_2 / select_strict 是否真触发?

用法 (zhuoyi 集群):
  cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
  export MODEL=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_hybrid_cvrp20/final_model
  python probe_mask_inference.py 2>&1 | tee probe_mask_output.log

注: 用单 GPU 跑, 不依赖训练栈; 跟 production vLLM 配置一致 (cuda graph
enabled, prefix caching enabled), 复现 mask run 实际行为.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from vllm import LLM, SamplingParams

from utils.cvrp_mask_state import MaskConfig
from utils.vllm_cvrp_mask_processor import CVRPMaskProcessor
from problems.cvrp import CVRP


MODEL = os.environ.get(
    "MODEL",
    "/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_hybrid_cvrp20/final_model",
)
N = 20
N_PROMPTS = 5
MAX_TOKENS = 4096

SECTION_2_ANCHOR = "2. **Step-by-step construction**:"
SECTION_3_ANCHOR = "3. **Verification**:"
SECTION_4_ANCHOR = "4. **Final routes**:"


def analyze(name: str, outputs):
    """统计 outputs 中关键 metric."""
    lens = []
    anchor_2 = anchor_3 = anchor_4 = think_close = 0
    eos_stop = 0
    dup_counts = []  # 每条 trajectory 的 dup 次数
    cov_full = 0
    for o in outputs:
        text = o.outputs[0].text
        token_ids = o.outputs[0].token_ids
        lens.append(len(token_ids))
        if SECTION_2_ANCHOR in text:
            anchor_2 += 1
        if SECTION_3_ANCHOR in text:
            anchor_3 += 1
        if SECTION_4_ANCHOR in text:
            anchor_4 += 1
        if "</think>" in text:
            think_close += 1
        if o.outputs[0].finish_reason == "stop":
            eos_stop += 1
        # parse SECTION_2 内 "→ select X" 算 dup
        selects = re.findall(r"→\s*[Ss]elect\s+(\d+)", text)
        cust_ids = [int(s) for s in selects if 1 <= int(s) <= N]
        if cust_ids:
            dup = len(cust_ids) - len(set(cust_ids))
            dup_counts.append(dup)
            if len(set(cust_ids)) == N:
                cov_full += 1
        else:
            dup_counts.append(0)

    print(f"\n[{name}]")
    print(f"  长度 (tokens): min={min(lens)}, p50={sorted(lens)[len(lens)//2]}, "
          f"max={max(lens)}, all={lens}")
    print(f"  含 SECTION_2 anchor ({SECTION_2_ANCHOR!r:>35}): {anchor_2}/{N_PROMPTS}")
    print(f"  含 SECTION_3 anchor ('3. **Verification**:'):           {anchor_3}/{N_PROMPTS}")
    print(f"  含 SECTION_4 anchor ('4. **Final routes**:'):           {anchor_4}/{N_PROMPTS}")
    print(f"  含 </think>:                                            {think_close}/{N_PROMPTS}")
    print(f"  自然 EOS (finish=stop, 不是 length 截断):                {eos_stop}/{N_PROMPTS}")
    print(f"  dup_per_traj (think 段 select 重复数): {dup_counts}, mean={sum(dup_counts)/len(dup_counts):.2f}")
    print(f"  fullcov (think 段访问 20 unique customer): {cov_full}/{N_PROMPTS}")


def main():
    print(f"=" * 80)
    print(f"  Mask Inference Probe")
    print(f"=" * 80)
    print(f"Model: {MODEL}")
    print(f"N prompts: {N_PROMPTS}, CVRP n={N}, max_tokens={MAX_TOKENS}\n")

    # ── 1. 启动 vLLM ────────────────────────────────────────────────
    print("[1/5] Loading vLLM model (跟 production 配置一致: cuda graph + prefix caching)...")
    llm = LLM(
        model=MODEL,
        max_model_len=5120,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # ── 2. 生成 5 个 CVRP prompt ──────────────────────────────────────
    print("\n[2/5] Building 5 CVRP n=20 prompts (固定 seed=2025)...")
    rng = np.random.default_rng(seed=2025)
    prob = CVRP()
    prompts = []
    for i in range(N_PROMPTS):
        inst = prob.generate_instance(N, rng)
        msgs = prob.build_prompt(inst)
        chat_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(chat_text)
    prompt_lens = [len(tokenizer(p).input_ids) for p in prompts]
    print(f"  Prompt 长度 (tokens): {prompt_lens}")

    # ── 3. With mask 跑 ──────────────────────────────────────────────
    print("\n[3/5] Generating WITH mask (CVRPMaskProcessor enabled)...")
    mask_proc = CVRPMaskProcessor(
        n=N, tokenizer=tokenizer,
        cfg=MaskConfig(enabled=True, n=N, debug_log=False),
    )
    sampling_mask = SamplingParams(
        n=1, temperature=1.0, top_p=1.0, max_tokens=MAX_TOKENS,
        logits_processors=[mask_proc],
    )
    outputs_mask = llm.generate(prompts, sampling_mask)

    # ── 4. Without mask 跑 (对照) ──────────────────────────────────
    print("\n[4/5] Generating WITHOUT mask (baseline 对照)...")
    sampling_nomask = SamplingParams(
        n=1, temperature=1.0, top_p=1.0, max_tokens=MAX_TOKENS,
    )
    outputs_nomask = llm.generate(prompts, sampling_nomask)

    # ── 5. 分析对比 ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ANALYSIS — mask vs no-mask")
    print("=" * 80)
    analyze("WITH    mask", outputs_mask)
    analyze("WITHOUT mask", outputs_nomask)

    # ── 6. 保存所有 completion 完整内容到文件 ──────────────────────
    import json
    out_path = "probe_mask_completions.json"
    dump_data = {
        "model": MODEL,
        "n_prompts": N_PROMPTS,
        "max_tokens": MAX_TOKENS,
        "with_mask": [
            {
                "prompt_idx": i,
                "len_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
                "completion_text": o.outputs[0].text,
            }
            for i, o in enumerate(outputs_mask)
        ],
        "without_mask": [
            {
                "prompt_idx": i,
                "len_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
                "completion_text": o.outputs[0].text,
            }
            for i, o in enumerate(outputs_nomask)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, ensure_ascii=False, indent=2)

    # 也保存成纯文本方便 grep / 直接看
    txt_path = "probe_mask_completions.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL}\n")
        f.write(f"N prompts: {N_PROMPTS}, max_tokens: {MAX_TOKENS}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("WITH MASK — 全部 5 个 completion\n")
        f.write("=" * 80 + "\n")
        for i, o in enumerate(outputs_mask):
            f.write(f"\n----- Prompt {i} | len={len(o.outputs[0].token_ids)} "
                    f"| finish={o.outputs[0].finish_reason} -----\n")
            f.write(o.outputs[0].text)
            f.write("\n----- END Prompt {} -----\n".format(i))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("WITHOUT MASK (baseline) — 全部 5 个 completion\n")
        f.write("=" * 80 + "\n")
        for i, o in enumerate(outputs_nomask):
            f.write(f"\n----- Prompt {i} | len={len(o.outputs[0].token_ids)} "
                    f"| finish={o.outputs[0].finish_reason} -----\n")
            f.write(o.outputs[0].text)
            f.write("\n----- END Prompt {} -----\n".format(i))

    print(f"\n[6/7] 完整 completion 已保存:")
    print(f"  - JSON 格式 (含 metadata): {out_path}")
    print(f"  - 纯文本格式 (方便 grep): {txt_path}")

    # ── 7. 自动诊断 ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [7/7] AUTO-DIAGNOSIS")
    print("=" * 80)

    mask_eos = sum(1 for o in outputs_mask if o.outputs[0].finish_reason == "stop")
    nomask_eos = sum(1 for o in outputs_nomask if o.outputs[0].finish_reason == "stop")
    print(f"\n自然 EOS 数: mask={mask_eos}/{N_PROMPTS}, nomask={nomask_eos}/{N_PROMPTS}")
    if mask_eos == 0 and nomask_eos >= 3:
        print("  → ✓ 确认: mask 的 forbid_eos 在 vLLM 端真生效 (ban EOS 让 sequence 不能自然结束)")

    mask_anchor = sum(1 for o in outputs_mask if SECTION_2_ANCHOR in o.outputs[0].text)
    nomask_anchor = sum(1 for o in outputs_nomask if SECTION_2_ANCHOR in o.outputs[0].text)
    print(f"\nSECTION_2 anchor 出现率: mask={mask_anchor}/{N_PROMPTS}, nomask={nomask_anchor}/{N_PROMPTS}")
    if mask_anchor == 0 and nomask_anchor >= 3:
        print("  → ❌ 确诊: mask 启用让模型完全不写 SECTION_2 anchor")
        print("     (degenerate 真因, 跟 mask logic 内容无关, 是 attach 本身的副作用)")
    elif mask_anchor < nomask_anchor:
        print(f"  → ⚠️ mask 启用让 anchor 出现率从 {nomask_anchor}/{N_PROMPTS} 降到 {mask_anchor}/{N_PROMPTS}")
    elif mask_anchor >= nomask_anchor * 0.5:
        print("  → ✓ mask 启用没明显阻止 anchor 写出, 模型仍能进入 SECTION_2")
        print("     问题在 SECTION_2 内部的 mask 规则 (select_strict / partial_select)")

    avg_dup_mask = sum(
        len(re.findall(r"→\s*[Ss]elect\s+(\d+)", o.outputs[0].text)) -
        len(set(re.findall(r"→\s*[Ss]elect\s+(\d+)", o.outputs[0].text)))
        for o in outputs_mask
    ) / N_PROMPTS
    avg_dup_nomask = sum(
        len(re.findall(r"→\s*[Ss]elect\s+(\d+)", o.outputs[0].text)) -
        len(set(re.findall(r"→\s*[Ss]elect\s+(\d+)", o.outputs[0].text)))
        for o in outputs_nomask
    ) / N_PROMPTS
    print(f"\n平均 dup_per_traj (think 段 select 重复): mask={avg_dup_mask:.2f}, nomask={avg_dup_nomask:.2f}")
    if avg_dup_mask > avg_dup_nomask * 5:
        print(f"  → ❌ mask 启用让 dup 暴涨 {avg_dup_mask/max(avg_dup_nomask, 0.01):.0f}x")

    avg_len_mask = sum(len(o.outputs[0].token_ids) for o in outputs_mask) / N_PROMPTS
    avg_len_nomask = sum(len(o.outputs[0].token_ids) for o in outputs_nomask) / N_PROMPTS
    print(f"\n平均 completion 长度: mask={avg_len_mask:.0f}, nomask={avg_len_nomask:.0f}")
    if avg_len_mask > avg_len_nomask * 1.3:
        print(f"  → mask 让 completion 平均长 {avg_len_mask/avg_len_nomask:.1f}x")

    print("\n" + "=" * 80)
    print(f"Done. 关键数据已 print 在上方.")
    print(f"Mask 真因可从 'AUTO-DIAGNOSIS' 段直接判断:")
    print(f"  - SECTION_2 anchor=0 → 模型在 SECTION_1 阶段就 degenerate")
    print(f"  - SECTION_2 anchor>0 + dup 高 → 模型进了 SECTION_2 但 mask logic 把它逼坏")
    print(f"  - mask completion 长度跟 nomask 接近 + 没 EOS → forbid_eos 没真生效 (attach 失败)")
    print("=" * 80)


if __name__ == "__main__":
    main()
