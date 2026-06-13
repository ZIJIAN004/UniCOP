"""
SFT 训练侧 <think> 格式 mask 验证 (上机跑全量 SFT 前的 preflight)。

为什么需要: train_sft_stage2.py 用 DataCollatorForCompletionOnlyLM 靠"response_template
子序列匹配"来切 prompt/completion 边界。如果基座 tokenizer 把 <think> 当普通 BPE,
`\n<think>` 的 \n 可能与 < 合并, 导致 response_template (`...assistant\n`) 在全序列里
对不上 → collator 找不到 → 整条样本全 mask → 不报错但 loss=0 白训。

本脚本完整复现 train_sft_stage2.py 的 instruct 拼接逻辑, 对前 N 条真实样本:
  1. 确认 <think>/</think> 是否 special token + 其 id
  2. 渲染 prompt(+手动补 <think>\n) + completion, 拼成训练 text
  3. 用同一个 collator 跑 mask, 检查:
       - response_template 是否被找到 (找不到 → collator 会 warn + 整条 mask)
       - 被训练区(label != -100) 解码出来是否 = "<think>\n...reasoning...</think>route<eos>"
       - <think>/</think>/eos 是否都落在被训练区 (模型才学得到输出它们)
  4. 报 PASS/FAIL 计数

用法 (zhihan, conda activate unicop):
    cd /Data04/yangzhihan/lzj/UniCOP
    export BASE_MODEL_TYPE=qwen3_instruct
    source paths.sh
    python UniCOP-Distill/check_sft_think_masking.py \
        --model "$BASE_MODEL" \
        --data UniCOP-Distill/data/chains_template_cvrp20.jsonl \
        --filter_problems cvrp --filter_sizes 20 --n_samples 3
"""

import argparse
import json
import sys

from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

# 与 train_sft_stage2.py 完全一致的 posthoc 剥离标记
_POSTHOC_SYSTEM_MARKER = "\n\nYour output MUST start with <think>"
_POSTHOC_USER_MARKER = "\n\nTarget solution ("


def strip_posthoc_system(system: str) -> str:
    idx = system.find(_POSTHOC_SYSTEM_MARKER)
    return system[:idx] if idx != -1 else system


def strip_posthoc_user(user: str) -> str:
    idx = user.find(_POSTHOC_USER_MARKER)
    return user[:idx] if idx != -1 else user


def detect_response_template(tokenizer) -> str:
    """与 train_sft_stage2.py._detect_response_template 一致。"""
    msgs = [{"role": "user", "content": "Hi"}]
    without_gp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    with_gp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return with_gp[len(without_gp):]


def build_train_text(tokenizer, system, user, output, probe_ends_with_think):
    """复现 train_sft_stage2.load_sft_dataset 的单条拼接, 返回 (prompt_text, completion_text)。"""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    if not probe_ends_with_think:
        prompt_text += "<think>\n"

    output_stripped = output.lstrip()
    if output_stripped.startswith("<think>"):
        output_stripped = output_stripped[len("<think>"):].lstrip("\n")
    completion_text = output_stripped + tokenizer.eos_token
    return prompt_text, completion_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--filter_problems", nargs="+", default=None)
    ap.add_argument("--filter_sizes", type=int, nargs="+", default=None)
    ap.add_argument("--n_samples", type=int, default=3)
    args = ap.parse_args()

    print(f"加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if getattr(tok, "add_bos_token", False):
        tok.add_bos_token = False
    # 复现训练端 pad 设置 (pad != eos)
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        for cand in ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>", "<|pad|>"]:
            tid = tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tok.eos_token_id:
                tok.pad_token = cand
                break
    tok.padding_side = "right"

    # ── 1. <think>/</think> 是否 special token ────────────────────────────
    think_id = tok.convert_tokens_to_ids("<think>")
    think_close_id = tok.convert_tokens_to_ids("</think>")
    unk = tok.unk_token_id
    print("\n[1] <think>/</think> token 检查")
    print(f"    <think>  → id={think_id}  ({'OK' if think_id not in (None, unk) else '⚠️ 未注册/UNK'})")
    print(f"    </think> → id={think_close_id}  ({'OK' if think_close_id not in (None, unk) else '⚠️ 未注册/UNK'})")
    # 用编码 '\n<think>' 看 \n 是否与 < 合并 (合并=response_template 边界可能错位)
    nl_think_ids = tok.encode("\n<think>", add_special_tokens=False)
    print(f"    encode('\\n<think>') = {nl_think_ids}")
    print(f"    (若 <think> 独立成 id={think_id}, 边界安全; 若被拆成多 BPE 且与 \\n 粘连, 需警惕)")

    # ── 2. probe + response_template ─────────────────────────────────────
    probe = tok.apply_chat_template(
        [{"role": "system", "content": "p"}, {"role": "user", "content": "p"}],
        tokenize=False, add_generation_prompt=True,
    )
    probe_ends_with_think = probe.rstrip().endswith("<think>")
    resp_tmpl = detect_response_template(tok)
    resp_ids = tok.encode(resp_tmpl, add_special_tokens=False)
    print("\n[2] response_template 检查")
    print(f"    probe 末尾带 <think>: {probe_ends_with_think} "
          f"({'thinking 模型路径' if probe_ends_with_think else 'instruct 路径(手动补 <think>\\n)'})")
    print(f"    response_template = {resp_tmpl!r}")
    print(f"    response_template_ids = {resp_ids}")

    collator = DataCollatorForCompletionOnlyLM(response_template=resp_ids, tokenizer=tok)

    # ── 3. 逐样本验证 mask ───────────────────────────────────────────────
    print(f"\n[3] 逐样本 mask 验证 (前 {args.n_samples} 条匹配样本)")
    passed = 0
    checked = 0
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.filter_problems and r.get("problem_type") not in args.filter_problems:
                continue
            if args.filter_sizes and r.get("n") not in args.filter_sizes:
                continue
            output = r.get("output", "")
            if not output or not output.strip():
                continue

            system = strip_posthoc_system(r["prompt"]["system"])
            user = strip_posthoc_user(r["prompt"]["user"])
            prompt_text, completion_text = build_train_text(
                tok, system, user, output, probe_ends_with_think)
            full_text = prompt_text + completion_text

            input_ids = tok.encode(full_text, add_special_tokens=False)
            batch = collator([{"input_ids": input_ids}])
            labels = batch["labels"][0].tolist()

            n_train = sum(1 for x in labels if x != -100)
            found = n_train > 0  # collator 找不到 response_template 时整条 mask → n_train==0

            checked += 1
            print(f"\n  ── 样本 #{checked} (problem={r.get('problem_type')}, n={r.get('n')}) ──")
            if not found:
                print("    ❌ FAIL: 被训练 token 数 = 0 → collator 没找到 response_template, "
                      "整条样本全 mask, 不计 loss! (think 边界对不上)")
                if checked >= args.n_samples:
                    break
                continue

            train_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
            train_text = tok.decode(train_ids, skip_special_tokens=False)

            has_open = "<think>" in train_text
            has_close = "</think>" in train_text
            has_route = ("Route" in train_text) or ("route" in train_text)
            ends_eos = train_ids[-1] == tok.eos_token_id

            print(f"    被训练 token 数: {n_train} / 总 {len(input_ids)}")
            print(f"    被训练区开头 60 字: {train_text[:60]!r}")
            print(f"    被训练区结尾 60 字: {train_text[-60:]!r}")
            print(f"    含 <think>: {has_open}  含 </think>: {has_close}  "
                  f"含 Route: {has_route}  以 eos 结尾: {ends_eos}")

            ok = has_open and has_close and has_route and ends_eos
            if ok:
                print("    ✅ PASS: think 格式完整落在被训练区, mask 正确")
                passed += 1
            else:
                print("    ⚠️ WARN: 被训练区缺少 <think>/</think>/Route/eos 之一, 请人工核对上面解码")

            if checked >= args.n_samples:
                break

    print(f"\n{'='*60}")
    print(f"  验证完成: {passed}/{checked} 条 PASS")
    if passed == checked and checked > 0:
        print("  ✅ 全部通过 → think 格式能被正确识别和训练, 可放心跑全量 SFT")
    else:
        print("  ❌ 有样本未通过 → 切勿直接跑全量 SFT, 先排查上面的边界/mask 问题")
    print(f"{'='*60}")

    # 非零退出码: 供 run 脚本 fail-fast 门禁
    if checked == 0 or passed != checked:
        sys.exit(1)


if __name__ == "__main__":
    main()
