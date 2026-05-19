"""Smoke test for evaluate.py 在 Qwen3-Thinking 上的兼容性。

7 个层面 (从 token 到端到端):
    1. tokenizer.decode 行为: skip_special_tokens=True vs False 时
       <think>/</think> 是否保留 (Qwen3 是 special token, R1-Distill 是 BPE)
    2. apply_chat_template 渲染含 add_generation_prompt 的 prompt
    3. HF generate 端到端 (单 prompt, do_sample=True)
    4. 生成长度合理 (在 max_new_tokens 内自然停止, 非截断)
    5. completion decode 后含完整 <think>...</think>{answer} 结构
    6. 结构性 token (<|im_end|>/<|endoftext|>) 不残留在 completion 字符串
    7. problems/cvrp.py 的 get_tour_distance + is_feasible 能 parse 出结果

需要 1 GPU. 用法:
    sbatch UniCOP-Reason/submit_smoke_eval.sh
或登录节点 (慢):
    export BASE_MODEL_TYPE=qwen3_thinking && source ../paths.sh
    python smoke_test_eval.py --model "$BASE_MODEL"
"""

import argparse
import os
import sys
import time

import numpy as np

PASS_TAG = "[PASS]"
FAIL_TAG = "[FAIL]"
WARN_TAG = "[WARN]"
INFO_TAG = "[INFO]"

failures = []
warnings_ = []


def section(title: str):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = PASS_TAG if ok else (WARN_TAG if warn_only else FAIL_TAG)
    print(f"  {tag} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        (warnings_ if warn_only else failures).append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.environ.get("BASE_MODEL", ""))
    parser.add_argument("--problem", type=str, default="cvrp")
    parser.add_argument("--problem_size", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.model:
        print("ERROR: 需要 --model 或 export BASE_MODEL=...")
        sys.exit(1)

    print(f"模型路径:        {args.model}")
    print(f"BASE_MODEL_TYPE: {os.environ.get('BASE_MODEL_TYPE', '(未设)')}")
    print(f"GEN_TEMPERATURE: {os.environ.get('GEN_TEMPERATURE', '(未设)')}")
    print(f"problem:         {args.problem} (size={args.problem_size})")
    print(f"max_new_tokens:  {args.max_new_tokens}")
    print(f"temperature:     {args.temperature}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from problems import get_problem

    # ── 0. 加载 ────────────────────────────────────────────────────────────
    section("0. 加载 tokenizer + 模型")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"  pad_token_id: {tokenizer.pad_token_id} ({tokenizer.pad_token!r})")
    print(f"  eos_token_id: {tokenizer.eos_token_id} ({tokenizer.eos_token!r})")
    print(f"  padding_side: {tokenizer.padding_side}")

    print(f"  加载模型...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ 加载完成 ({time.time()-t0:.1f}s)")
    print(f"  model.dtype:  {model.dtype}")
    print(f"  device:       {next(model.parameters()).device}")

    # ── 1. tokenizer.decode 对 <think>/</think> 的行为 ──────────────────────
    section("1. tokenizer.decode: <think>/</think> 是否被 skip_special_tokens 影响")
    sample = "<think>thinking text</think>\nfinal answer"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    dec_true = tokenizer.decode(ids, skip_special_tokens=True)
    dec_false = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"  原文:             {sample!r}")
    print(f"  decode(True):     {dec_true!r}")
    print(f"  decode(False):    {dec_false!r}")
    think_keep_true = "</think>" in dec_true
    think_keep_false = "</think>" in dec_false
    print(f"  </think> 在 decode(True):  {think_keep_true}")
    print(f"  </think> 在 decode(False): {think_keep_false}")
    check("decode(skip_special_tokens=False) 保住 </think>",
          think_keep_false,
          "evaluate.py 修复后必须为 True; 若 False 整个 think/answer 切分会失效")
    if not think_keep_true:
        print(f"  {INFO_TAG} Qwen3 路径确认: </think> 是 special token, "
              f"True 时被剥, False 时保留")

    # ── 2. chat_template 渲染 ───────────────────────────────────────────────
    section("2. apply_chat_template 渲染")
    prob = get_problem(args.problem)
    rng = np.random.default_rng(args.seed)
    instance = prob.generate_instance(args.problem_size, rng)
    chat_msgs = prob.build_prompt(instance)
    chat_text = tokenizer.apply_chat_template(
        chat_msgs, tokenize=False, add_generation_prompt=True,
    )
    print(f"  messages 数量: {len(chat_msgs)} (role: {[m['role'] for m in chat_msgs]})")
    print(f"  渲染文本末尾 80 字: {chat_text[-80:]!r}")
    ends_with_think = chat_text.rstrip().endswith("<think>")
    check("Qwen3 chat_template 末尾自动 prepend <think>",
          ends_with_think,
          "若 False, 模型不会自然进入 thinking 模式")

    # ── 3. HF generate 端到端 ──────────────────────────────────────────────
    section("3. HF generate 端到端 (do_sample=True, T={})".format(args.temperature))
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    print(f"  prompt token 数: {prompt_len}")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0
    completion_ids = out[0][prompt_len:]
    n_gen = len(completion_ids)
    print(f"  ✓ generate 完成 ({gen_time:.1f}s, {n_gen} tokens, "
          f"{n_gen/gen_time:.1f} tok/s)")

    # ── 4. 生成长度合理 ────────────────────────────────────────────────────
    section("4. 生成长度 / 自然停止")
    is_truncated = (n_gen >= args.max_new_tokens)
    check("生成自然停止 (未顶到 max_new_tokens)",
          not is_truncated,
          f"n_gen={n_gen}, max={args.max_new_tokens}; 顶到说明 max 偏小或模型不停",
          warn_only=is_truncated)
    # EOS 验证: 最后一个 token 应该是某个 EOS
    last_token = completion_ids[-1].item()
    eos_candidates = []
    if isinstance(tokenizer.eos_token_id, int):
        eos_candidates.append(tokenizer.eos_token_id)
    elif isinstance(tokenizer.eos_token_id, list):
        eos_candidates.extend(tokenizer.eos_token_id)
    eos_candidates += [151645, 151643]  # Qwen3 双 EOS
    eos_candidates = list(set(eos_candidates))
    last_is_eos = last_token in eos_candidates
    print(f"  最后一个 token: {last_token} ({tokenizer.decode([last_token])!r})")
    print(f"  EOS 候选: {eos_candidates}")
    check("生成在 EOS 处停止 (非截断)",
          last_is_eos or is_truncated,
          f"last_token={last_token}, 不在 EOS 列表",
          warn_only=is_truncated)  # 已截断的话不算 fail

    # ── 5. decode 含完整 <think>...</think>{answer} 结构 ───────────────────
    section("5. decode 后 completion 含完整 think/answer 结构")
    # 模拟 evaluate.py 修复后的 decode 逻辑
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    for tok in ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"):
        completion = completion.replace(tok, "")

    print(f"  completion 总长度: {len(completion)} 字符")
    print(f"  开头 300 字: {completion[:300]!r}")
    print(f"  末尾 300 字: {completion[-300:]!r}")

    has_close = "</think>" in completion
    check("completion 含 </think>",
          has_close,
          "若 False, parse 必然失败 (评估代码靠 </think> 切 thinking/answer)")
    if has_close:
        close_idx = completion.rfind("</think>")
        thinking_part = completion[:close_idx]
        answer_part = completion[close_idx + len("</think>"):]
        print(f"  thinking 段 (</think> 之前): {len(thinking_part)} 字符")
        print(f"  answer   段 (</think> 之后): {len(answer_part)} 字符")
        check("thinking 段 > 200 字符",
              len(thinking_part) > 200,
              f"实际 {len(thinking_part)}")
        check("answer 段 > 20 字符",
              len(answer_part) > 20,
              f"实际 {len(answer_part)}")

    # ── 6. 结构性 special token 不残留 ─────────────────────────────────────
    section("6. 结构性 special token 不残留")
    leaked = []
    for tok in ("<|im_start|>", "<|im_end|>", "<|endoftext|>",
                "<｜end▁of▁sentence｜>", "<|begin_of_text|>"):
        if tok in completion:
            leaked.append(tok)
    check("结构性 special token 已被 replace 清掉",
          not leaked,
          f"残留: {leaked}", warn_only=True)

    # ── 7. CVRP / TSP parse 验证 ───────────────────────────────────────────
    section("7. problems/{}.py parse 验证".format(args.problem))
    distance = prob.get_tour_distance(completion, instance)
    feasible = prob.is_feasible(completion, instance)
    print(f"  get_tour_distance(): {distance}")
    print(f"  is_feasible():       {feasible}")
    check("parse 成功 (distance 非 None)",
          distance is not None,
          "evaluate.py 跑 100 条会显示 'parsed=X/100', 这里单条必须成功才有意义",
          warn_only=True)  # 模型实际质量决定, 不强制
    if distance is not None:
        check("解可行 (容量/覆盖约束满足)",
              feasible,
              f"distance={distance:.4f} 但不可行说明 parse 拿到了违规解",
              warn_only=True)

    # ── 8. 二次生成: 验证采样真的随机化 ─────────────────────────────────────
    section("8. 采样随机性 (do_sample=True 真的开启)")
    torch.manual_seed(args.seed + 1)
    with torch.no_grad():
        out2 = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95, top_k=20,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion2_ids = out2[0][prompt_len:]
    same_as_first = torch.equal(completion_ids[:256], completion2_ids[:256])
    print(f"  两次生成前 256 token 完全相同: {same_as_first}")
    check("采样产生不同输出 (do_sample 生效)",
          not same_as_first,
          "若两次完全一样, do_sample=True 没生效 (可能 temperature/seed bug)",
          warn_only=True)

    # ── 结尾汇总 ────────────────────────────────────────────────────────────
    section("总结")
    if failures:
        print(f"  {FAIL_TAG} {len(failures)} 项失败:")
        for f in failures:
            print(f"      - {f}")
    else:
        print(f"  {PASS_TAG} 全部硬性检查通过")
    if warnings_:
        print(f"  {WARN_TAG} {len(warnings_)} 项警告:")
        for w in warnings_:
            print(f"      - {w}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
