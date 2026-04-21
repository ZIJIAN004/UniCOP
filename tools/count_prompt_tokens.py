"""统计各 problem × size 的 prompt token 数 (含 chat_template 渲染后)。

用法 (远程):
    cd /Data04/yangzhihan/lzj/UniCOP
    python tools/count_prompt_tokens.py \
        --model /Data04/yangzhihan/lzj/UniCOP-Reason.bak_/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

输出: 表格, 每格是 5 次采样的中位数。重点看 "chat" 列 (vLLM 实际拿到的 prompt 长度)。
"""
import argparse
import os
import sys

import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base model 路径 (含 tokenizer)")
    parser.add_argument("--reason_dir",
                        default="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason",
                        help="UniCOP-Reason 目录 (有 problems/ 子模块)")
    parser.add_argument("--samples", type=int, default=5,
                        help="每个 (problem, size) 的采样次数,取中位数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, args.reason_dir)
    from problems import get_problem

    print(f"loading tokenizer from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  model_max_length = {tok.model_max_length}")
    print(f"  pad={tok.pad_token_id}  eos={tok.eos_token_id}  pad==eos: {tok.pad_token_id == tok.eos_token_id}\n")

    problems = ["tsp", "cvrp", "vrptw", "tsptw"]
    sizes = [20, 50, 100]

    print(f"{'problem':<8} {'n':>4}  {'sys':>5}  {'user':>5}  {'sys+user':>9}  {'chat':>5}  "
          f"{'chat+compl=4096':>15}  {'chat+compl=8192':>15}")
    print("-" * 90)

    rng = np.random.default_rng(args.seed)
    results = []
    for ptype in problems:
        prob = get_problem(ptype)
        for n in sizes:
            rows = []
            for _ in range(args.samples):
                inst = prob.generate_instance(n, rng)
                prompt = prob.build_prompt(inst)
                sys_ids = tok.encode(prompt[0]["content"], add_special_tokens=False)
                user_ids = tok.encode(prompt[1]["content"], add_special_tokens=False)
                chat_txt = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                chat_ids = tok.encode(chat_txt, add_special_tokens=False)
                rows.append((len(sys_ids), len(user_ids), len(chat_ids)))
            sys_med = int(np.median([r[0] for r in rows]))
            user_med = int(np.median([r[1] for r in rows]))
            chat_med = int(np.median([r[2] for r in rows]))
            plus_4k = chat_med + 4096
            plus_8k = chat_med + 8192
            mark_4k = " ⚠️" if plus_4k > 5120 else ""
            mark_8k = " ⚠️" if plus_8k > 5120 else ""
            print(f"{ptype:<8} {n:>4}  {sys_med:>5}  {user_med:>5}  {sys_med + user_med:>9}  "
                  f"{chat_med:>5}  {plus_4k:>13}{mark_4k:>2}  {plus_8k:>13}{mark_8k:>2}")
            results.append((ptype, n, sys_med, user_med, chat_med))

    print()
    print("解读:")
    print(f"  chat           = 应用 chat_template 后的 prompt 总 token (vLLM 实际拿到的)")
    print(f"  chat+compl=4K  = 若 max_completion_length=4096, vLLM 需要的 max_model_len")
    print(f"  chat+compl=8K  = 若 max_completion_length=8192, 同上")
    print(f"  ⚠️             = 超过 auto_all.sh 当前 VLLM_MAX_MODEL_LEN=5120")
    max_chat = max(r[4] for r in results)
    print(f"\n  最长 prompt = {max_chat} tokens (来自 {[r for r in results if r[4] == max_chat][0][:2]})")


if __name__ == "__main__":
    main()
