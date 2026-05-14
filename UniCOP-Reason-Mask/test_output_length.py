"""
输出长度测试脚本：系统评估模型在不同问题类型和节点规模下的 completion 长度分布。

用途：
  - 确定合理的 max_completion_length 训练超参
  - 对比 enable_thinking=True / False 的长度差异
  - 评估截断率，判断当前设置是否合理

运行示例：
  python test_output_length.py --model /home/ntu/lzj/UniCOP/UniCOP-Reason/model/Qwen3-4B
  python test_output_length.py --model /path/to/model --max_length 2048 --num_samples 10 --no_thinking
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from problems import get_problem

# ── 测试矩阵 ──────────────────────────────────────────────────────────────────
PROBLEM_TYPES = ["tsp", "tsptw", "tspdl", "cvrp", "vrptw"]
NODE_SIZES    = [20, 50, 100]


def run_single(model, tokenizer, problem_type, n, num_samples, max_length, enable_thinking, seed=42):
    """对一个 (problem_type, n) 组合跑 num_samples 次推理，返回统计结果。"""
    problem = get_problem(problem_type)
    rng = np.random.default_rng(seed=seed)

    lengths   = []
    truncated = 0

    for i in range(num_samples):
        instance = problem.generate_instance(n, rng)
        prompt   = problem.build_prompt(instance)

        # 应用 chat template
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            chat_text = tokenizer.apply_chat_template(prompt, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            # 部分模型 tokenizer 不支持 enable_thinking 参数
            chat_text = tokenizer.apply_chat_template(prompt, **kwargs)

        inputs     = tokenizer(chat_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        comp_len = output.shape[1] - prompt_len
        is_trunc = comp_len >= max_length

        lengths.append(comp_len)
        if is_trunc:
            truncated += 1

        status = "TRUNCATED" if is_trunc else "ok"
        print(f"    sample {i+1:2d}/{num_samples}: {comp_len:5d} tok  [{status}]  ({elapsed:.1f}s)")

    trunc_rate = truncated / num_samples
    return {
        "mean":          float(np.mean(lengths)),
        "std":           float(np.std(lengths)),
        "min":           int(np.min(lengths)),
        "max":           int(np.max(lengths)),
        "truncated":     truncated,
        "trunc_rate":    trunc_rate,
        "lengths":       lengths,
    }


def print_table(results, node_sizes, problem_types):
    """打印汇总表格：mean ± std (trunc%)"""
    col_w = 22
    header = f"{'Problem':<10}" + "".join(f"{'n='+str(n):>{col_w}}" for n in node_sizes)
    print("\n" + "=" * (10 + col_w * len(node_sizes)))
    print("  输出长度汇总  mean±std (截断率)")
    print("=" * (10 + col_w * len(node_sizes)))
    print(header)
    print("-" * (10 + col_w * len(node_sizes)))

    for pt in problem_types:
        row = f"{pt:<10}"
        for n in node_sizes:
            if n in results.get(pt, {}):
                r    = results[pt][n]
                cell = f"{r['mean']:.0f}±{r['std']:.0f} ({r['trunc_rate']:.0%})"
            else:
                cell = "N/A"
            row += f"{cell:>{col_w}}"
        print(row)

    print("=" * (10 + col_w * len(node_sizes)))
    print("格式：mean±std（截断率），截断率=0% 说明 max_length 设置充足\n")


def main():
    parser = argparse.ArgumentParser(description="系统测试模型输出长度分布")
    parser.add_argument("--model",       type=str,  required=True,
                        help="模型路径")
    parser.add_argument("--max_length",  type=int,  default=4096,
                        help="生成最大 token 数（截断上限，默认 4096）")
    parser.add_argument("--num_samples", type=int,  default=5,
                        help="每个 (problem, n) 组合的样本数（默认 5）")
    parser.add_argument("--problems",    type=str,  nargs="+", default=PROBLEM_TYPES,
                        choices=PROBLEM_TYPES,
                        help="要测试的问题类型（默认全部）")
    parser.add_argument("--sizes",       type=int,  nargs="+", default=NODE_SIZES,
                        help="要测试的节点规模（默认 20 50 100）")
    parser.add_argument("--no_thinking", action="store_true",
                        help="禁用 Qwen3 thinking 模式（enable_thinking=False）")
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--output",      type=str,  default="output_length_report.json",
                        help="结果保存路径（默认 output_length_report.json）")
    args = parser.parse_args()

    enable_thinking = not args.no_thinking
    mode_str = "thinking=OFF" if args.no_thinking else "thinking=ON"

    print(f"\n{'='*60}")
    print(f"  输出长度测试")
    print(f"  模型:      {args.model}")
    print(f"  模式:      {mode_str}")
    print(f"  max_length:{args.max_length}")
    print(f"  样本数:    {args.num_samples} × {len(args.problems)} 问题 × {len(args.sizes)} 规模")
    print(f"{'='*60}\n")

    # 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("模型加载完成\n")

    # 遍历测试矩阵
    results = {}
    total_combos = len(args.problems) * len(args.sizes)
    combo_idx    = 0

    for pt in args.problems:
        results[pt] = {}
        for n in args.sizes:
            combo_idx += 1
            print(f"[{combo_idx}/{total_combos}] {pt}  n={n}  ({mode_str})")

            stats = run_single(
                model, tokenizer,
                problem_type=pt,
                n=n,
                num_samples=args.num_samples,
                max_length=args.max_length,
                enable_thinking=enable_thinking,
                seed=args.seed,
            )
            results[pt][n] = stats

            print(f"  → mean={stats['mean']:.0f}  std={stats['std']:.0f}"
                  f"  min={stats['min']}  max={stats['max']}"
                  f"  截断率={stats['trunc_rate']:.0%}\n")

    # 打印汇总表格
    print_table(results, args.sizes, args.problems)

    # 保存结果
    report = {
        "model":          args.model,
        "max_length":     args.max_length,
        "num_samples":    args.num_samples,
        "enable_thinking": enable_thinking,
        "results":        results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
