"""
诊断脚本：检查 SFT merged 模型在 TSP/VRPTW 上的 terminal reward 四维分布。
专门排查 R_coverage 偏低的根因。

用法（单卡）：
    python diag_coverage.py --model_path /path/to/merged_model \
        --problem tsp --problem_size 20 --num_test 20 --num_samples 8

输出：
    1. 四维 terminal reward 统计（parse / coverage / constraint / format）
    2. coverage=0 的 completion 详细诊断（缺哪些客户、有无重复、路线长度）
    3. 原始模型输出示例（方便人工检查）
"""

import argparse
import json
import sys
import urllib.request

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from problems import get_problem
from terminal_reward import compute_terminal_components
from utils.parse import parse_single_route, parse_multi_route

_SCKEY = "SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

SINGLE_ROUTE = {"tsp", "tsptw", "tspdl"}


def notify(title, desp=""):
    try:
        data = urllib.parse.urlencode({"title": title, "desp": desp}).encode()
        urllib.request.urlopen(
            f"https://sctapi.ftqq.com/{_SCKEY}.send", data, timeout=10
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--problem", type=str, default="tsptw", choices=["tsp", "cvrp", "tsptw", "vrptw"])
    parser.add_argument("--problem_size", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pt = args.problem
    n = args.problem_size
    is_multi = pt in ("cvrp", "vrptw")

    print(f"问题: {pt}  n={n}  样本: {args.num_test} × {args.num_samples}")
    print(f"模型: {args.model_path}")
    print(f"temperature={args.temperature}  max_new_tokens={args.max_new_tokens}")

    # ── 加载模型 ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        for cand in ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>"]:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
                tokenizer.pad_token = cand
                break
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ── 生成实例 ──────────────────────────────────────────────────────
    prob = get_problem(pt)
    rng = np.random.default_rng(args.seed)
    instances = [prob.generate_instance(n, rng) for _ in range(args.num_test)]
    prompts = [prob.build_prompt(inst) for inst in instances]

    # ── 逐实例生成 & 诊断 ────────────────────────────────────────────
    all_parse, all_cov, all_con, all_fmt = [], [], [], []
    diag_details = []

    for i, (inst, prompt) in enumerate(zip(instances, prompts)):
        chat_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )

        for s in range(args.num_samples):
            comp_ids = outputs[s][prompt_len:]
            comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
            comp_tokens = len(comp_ids)
            truncated = comp_tokens >= args.max_new_tokens

            c = compute_terminal_components(comp_text, inst, pt)
            all_parse.append(c["parse"])
            all_cov.append(c["coverage"])
            all_con.append(c["constraint"])
            all_fmt.append(c["format"])

            # coverage=0 时做详细诊断
            if c["coverage"] < 1.0:
                if is_multi:
                    parsed = parse_multi_route(comp_text, n)
                    if parsed:
                        visits = [v for r in parsed for v in r if v != 0]
                    else:
                        visits = []
                else:
                    parsed = parse_single_route(comp_text, n)
                    visits = [v for v in parsed if v != 0] if parsed else []

                unique = set(visits)
                missing = sorted(set(range(1, n + 1)) - unique)
                duplicates = len(visits) - len(unique)

                # 取 </think> 后的答案段
                think_end = comp_text.rfind("</think>")
                answer = comp_text[think_end:think_end + 500] if think_end != -1 else comp_text[-500:]

                detail = {
                    "inst": i, "sample": s,
                    "parse": c["parse"], "coverage": c["coverage"],
                    "truncated": truncated, "tokens": comp_tokens,
                    "num_visits": len(visits), "num_unique": len(unique),
                    "missing": missing, "duplicates": duplicates,
                    "parsed_route": str(parsed)[:200] if parsed else "None",
                    "answer_tail": answer[:300],
                }
                diag_details.append(detail)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{args.num_test}] parse={np.mean(all_parse):.2%} "
                  f"cov={np.mean(all_cov):.2%} con={np.mean(all_con):.3f} "
                  f"fmt={np.mean(all_fmt):.2%}")

    # ── 汇总 ─────────────────────────────────────────────────────────
    total = len(all_parse)
    print(f"\n{'='*60}")
    print(f"  {pt.upper()} n={n}  |  {args.num_test} 实例 × {args.num_samples} 采样 = {total}")
    print(f"  R_parse_rate:      {np.mean(all_parse):.2%}")
    print(f"  R_coverage_rate:   {np.mean(all_cov):.2%}")
    print(f"  R_constraint_mean: {np.mean(all_con):.4f}")
    print(f"  R_format_mean:     {np.mean(all_fmt):.2%}")
    print(f"{'='*60}")

    # ── coverage=0 的详细诊断 ─────────────────────────────────────────
    num_cov_fail = sum(1 for x in all_cov if x < 1.0)
    print(f"\ncoverage=0 的 completion: {num_cov_fail}/{total} ({num_cov_fail/total:.1%})")

    if diag_details:
        # 分类统计
        parse_fail = [d for d in diag_details if d["parse"] < 1.0]
        parse_ok_cov_fail = [d for d in diag_details if d["parse"] >= 1.0]

        print(f"  其中 parse 失败:       {len(parse_fail)}")
        print(f"  其中 parse 成功但覆盖不全: {len(parse_ok_cov_fail)}")

        if parse_ok_cov_fail:
            miss_counts = [len(d["missing"]) for d in parse_ok_cov_fail]
            dup_counts = [d["duplicates"] for d in parse_ok_cov_fail]
            visit_counts = [d["num_visits"] for d in parse_ok_cov_fail]
            trunc_count = sum(1 for d in parse_ok_cov_fail if d["truncated"])
            print(f"\n  parse OK 但 coverage=0 的统计:")
            print(f"    缺失客户数: mean={np.mean(miss_counts):.1f}  "
                  f"min={min(miss_counts)}  max={max(miss_counts)}")
            print(f"    重复客户数: mean={np.mean(dup_counts):.1f}  "
                  f"min={min(dup_counts)}  max={max(dup_counts)}")
            print(f"    实际访问数: mean={np.mean(visit_counts):.1f}  (期望={n})")
            print(f"    截断占比:   {trunc_count}/{len(parse_ok_cov_fail)}")

            # 缺失客户频次统计
            from collections import Counter
            miss_freq = Counter()
            for d in parse_ok_cov_fail:
                miss_freq.update(d["missing"])
            if miss_freq:
                print(f"\n  最常缺失的客户节点 (top 10):")
                for node, cnt in miss_freq.most_common(10):
                    print(f"    Node {node}: 缺失 {cnt} 次")

        # 打印前 5 条诊断详情
        print(f"\n  前 5 条 coverage=0 详情:")
        for d in diag_details[:5]:
            print(f"  ── inst={d['inst']} sample={d['sample']} ──")
            print(f"    parse={d['parse']} trunc={d['truncated']} tokens={d['tokens']}")
            print(f"    visits={d['num_visits']}  unique={d['num_unique']}  "
                  f"missing({len(d['missing'])}): {d['missing'][:15]}")
            print(f"    dups={d['duplicates']}")
            print(f"    route: {d['parsed_route']}")
            print(f"    answer: {d['answer_tail'][:200]}")

    # ── 通知 ──────────────────────────────────────────────────────────
    summary = (
        f"{pt.upper()} n={n}: parse={np.mean(all_parse):.0%} "
        f"cov={np.mean(all_cov):.0%} con={np.mean(all_con):.2f} "
        f"fmt={np.mean(all_fmt):.0%}"
    )
    notify(f"diag_coverage 完成: {summary}")
    print(f"\n完成。")


if __name__ == "__main__":
    main()
