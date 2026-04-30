"""从 Stage 1 solutions 中采样，包上短 <think> 壳，混入 Stage 2 chains 生成混合训练集。

用法:
    python mix_replay_data.py
    python mix_replay_data.py --num_replay 500 --problem cvrp --size 20
    python mix_replay_data.py --solutions data/solutions.jsonl --chains data/chains_v3_clean.jsonl

输出: data/chains_v3_mixed.jsonl（供 train_sft_stage2.py 使用）
"""

import argparse
import json
import os
import random

THINK_TEMPLATES = [
    "I need to plan vehicle routes for this CVRP instance. Let me first check the"
    " vehicle capacity and customer demands. Each route must start and end at the depot"
    " (node 0), and the total demand on any route cannot exceed the capacity."
    " I should group geographically close customers together to minimize travel distance,"
    " while carefully tracking the remaining capacity on each route."
    " Let me also verify that every customer is visited exactly once across all routes.",

    "Let me analyze the customer locations and demands. The key constraints are:"
    " (1) each customer must be visited exactly once, (2) each route starts and ends"
    " at the depot, (3) the sum of demands on each route must not exceed vehicle capacity."
    " I'll start by identifying clusters of nearby customers, then assign them to routes"
    " while monitoring cumulative demand. After constructing the routes, I need to double-check"
    " that no customer is missed or visited twice.",

    "For this capacitated vehicle routing problem, I'll plan the routes step by step."
    " First, I need to note the depot location and vehicle capacity. Then I'll look at"
    " the customer positions and demands to find natural groupings. I should be careful"
    " not to overload any single route — if adding a customer would exceed capacity,"
    " I'll start a new route. Finally, I'll verify all customers are covered and each"
    " route returns to the depot.",

    "Let me think about how to construct feasible routes. The depot is at node 0, and"
    " I have customers with known locations and demands. My approach: scan the customers,"
    " group nearby ones into routes, and ensure each route's total demand stays within"
    " the vehicle capacity. I need to be systematic — first plan which customers go on"
    " which route, then determine the visit order to minimize distance. Every customer"
    " must appear in exactly one route.",

    "I'll solve this CVRP by constructing routes one at a time. Starting from the depot,"
    " I'll visit nearby customers and keep a running total of the demand served."
    " When the next customer would exceed the remaining capacity, I'll return to the depot"
    " and start a new route. I need to make sure I don't forget any customer."
    " Let me also consider the spatial layout to avoid unnecessarily long detours."
    " After building all routes, I'll verify feasibility: coverage, capacity, and depot returns.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", default="data/solutions.jsonl")
    parser.add_argument("--chains", default="data/chains_v3_clean.jsonl")
    parser.add_argument("--output", default="data/chains_v3_mixed.jsonl")
    parser.add_argument("--problem", default="cvrp")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num_replay", type=int, default=1000,
                        help="从 Stage 1 数据中采样的回放条数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 读取 Stage 1 solutions ──────────────────────────────────────────
    solutions = []
    with open(args.solutions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("problem_type") == args.problem and r.get("n") == args.size:
                solutions.append(r)

    print(f"Stage 1 solutions ({args.problem} n={args.size}): {len(solutions)} 条")

    if len(solutions) == 0:
        print("ERROR: 没有找到匹配的 Stage 1 数据")
        return

    num_replay = min(args.num_replay, len(solutions))
    sampled = random.sample(solutions, num_replay)
    print(f"采样 {num_replay} 条作为回放数据")

    # ── 包装成 chains 格式 ──────────────────────────────────────────────
    replay_records = []
    for r in sampled:
        think_text = random.choice(THINK_TEMPLATES)
        output = f"<think>\n{think_text}\n</think>\n{r['solution']}"

        replay_records.append({
            "problem_type": r["problem_type"],
            "n":            r["n"],
            "prompt":       r["prompt"],
            "output":       output,
            "source":       "replay",
        })

    # ── 读取 Stage 2 chains ─────────────────────────────────────────────
    chains = []
    with open(args.chains, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("problem_type") == args.problem and r.get("n") == args.size:
                r["source"] = "gemini"
                chains.append(r)

    print(f"Stage 2 chains ({args.problem} n={args.size}): {len(chains)} 条")

    # ── 混合并打乱 ─────────────────────────────────────────────────────
    mixed = chains + replay_records
    random.shuffle(mixed)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_gemini = sum(1 for r in mixed if r.get("source") == "gemini")
    n_replay = sum(1 for r in mixed if r.get("source") == "replay")
    print(f"\n混合数据集: {args.output}")
    print(f"  Gemini chains: {n_gemini}")
    print(f"  Replay:        {n_replay}")
    print(f"  总计:          {len(mixed)}")
    print(f"  比例:          replay:gemini = {n_replay/max(n_gemini,1):.1f}:1")


if __name__ == "__main__":
    main()
