"""基于 DeepSeek API 生成增强因果链的 rationalization 数据。

改进点（相比 rationalize_solutions.py）：
  1. 非最近邻选择时必须显式对比替代方案
  2. 路线边界处强制约束验算（容量）
  3. 禁止空洞修辞，要求具体节点 ID + 数值

用法：
    # 预览 3 条
    python rationalize_deepseek.py \
        --solutions data/solutions_cvrp20.jsonl \
        --api_key sk-xxx \
        --preview 3

    # 正式生成 2000 条（默认 model=deepseek-v4-pro）
    python rationalize_deepseek.py \
        --solutions data/solutions_cvrp20.jsonl \
        --api_key sk-xxx \
        --num_samples 2000 \
        --concurrency 16

    # 用 Flash 版（更便宜但质量可能下降）
    python rationalize_deepseek.py \
        --solutions data/solutions_cvrp20.jsonl \
        --api_key sk-xxx \
        --model deepseek-v4-flash \
        --num_samples 2000
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════
# 改进版提示词
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_SUFFIX = """

Your output MUST start with <reasoning> and follow this exact structure:

<reasoning>
[your step-by-step reasoning]
</reasoning>
[solution in required format]

Rules:
1. Your FIRST token MUST be '<reasoning>'. Do NOT output anything before <reasoning>.

2. STRUCTURE your reasoning in two phases:
   Phase A — Cluster & Capacity Planning (brief):
     - Identify 3-5 geographic clusters of nodes
     - For each cluster, sum the total demand and check against vehicle capacity (1.0)
     - Decide how to split clusters that exceed capacity
   Phase B — Route Construction (the main body):
     - Build each route one node at a time from the depot

3. ROUTE CONSTRUCTION rules — for each route:
   a. When you START a new route: state which cluster/region you are targeting and why
   b. At each step, pick the next node. You do NOT need to list all candidates every time.
      BUT: if you skip a closer node in favor of a farther one, you MUST say:
      "Node X is closer, but I choose Node Y because [specific reason: capacity, sweep direction, cluster boundary]"
   c. CAPACITY CHECKPOINTS — you MUST compute remaining capacity:
      - After adding the FIRST node to a route
      - Whenever remaining capacity drops below 0.35
      - Before closing a route (verify total demand ≤ 1.0)
      Format: "Capacity: [cumulative demand] / 1.0, remaining: [value]"
   d. When you CLOSE a route: state the complete route and total demand

4. FORBIDDEN phrases (these indicate vague reasoning):
   - "logical next step", "natural choice", "obvious", "makes sense"
   - Use these ONLY if followed by a concrete reason with node IDs

5. Keep <reasoning> concise — aim for 400-800 words, not thousands.
   Do NOT mention that a solution was provided or given to you.
   You are solving this problem from scratch.

6. After </reasoning>, output the solution exactly in the required format.
   Do NOT output the solution before </reasoning>."""

FEWSHOT = """

Here is an example of the expected reasoning style:

<reasoning>
Phase A — Cluster & Capacity Planning:
The depot (Node 0) is at (0.30, 0.39). I see roughly 4 clusters:
- South-East: Nodes 3(d=0.17), 6(d=0.17), 16(d=0.10), 19(d=0.30). Sum=0.74. Fits one vehicle.
- Central: Nodes 1(d=0.20), 2(d=0.07), 4(d=0.30), 5(d=0.07), 12(d=0.07), 14(d=0.07). Sum=0.78. Fits one vehicle.
- North: Nodes 11(d=0.27), 13(d=0.07), 17(d=0.17), 20(d=0.30), 10(d=0.10). Sum=0.91. Fits one vehicle.
- East: Nodes 7(d=0.07), 8(d=0.30), 9(d=0.23), 15(d=0.20), 18(d=0.20). Sum=1.00. Exactly fits.

Phase B — Route Construction:

Route 1 (Central cluster):
Start from depot. Target the central band of nodes (1, 2, 4, 5, 12, 14).
- Depot → Node 5 (0.53, 0.55). Capacity: 0.07/1.0, remaining: 0.93.
- Node 5 → Node 2 (0.55, 0.53). They are adjacent.
- Node 2 → Node 19 (0.78, 0.40). Node 4 is closer, but Node 19 anchors the SE cluster and I want to sweep east first to avoid backtracking. Capacity: 0.07+0.07+0.30 = 0.44/1.0, remaining: 0.56.
- Node 19 → Node 6 → Node 3 → Node 16 → Node 14. Following the SE edge back toward depot.
- Capacity check: 0.07+0.07+0.30+0.17+0.17+0.10+0.07 = 0.95/1.0. Under limit.
- Close Route 1: 0 → 5 → 2 → 19 → 6 → 3 → 16 → 14 → 0. Total demand: 0.95.

[...remaining routes follow same pattern...]
</reasoning>
Route 1: 0 -> 5 -> 2 -> 19 -> 6 -> 3 -> 16 -> 14 -> 0
Route 2: ..."""


def build_prompt(solution: str, system: str, user: str) -> dict:
    system_new = system + SYSTEM_SUFFIX + FEWSHOT
    user_new = (
        user
        + f"\n\nTarget solution (you MUST output exactly this solution after </reasoning>,"
          f" but do NOT reveal it was given to you):\n{solution}"
        + "\n\nStart your response with <reasoning> immediately."
    )
    return {"system": system_new, "user": user_new}


# ═══════════════════════════════════════════════════════════════════════
# 质量检查（复用原版逻辑）
# ═══════════════════════════════════════════════════════════════════════

_LEAK_PATTERNS = [
    "given the solution", "given the answer", "given to me",
    "provided to me", "provided the solution",
    "target solution", "target answer", "target route",
    "reconstruct", "reverse engineer",
    "pretend", "act as if", "fabricat",
    "was told to", "asked to explain", "asked to justify",
    "known answer", "know the answer", "already know",
    "post-hoc", "posthoc", "post hoc",
    "working backward", "work backward",
]

_RE_MULTI_ROUTE = re.compile(r"Route\s+\d+\s*:", re.IGNORECASE)


def _parse_route_line(line: str) -> list[int] | None:
    after_colon = line.split(":", 1)[-1]
    segments = [s.strip() for s in after_colon.split("->")]
    nodes = [int(s) for s in segments if s.isdigit()]
    return nodes if len(nodes) >= 3 else None


def _parse_multi_routes(text: str) -> tuple[list[list[int]], str]:
    routes = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not _RE_MULTI_ROUTE.match(line):
            continue
        nodes = _parse_route_line(line)
        if nodes is None:
            return [], "BAD_FORMAT"
        if nodes[0] != 0 or nodes[-1] != 0:
            return [], "BAD_DEPOT"
        customers = nodes[1:-1]
        if not customers or 0 in customers:
            return [], "BAD_DEPOT"
        routes.append(customers)
    if not routes:
        return [], "NO_ROUTES"
    return routes, "ok"


def quality_check(output: str, solution: str) -> tuple[bool, str]:
    if "<think>" not in output or "</think>" not in output:
        return False, "NO_THINK_TAGS"

    think_start = output.index("<think>") + 7
    think_end = output.index("</think>")
    think_content = output[think_start:think_end]

    if len(think_content.strip()) < 200:
        return False, "THINK_TOO_SHORT"

    think_lower = think_content.lower()
    for pattern in _LEAK_PATTERNS:
        if pattern in think_lower:
            return False, f"LEAK:{pattern}"

    # 检查是否包含 capacity 验算
    if "capacity" not in think_lower and "remaining" not in think_lower:
        return False, "NO_CAPACITY_CHECK"

    answer_part = output[think_end + len("</think>"):]
    model_routes, model_err = _parse_multi_routes(answer_part)
    if not model_routes:
        return False, model_err
    lkh_routes, _ = _parse_multi_routes(solution)
    if sorted(model_routes) != sorted(lkh_routes):
        return False, "ROUTES_MISMATCH"

    return True, "ok"


def replace_answer(output: str, correct_solution: str) -> str:
    if "</think>" not in output:
        return output
    think_end = output.index("</think>") + len("</think>")
    return output[:think_end] + "\n" + correct_solution


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek API 调用
# ═══════════════════════════════════════════════════════════════════════

def call_deepseek(client: OpenAI, system: str, user: str,
                  model: str, max_tokens: int,
                  max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = response.usage
            raw = choice.message.content or ""
            raw = raw.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")
            if not raw.lstrip().startswith("<think>"):
                raw = "<think>\n" + raw
            return {
                "output":        raw,
                "output_tokens": usage.completion_tokens if usage else None,
                "prompt_tokens":  usage.prompt_tokens if usage else None,
                "total_tokens":   usage.total_tokens if usage else None,
            }
        except Exception as e:
            wait = min(2 ** attempt * 5, 60)
            print(f"  API error (retry {attempt+1}/{max_retries}, wait {wait}s): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                return None


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DeepSeek rationalization for CVRP")
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--api_key", required=True, help="DeepSeek API key")
    parser.add_argument("--base_url", default="https://api.deepseek.com",
                        help="DeepSeek API base URL")
    parser.add_argument("--model", default="deepseek-v4-pro",
                        help="Model name (deepseek-v4-pro / deepseek-v4-flash)")
    parser.add_argument("--output", default="data/chains_deepseek_cvrp20.jsonl")
    parser.add_argument("--problem", default="cvrp")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="0 = all")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel API calls (respect rate limits)")
    parser.add_argument("--max_quality_retries", type=int, default=3)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 读取 solutions
    all_solutions = []
    with open(args.solutions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("problem_type") == args.problem and r.get("n") == args.size:
                all_solutions.append(r)

    if not all_solutions:
        print("ERROR: no matching solutions found", file=sys.stderr)
        sys.exit(1)

    print(f"Solutions ({args.problem} n={args.size}): {len(all_solutions)}")

    if args.num_samples > 0:
        sampled = random.sample(all_solutions, min(args.num_samples, len(all_solutions)))
    else:
        sampled = all_solutions
    print(f"Sampled: {len(sampled)}")

    # DeepSeek client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"API: {args.base_url}  model: {args.model}")

    # 断点续跑
    existing_ids = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
    if existing_ids:
        print(f"  Skipping {len(existing_ids)} existing")

    tasks = [(r, r["id"]) for r in sampled if r["id"] not in existing_ids]
    print(f"  To generate: {len(tasks)}")

    if not tasks:
        print("All done!")
        return

    # 生成逻辑
    write_lock = threading.Lock()
    stats_lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "retried": 0}
    t_start = time.time()
    max_qr = args.max_quality_retries

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def process_one(item):
        r, sample_id = item
        prompt_dict = build_prompt(
            r["solution"], r["prompt"]["system"], r["prompt"]["user"]
        )

        for attempt in range(max_qr):
            result = call_deepseek(
                client, prompt_dict["system"], prompt_dict["user"],
                args.model, args.max_tokens
            )
            if result is None:
                continue

            output = result["output"]
            ok, reason = quality_check(output, r["solution"])
            if not ok:
                print(f"    [{sample_id}] Quality fail: {reason}  "
                      f"(output[:200] = {output[:200]!r})")
                continue

            output = replace_answer(output, r["solution"])
            with stats_lock:
                stats["ok"] += 1
                if attempt > 0:
                    stats["retried"] += 1

            return {
                "id":            sample_id,
                "problem_type":  r["problem_type"],
                "n":             r["n"],
                "sample_idx":    r.get("sample_idx", 0),
                "prompt":        prompt_dict,
                "lkh_answer":    r["solution"],
                "output":        output,
                "output_tokens": result["output_tokens"],
                "prompt_tokens": result["prompt_tokens"],
                "total_tokens":  result["total_tokens"],
                "timestamp":     datetime.now().isoformat(),
            }

        with stats_lock:
            stats["fail"] += 1
        return None

    # 预览模式（也并发）
    if args.preview > 0:
        preview_tasks = tasks[:args.preview]
        preview_conc = min(args.preview, args.concurrency)
        print(f"\nPreview: generating {len(preview_tasks)} samples "
              f"(concurrency={preview_conc})...\n")

        results = []
        with ThreadPoolExecutor(max_workers=preview_conc) as pool:
            futures = {pool.submit(process_one, t): t for t in preview_tasks}
            for future in as_completed(futures):
                record = future.result()
                item = futures[future]
                if record is None:
                    print(f"  [{item[1]}] FAILED")
                    continue
                results.append(record)
                think_start = record["output"].index("<think>") + 7
                think_end = record["output"].index("</think>")
                think_text = record["output"][think_start:think_end].strip()
                answer_text = record["output"][think_end + len("</think>"):].strip()
                print(f"  [{record['id']}]  tokens: in={record['prompt_tokens']} out={record['output_tokens']}")
                print(f"  Think ({len(think_text)} chars):")
                for line in think_text.split("\n")[:30]:
                    print(f"    {line}")
                if think_text.count("\n") > 30:
                    print(f"    ... ({think_text.count(chr(10))-30} more lines)")
                print(f"  Answer:")
                print(f"    {answer_text[:300]}")
                print()

        if results:
            with open(args.output, "a", encoding="utf-8") as fout:
                for record in results:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Preview done: {stats['ok']}/{len(preview_tasks)} passed")
        return

    # 全量并发
    total_tasks = len(tasks)
    print(f"\nStarting generation (concurrency={args.concurrency}, "
          f"tasks={total_tasks}, max_retries={max_qr})...\n")

    with open(args.output, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_one, t): t for t in tasks}
            done_count = 0
            for future in as_completed(futures):
                record = future.result()
                done_count += 1
                if record is not None:
                    with write_lock:
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                if done_count % 50 == 0 or done_count == total_tasks:
                    elapsed_min = (time.time() - t_start) / 60
                    with stats_lock:
                        ok, fail = stats["ok"], stats["fail"]
                    rate = ok / max(ok + fail, 1) * 100
                    speed = done_count / max(elapsed_min, 0.01)
                    eta = (total_tasks - done_count) / max(speed, 0.01)
                    cost_input = (ok * 1500) / 1e6 * 0.435  # rough estimate
                    cost_output = (ok * 3000) / 1e6 * 0.87
                    print(f"  {done_count}/{total_tasks}  "
                          f"ok={ok} fail={fail} rate={rate:.0f}%  "
                          f"speed={speed:.0f}/min  ETA={eta:.0f}min  "
                          f"cost~${cost_input + cost_output:.2f}")

    elapsed_total = (time.time() - t_start) / 60
    total = stats["ok"] + stats["fail"]
    print(f"\nDone: {stats['ok']}/{total} passed ({stats['ok']/max(total,1)*100:.0f}%)  "
          f"retried={stats['retried']}  time={elapsed_total:.1f}min")


if __name__ == "__main__":
    main()
