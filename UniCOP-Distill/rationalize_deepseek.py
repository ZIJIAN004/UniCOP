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

BASE_SYSTEM = {
    "cvrp": (
        "You are a logistics route planning expert solving the Capacitated Vehicle Routing Problem (CVRP).\n"
        "Rules: Multiple vehicles depart from node 0; each vehicle visits a subset of customers and returns to node 0; "
        "total demand per route must not exceed vehicle capacity; each customer is visited exactly once; minimize total distance.\n"
        "Before answering, think through the problem in <reasoning>...</reasoning>. "
        "Consider how to balance grouping customers into feasible routes against minimizing total travel distance. "
        "Clustering or savings-based ideas may be a useful lens.\n"
        "After completing your analysis, output in the following format (one route per line, nodes in visit order):\n"
        "Route 1: 0 -> node -> ... -> 0\n"
        "Route 2: 0 -> node -> ... -> 0"
    ),
    "tsp": (
        "You are a route planning expert solving the Travelling Salesman Problem (TSP).\n"
        "Rules: Starting from node 0, visit all customer nodes exactly once and return to node 0, minimizing total distance.\n"
        "Before answering, think through the problem in <reasoning>...</reasoning>. "
        "Consider how the spatial layout of nodes might suggest a natural visit order, "
        "and whether the initial route can be improved. Greedy construction or local swap strategies may be useful starting points.\n"
        "After completing your analysis, output in the following format:\n"
        "Route: 0 -> A -> B -> C -> ... -> 0"
    ),
    "tsptw": (
        "You are a route planning expert solving the Travelling Salesman Problem with Time Windows (TSPTW).\n"
        "Rules:\n"
        "- Start from node 0 (depot), visit all customer nodes exactly once, and return to node 0\n"
        "- Travel time between nodes = Euclidean distance\n"
        "- Each customer node has a time window [earliest, latest]: arrival time must be <= latest\n"
        "- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue\n"
        "- Objective: minimize total travel distance\n"
        "Before answering, think through the problem in <reasoning>...</reasoning>.\n"
        "After completing your analysis, output in the following format:\n"
        "Route: 0 -> A -> B -> C -> ... -> 0"
    ),
    "vrptw": (
        "You are a logistics scheduling expert solving the Vehicle Routing Problem with Time Windows (VRPTW).\n"
        "Rules:\n"
        "- Multiple vehicles depart from node 0 (depot); each vehicle visits a subset of customers and returns to node 0\n"
        "- All customer nodes are visited exactly once\n"
        "- Travel time between nodes = Euclidean distance\n"
        "- Each customer node has a time window [earliest, latest]: arrival time must be <= latest\n"
        "- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue\n"
        "- Objective: minimize total travel distance across all routes\n"
        "Before answering, think through the problem in <reasoning>...</reasoning>.\n"
        "After completing your analysis, output in the following format (one route per line, nodes in visit order):\n"
        "Route 1: 0 -> node -> ... -> 0\n"
        "Route 2: 0 -> node -> ... -> 0"
    ),
}

SYSTEM_SUFFIX = """

Your output MUST start with <reasoning> and follow this exact structure:

<reasoning>
[your reasoning here]
</reasoning>
[solution in required format]

Rules:
1. Your FIRST token MUST be '<reasoning>'. Do NOT output anything before <reasoning>.
2. In <reasoning>, show your step-by-step decision process for constructing the route from scratch. At each step, state where you are, which nearby nodes are candidates, and why you pick the next one (e.g. nearest distance, capacity constraint, cluster boundary). Write as if you are solving this problem yourself for the first time.
3. When you skip a closer node for a farther one, briefly note why (e.g. "Node 5 is closer but would leave an isolated node; choosing Node 8 to clear this cluster first").
4. For each route, verify total demand does not exceed vehicle capacity before closing it.
5. Keep <reasoning> concise (a few hundred words at most). Do NOT mention that a solution was provided or given to you. Do NOT describe your task as 'reconstructing', 'explaining', or 'justifying' a solution. You are solving this problem from scratch — your reasoning should read as original problem-solving, not as post-hoc analysis of a known answer.
6. After </reasoning>, output the solution exactly in the required format.
7. Do NOT output the solution before <reasoning>. The solution ONLY appears after </reasoning>."""

FEWSHOT = ""

STRUCTURED_FORMAT = """

Additionally, structure your step-by-step construction using this format. Build each route one node at a time:

At the start of each new route, list: "Unvisited: {node_id, node_id, ...}"

Each step format:
  [R1,step] at Node N, cap=X.XX. Candidates: A(d=X.XX, dem=X.XX), B(d=X.XX, dem=X.XX). [≤10 words: why] → M (cap→X.XX)

When no unvisited node fits remaining capacity:
  [R1,step] cap=X.XX. Remaining nodes all exceed capacity (A: dem=X.XX, B: dem=X.XX). → return depot

After all routes, verify: "R1:{nodes}=count | R2:{nodes}=count | ... Total: X/N ✓"
"""


def build_prompt(solution: str, problem_type: str, user: str,
                 use_fewshot: bool = True,
                 structured: bool = False) -> dict:
    base = BASE_SYSTEM.get(problem_type, BASE_SYSTEM["cvrp"])
    system_new = base + SYSTEM_SUFFIX + (FEWSHOT if use_fewshot else "")
    if structured:
        system_new += STRUCTURED_FORMAT
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


def quality_check(output: str, solution: str,
                  max_output_tokens: int = 0) -> tuple[bool, str]:
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
    capacity_words = ["capacity", "remaining", "demand", "load", "≤", "<=", "exceed"]
    if not any(w in think_lower for w in capacity_words):
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
                  use_thinking: bool = True,
                  max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            extra = {}
            if not use_thinking:
                extra["thinking"] = {"type": "disabled"}
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                extra_body=extra if extra else None,
            )
            choice = response.choices[0]
            usage = response.usage

            if use_thinking:
                reasoning = getattr(choice.message, "reasoning_content", "") or ""
                answer = (choice.message.content or "").strip()
                raw = f"<think>\n{reasoning}\n</think>\n{answer}"
            else:
                raw = choice.message.content or ""
                raw = raw.lstrip().removeprefix("</think>").lstrip()
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
    parser.add_argument("--no_fewshot", action="store_true",
                        help="Disable fewshot example to save ~350 input tokens/sample")
    parser.add_argument("--structured", action="store_true",
                        help="Enable structured step format (think-then-select at each step)")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Disable model's built-in thinking (produces rigid format)")
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
    use_thinking = not args.disable_thinking
    print(f"API: {args.base_url}  model: {args.model}  "
          f"thinking: {use_thinking}  structured: {args.structured}  "
          f"fewshot: {not args.no_fewshot}")

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
            r["solution"], r["problem_type"], r["prompt"]["user"],
            use_fewshot=not args.no_fewshot, structured=args.structured,
        )

        for attempt in range(max_qr):
            result = call_deepseek(
                client, prompt_dict["system"], prompt_dict["user"],
                args.model, args.max_tokens, use_thinking=use_thinking
            )
            if result is None:
                continue

            output = result["output"]
            out_tokens = result["output_tokens"] or 0
            if args.max_tokens and out_tokens >= args.max_tokens:
                print(f"    [{sample_id}] Skipped: output_tokens={out_tokens} >= {args.max_tokens} (truncated)")
                continue
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

        # Debug: print the exact prompt for the first sample
        first_r = preview_tasks[0][0]
        debug_prompt = build_prompt(
            first_r["solution"], first_r["problem_type"],
            first_r["prompt"]["user"], use_fewshot=not args.no_fewshot,
            structured=args.structured,
        )
        print("=== DEBUG: SYSTEM PROMPT ===")
        print(debug_prompt["system"])
        print("=== DEBUG: USER PROMPT (first 500 chars) ===")
        print(debug_prompt["user"][:500])
        print("=== END DEBUG ===\n")

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
