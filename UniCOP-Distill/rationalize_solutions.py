"""读取现有 LKH 解，通过多个 vLLM 服务器并行生成 rationalization 数据。

用法：
    # 单服务器
    python rationalize_solutions.py \
        --solutions data/solutions_cvrp20.jsonl \
        --vllm_urls http://localhost:8000/v1

    # 多服务器（4 卡并行）
    python rationalize_solutions.py \
        --solutions data/solutions_cvrp20.jsonl \
        --vllm_urls http://localhost:8000/v1 http://localhost:8001/v1 \
                    http://localhost:8002/v1 http://localhost:8003/v1

输出格式与 generate_chains.py / chains_v3_clean.jsonl 兼容，
可直接用 train_sft_stage2.py 训练。
"""

import argparse
import itertools
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

_POSTHOC_SUFFIX = (
    "\n\nYour output MUST start with <think> and follow this exact structure:\n\n"
    "<think>\n[your reasoning here]\n</think>\n[solution in required format]\n\n"
    "Rules:\n"
    "1. Your FIRST token MUST be '<think>'. Do NOT output anything before <think>.\n"
    "2. In <think>, show your step-by-step decision process for "
    "constructing the route from scratch. At each step, state where you are, "
    "which nearby nodes are candidates, and why you pick the next one "
    "(e.g. nearest distance, problem-specific constraints). "
    "Write as if you are solving this problem yourself for the first time.\n"
    "3. Keep <think> concise (a few hundred words at most). "
    "Do NOT mention that a solution was provided or given to you. "
    "Do NOT describe your task as 'reconstructing', 'explaining', or 'justifying' a solution. "
    "You are solving this problem from scratch — your reasoning should read as original problem-solving, "
    "not as post-hoc analysis of a known answer.\n"
    "4. After </think>, output the solution exactly in the required format.\n"
    "5. Do NOT output the solution before <think>. The solution ONLY appears after </think>."
)

_LEAK_PATTERNS = [
    "given the solution", "given the answer", "given to me", "given this solution",
    "provided to me", "provided the solution", "provided the answer",
    "target solution", "target answer", "target route",
    "reconstruct", "reverse engineer",
    "pretend", "act as if", "fabricat",
    "was told to", "asked to explain", "asked to justify",
    "known answer", "know the answer", "already know",
    "post-hoc", "posthoc", "post hoc",
    "working backward", "work backward",
]

_SCKEY = "SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
_CALC_SAMPLE_COUNT = 20


def build_posthoc_prompt(solution: str, system: str, user: str) -> dict:
    system_posthoc = system + _POSTHOC_SUFFIX
    user_posthoc = (
        user
        + f"\n\nTarget solution (you MUST output exactly this solution after </think>,"
          f" but do NOT reveal it was given to you):\n{solution}"
        + "\n\nStart your response with <think> immediately. "
          "Solve this problem step by step, then output the target solution after </think>."
    )
    return {"system": system_posthoc, "user": user_posthoc}


def calc_max_model_len(solutions: list, max_tokens: int, tokenizer_path: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    sample = random.sample(solutions, min(_CALC_SAMPLE_COUNT, len(solutions)))
    max_prompt_tokens = 0
    for r in sample:
        p = build_posthoc_prompt(r["solution"], r["prompt"]["system"], r["prompt"]["user"])
        text = p["system"] + "\n" + p["user"]
        n_tokens = len(tok.encode(text))
        max_prompt_tokens = max(max_prompt_tokens, n_tokens)

    total = max_prompt_tokens + max_tokens
    aligned = ((total + 255) // 256) * 256
    return aligned


def _parse_route_line(line: str) -> list[int] | None:
    """解析单行 Route 行，返回节点列表（含首尾 depot）。"""
    after_colon = line.split(":", 1)[-1]
    segments = [s.strip() for s in after_colon.split("->")]
    nodes = [int(s) for s in segments if s.isdigit()]
    return nodes if len(nodes) >= 3 else None


_RE_MULTI_ROUTE = re.compile(r"Route\s+\d+\s*:", re.IGNORECASE)
_RE_SINGLE_ROUTE = re.compile(r"Route\s*:", re.IGNORECASE)


def _parse_single_route(text: str) -> tuple[list[int], str]:
    """TSP: 单条路线 'Route: 0 -> 3 -> ... -> 0'，返回客户节点序列。"""
    for line in text.strip().splitlines():
        line = line.strip()
        if not _RE_SINGLE_ROUTE.match(line):
            continue
        nodes = _parse_route_line(line)
        if nodes is None:
            return [], "BAD_FORMAT"
        if nodes[0] != 0 or nodes[-1] != 0:
            return [], "BAD_DEPOT"
        customers = nodes[1:-1]
        if not customers or 0 in customers:
            return [], "BAD_DEPOT"
        return customers, "ok"
    return [], "NO_ROUTE"


def _parse_multi_routes(text: str) -> tuple[list[list[int]], str]:
    """CVRP/VRPTW: 多条路线 'Route 1: 0 -> ... -> 0'，返回各路线的客户节点序列。"""
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


def _is_multi_route(text: str) -> bool:
    return bool(_RE_MULTI_ROUTE.search(text))


def quality_check(output: str, lkh_solution: str) -> tuple[bool, str]:
    if "<think>" not in output or "</think>" not in output:
        return False, "NO_THINK_TAGS"

    think_start = output.index("<think>") + 7
    think_end = output.index("</think>")
    think_content = output[think_start:think_end]

    if len(think_content.strip()) < 100:
        return False, "THINK_TOO_SHORT"

    think_lower = think_content.lower()
    for pattern in _LEAK_PATTERNS:
        if pattern in think_lower:
            return False, f"LEAK:{pattern}"

    answer_part = output[think_end + len("</think>"):]
    multi = _is_multi_route(lkh_solution)

    if multi:
        model_routes, model_err = _parse_multi_routes(answer_part)
        if not model_routes:
            return False, model_err
        lkh_routes, _ = _parse_multi_routes(lkh_solution)
        if sorted(model_routes) != sorted(lkh_routes):
            return False, "ROUTES_MISMATCH"
    else:
        model_seq, model_err = _parse_single_route(answer_part)
        if not model_seq:
            return False, model_err
        lkh_seq, _ = _parse_single_route(lkh_solution)
        if model_seq != lkh_seq:
            return False, "ROUTE_MISMATCH"

    return True, "ok"


def replace_answer(output: str, correct_solution: str) -> str:
    """把 </think> 后面的内容替换成 LKH 正确解。"""
    if "</think>" not in output:
        return output
    think_end = output.index("</think>") + len("</think>")
    return output[:think_end] + "\n" + correct_solution


def call_vllm(client: OpenAI, system: str, user: str,
              model: str, max_tokens: int,
              max_retries: int = 3, retry_delay: float = 5.0) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=1.0,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = response.usage
            raw = choice.message.content or ""
            # R1-Distill chat template 在 assistant 前缀加了 <think>\n，
            # vLLM 只返回模型生成部分（不含 <think>），需要补回来
            if not raw.lstrip().startswith("<think>"):
                raw = "<think>\n" + raw
            return {
                "output":        raw,
                "output_tokens": usage.completion_tokens if usage else None,
                "prompt_tokens":  usage.prompt_tokens if usage else None,
                "total_tokens":   usage.total_tokens if usage else None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  vLLM ERROR (retry {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                print(f"  vLLM ERROR (gave up after {max_retries} retries): {e}")
                return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", required=True,
                        help="LKH solutions JSONL 文件路径")
    parser.add_argument("--vllm_urls", nargs="+", default=[],
                        help="vLLM 服务器 URL 列表")
    parser.add_argument("--calc_max_model_len", action="store_true",
                        help="采样 20 条计算 max-model-len 后退出")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer 路径（--calc_max_model_len 时必填）")
    parser.add_argument("--output", default="data/chains_self.jsonl")
    parser.add_argument("--problem", default="cvrp")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="从 solutions 中取多少条（0=全部）")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=32,
                        help="总并发数（均摊到各 vLLM 服务器）")
    parser.add_argument("--max_quality_retries", type=int, default=5,
                        help="每条样本质量不合格时的最大重试次数")
    parser.add_argument("--preview", type=int, default=0,
                        help="只生成前 N 条并打印详情，用于人工验证")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 读取 solutions ─────────────────────────────────────────────────
    all_solutions = []
    with open(args.solutions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("problem_type") == args.problem and r.get("n") == args.size:
                all_solutions.append(r)

    if len(all_solutions) == 0:
        print("ERROR: 没有找到匹配的 solutions", file=sys.stderr)
        sys.exit(1)

    if args.calc_max_model_len:
        if not args.tokenizer:
            print("ERROR: --calc_max_model_len 需要 --tokenizer", file=sys.stderr)
            sys.exit(1)
        result = calc_max_model_len(all_solutions, args.max_tokens, args.tokenizer)
        print(result)
        return

    print(f"Solutions ({args.problem} n={args.size}): {len(all_solutions)} 条")

    if not args.vllm_urls:
        print("ERROR: 需要 --vllm_urls", file=sys.stderr)
        sys.exit(1)

    if args.num_samples <= 0:
        sampled = all_solutions
    else:
        sampled = random.sample(all_solutions, min(args.num_samples, len(all_solutions)))
    print(f"采样 {len(sampled)} 条做 rationalization")

    # ── 创建 vLLM clients ─────────────────────────────────────────────
    clients = []
    for url in args.vllm_urls:
        c = OpenAI(base_url=url, api_key="dummy")
        models = c.models.list()
        model_name = models.data[0].id
        clients.append((c, model_name))
        print(f"  vLLM: {url} → {model_name}")

    client_cycle = itertools.cycle(range(len(clients)))
    cycle_lock = threading.Lock()

    def get_next_client():
        with cycle_lock:
            idx = next(client_cycle)
        return clients[idx]

    # ── 断点续跑 ─────────────────────────────────────────────────────
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
        print(f"  已有 {len(existing_ids)} 条，跳过重复")

    tasks = [(r, r["id"]) for r in sampled if r["id"] not in existing_ids]
    print(f"  待生成: {len(tasks)} 条")

    if not tasks:
        print("全部已完成!")
        return

    # ── 并发生成 ─────────────────────────────────────────────────────
    write_lock = threading.Lock()
    stats_lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "retried": 0}
    t_start = time.time()
    max_qr = args.max_quality_retries

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def process_one(item):
        r, sample_id = item
        prompt_dict = build_posthoc_prompt(
            r["solution"], r["prompt"]["system"], r["prompt"]["user"],
        )

        for attempt in range(max_qr):
            client, model_name = get_next_client()
            result = call_vllm(client, prompt_dict["system"], prompt_dict["user"],
                               model_name, args.max_tokens)
            if result is None:
                continue

            output = result["output"]
            output_tokens = result.get("output_tokens") or 0
            if output_tokens > args.max_tokens:
                continue

            ok, reason = quality_check(output, r["solution"])
            if not ok:
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

    # ── 预览模式 ─────────────────────────────────────────────────────
    if args.preview > 0:
        preview_tasks = tasks[:args.preview]
        print(f"\n预览模式: 生成前 {len(preview_tasks)} 条...\n")
        with open(args.output, "a", encoding="utf-8") as fout:
            for item in preview_tasks:
                record = process_one(item)
                if record is None:
                    print(f"  [{item[1]}] FAILED (重试 {max_qr} 次均不合格)\n")
                    continue
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                think_start = record["output"].index("<think>") + 7
                think_end = record["output"].index("</think>")
                think_text = record["output"][think_start:think_end].strip()
                answer_text = record["output"][think_end + len("</think>"):].strip()
                print(f"  [{record['id']}]  tokens={record['output_tokens']}")
                print(f"  Think ({len(think_text)} chars):")
                print(f"    {think_text[:500]}")
                if len(think_text) > 500:
                    print(f"    ... (省略 {len(think_text)-500} chars)")
                print(f"  Answer (前200 chars):")
                print(f"    {answer_text[:200]}")
                print()
        print(f"预览完成: {stats['ok']}/{len(preview_tasks)} 合格，"
              f"已保存到 {args.output}")
        print("检查无误后去掉 --preview 重新运行（已生成的会自动跳过）")
        return

    # ── 全量并发生成 ─────────────────────────────────────────────────
    total_tasks = len(tasks)
    print(f"\n开始生成 (并发={args.concurrency}, 服务器={len(clients)}, "
          f"总任务={total_tasks}, 每条最多重试={max_qr})...\n")

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
                if done_count % 100 == 0 or done_count == total_tasks:
                    elapsed_min = (time.time() - t_start) / 60
                    with stats_lock:
                        ok, fail = stats["ok"], stats["fail"]
                        retried = stats["retried"]
                    rate = ok / max(ok + fail, 1) * 100
                    speed = done_count / max(elapsed_min, 0.01)
                    eta = (total_tasks - done_count) / max(speed, 0.01)
                    print(f"  进度: {done_count}/{total_tasks}  "
                          f"合格: {ok}  失败: {fail}  重试成功: {retried}  "
                          f"合格率: {rate:.0f}%  "
                          f"速度: {speed:.0f}/min  "
                          f"ETA: {eta:.0f}min")

    total = stats["ok"] + stats["fail"]
    elapsed_total = (time.time() - t_start) / 60
    print(f"\n完成: {stats['ok']}/{total} 合格 ({stats['ok']/max(total,1)*100:.0f}%)  "
          f"重试成功: {stats['retried']}  耗时: {elapsed_total:.1f}min")
    if stats["fail"] > 0:
        print(f"  ⚠ {stats['fail']} 条在 {max_qr} 次重试后仍不合格")
    _notify(f"Rationalize 完成: {stats['ok']}/{total} 合格")


def _notify(title: str, desp: str = ""):
    import urllib.request
    import urllib.parse
    try:
        data = urllib.parse.urlencode({"title": title[:100], "desp": desp[:500]}).encode()
        req = urllib.request.Request(f"https://sctapi.ftqq.com/{_SCKEY}.send", data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
