"""
Stage 1 数据生成：用传统求解器批量生成 (问题, 解) 对。
不调用 Gemini，不生成推理链，纯粹教模型"什么是合法的解"。

求解器分配同 lkh_solver.py：
  TSP   → LKH
  CVRP  → PyVRP (HGS)
  TSPTW → PyVRP (单车 VRPTW)
  VRPTW → PyVRP

输出格式 (JSONL):
  {
    "id":             "tsp_n20_s42_i0",
    "problem_type":   "tsp",
    "n":              20,
    "sample_idx":     0,
    "prompt":         {"system": "...", "user": "..."},
    "solution":       "Route: 0 -> 3 -> ...",
    "solver_distance": 3.456,
    "timestamp":      "2026-04-29T10:00:00"
  }

运行示例:
  python stage1_solution/generate_solutions.py --lkh_bin /path/to/LKH
  python stage1_solution/generate_solutions.py --problems tsp cvrp --sizes 20 50 --num_samples 2000
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_UNICOP_REASON_DIR = os.path.join(_PROJECT_DIR, "..", "UniCOP-Reason")

sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.abspath(_UNICOP_REASON_DIR))

from lkh_solver import solve as lkh_solve, LKH_BIN  # noqa: E402

# ── 默认参数 ──────────────────────────────────────────────────────────────────
PROBLEM_TYPES = ["tsp", "cvrp", "tsptw", "vrptw"]
NODE_SIZES = [20, 50, 100]
_SCKEY = "SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

# 按 (问题类型, 规模) 自动缩放求解器参数
# runs 仅影响 LKH/TSP；timeout 影响所有求解器（PyVRP 用 MaxRuntime）
_SOLVER_PARAMS = {
    ("tsp",   20):  {"runs": 5,  "timeout": 10},
    ("tsp",   50):  {"runs": 5,  "timeout": 30},
    ("tsp",  100):  {"runs": 10, "timeout": 120},
    ("cvrp",  20):  {"runs": 1,  "timeout": 15},
    ("cvrp",  50):  {"runs": 1,  "timeout": 60},
    ("cvrp", 100):  {"runs": 1,  "timeout": 180},
    ("tsptw", 20):  {"runs": 1,  "timeout": 15},
    ("tsptw", 50):  {"runs": 1,  "timeout": 60},
    ("tsptw",100):  {"runs": 1,  "timeout": 180},
    ("vrptw", 20):  {"runs": 1,  "timeout": 20},
    ("vrptw", 50):  {"runs": 1,  "timeout": 90},
    ("vrptw",100):  {"runs": 1,  "timeout": 240},
}
_SOLVER_PARAMS_DEFAULT = {"runs": 10, "timeout": 120}


def _strip_think_instructions(system: str) -> str:
    """从 system prompt 中剥离 <think> 推理指令（Stage 1 不需要思维链引导）。"""
    system = re.sub(
        r'Before answering, think through the problem in <think>\.\.\.</think>\.[^\n]*\n?',
        '', system,
    )
    system = system.replace("After completing your analysis, output", "Output")
    system = re.sub(r'\n{3,}', '\n\n', system).strip()
    return system


def load_existing_ids(output_path: str) -> set:
    ids = set()
    if not os.path.exists(output_path):
        return ids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def count_samples(output_path: str) -> dict:
    counts = defaultdict(int)
    if not os.path.exists(output_path):
        return counts
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                counts[(r["problem_type"], r["n"])] += 1
            except Exception:
                pass
    return counts


def _solve_one(args_tuple):
    """单条数据：生成实例 → 求解 → 构建 prompt → 返回 record (进程安全)。"""
    pt, n, sample_idx, sample_id, seed, lkh_bin = args_tuple

    # 延迟 import，每个子进程独立加载
    sys.path.insert(0, os.path.abspath(_UNICOP_REASON_DIR))
    from problems import get_problem

    problem = get_problem(pt)
    pt_offset = sum(ord(c) for c in pt)
    rng = np.random.default_rng(seed=seed + n + pt_offset + sample_idx)

    instance = problem.generate_instance(n, rng)

    params = _SOLVER_PARAMS.get((pt, n), _SOLVER_PARAMS_DEFAULT)
    solution = lkh_solve(pt, instance, lkh_bin=lkh_bin,
                         runs=params["runs"], seed=seed, timeout=params["timeout"])
    if solution is None:
        return None

    distance = problem.get_tour_distance(solution, instance)

    orig_prompt = problem.build_prompt(instance)
    prompt_dict = {}
    for msg in orig_prompt:
        if msg["role"] == "system":
            prompt_dict["system"] = _strip_think_instructions(msg["content"])
        elif msg["role"] == "user":
            prompt_dict["user"] = msg["content"]

    return {
        "id": sample_id,
        "problem_type": pt,
        "n": n,
        "sample_idx": sample_idx,
        "prompt": prompt_dict,
        "solution": solution,
        "solver_distance": round(distance, 6) if distance is not None else None,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1: 批量生成 solver 解（无 Gemini）")
    parser.add_argument("--problems", type=str, nargs="+", default=PROBLEM_TYPES,
                        choices=PROBLEM_TYPES)
    parser.add_argument("--sizes", type=int, nargs="+", default=NODE_SIZES)
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="每个 (problem, n) 组合的样本数")
    parser.add_argument("--lkh_bin", type=str, default=os.environ.get("LKH_BIN", LKH_BIN))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/solutions.jsonl")
    parser.add_argument("--workers", type=int, default=32,
                        help="并行求解进程数")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    existing_ids = load_existing_ids(args.output)
    valid_counts = count_samples(args.output)

    combos_need = []
    for pt in args.problems:
        for n in args.sizes:
            current = valid_counts.get((pt, n), 0)
            if current < args.num_samples:
                combos_need.append((pt, n, args.num_samples - current))

    total_need = sum(c[2] for c in combos_need)

    print(f"\n{'='*60}")
    print(f"  Stage 1: Solver 解数据生成")
    print(f"  目标: 每组合 {args.num_samples} 条")
    print(f"  现有: {sum(valid_counts.values())} 条  |  需补充: {total_need} 条")
    print(f"  并行: {args.workers} 进程")
    print(f"  输出: {args.output}")
    print(f"{'='*60}\n")

    if total_need == 0:
        print("所有组合已达标！")
        return

    t0 = time.time()

    with open(args.output, "a", encoding="utf-8") as fout:
        for pt, n, gap in combos_need:
            current = valid_counts.get((pt, n), 0)
            params = _SOLVER_PARAMS.get((pt, n), _SOLVER_PARAMS_DEFAULT)
            print(f"[{pt}_n{n}] 已有 {current}/{args.num_samples}，需补充 {gap} 条"
                  f"  (runs={params['runs']}, timeout={params['timeout']}s)")

            start_idx = 0
            for eid in existing_ids:
                if eid.startswith(f"{pt}_n{n}_s{args.seed}_i"):
                    try:
                        idx = int(eid.split("_i")[-1])
                        start_idx = max(start_idx, idx + 1)
                    except ValueError:
                        pass

            tasks = []
            for i in range(gap):
                sid = f"{pt}_n{n}_s{args.seed}_i{start_idx + i}"
                if sid in existing_ids:
                    continue
                tasks.append((pt, n, start_idx + i, sid, args.seed, args.lkh_bin))

            saved = 0
            failed = 0
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_solve_one, t): t for t in tasks}
                for future in as_completed(futures):
                    record = future.result()
                    if record is not None:
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        existing_ids.add(record["id"])
                        saved += 1
                    else:
                        failed += 1

                    total_done = saved + failed
                    if total_done % 100 == 0:
                        print(f"    进度: {total_done}/{len(tasks)}  (成功 {saved}, 失败 {failed})")

            print(f"    完成: 成功 {saved}/{len(tasks)}，失败 {failed}")

    elapsed = time.time() - t0
    final_counts = count_samples(args.output)
    total_saved = sum(final_counts.values())
    msg = f"Stage 1 数据生成完成: {total_saved} 条, 耗时 {elapsed:.0f}s"
    print(f"\n{msg}")
    _notify_serverchan(msg)


def _notify_serverchan(title: str, desp: str = ""):
    try:
        data = urllib.parse.urlencode({"title": title[:100], "desp": desp[:500]}).encode()
        req = urllib.request.Request(f"https://sctapi.ftqq.com/{_SCKEY}.send", data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
