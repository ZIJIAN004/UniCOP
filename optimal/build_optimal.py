"""
optimal/build_optimal.py
为 evaluate.py 使用的测试集逐实例求 (近) 最优解，缓存为 JSON，供 optimality gap 使用。

⚠️ 复现对齐（必须与 evaluate.py 完全一致）：
  evaluate.py:evaluate_single 中 `rng = np.random.default_rng(seed=9999)`，
  随后顺序调用 `prob.generate_instance(problem_size, rng)` 共 num_test 次。
  本脚本以同样的 seed / 顺序重建实例，因此第 i 个实例与 evaluate.py 第 i 个一一对应。
  前缀一致性：evaluate.py 用 num_test=K (K<缓存量) 评测时，取的就是同一序列的前 K 个，
  故可“缓存大集合、评测抽前缀子集”。改动 evaluate.py 的 seed 时务必同步本脚本 --seed。

用法（在 UniCOP-Reason 目录下）：
  python -m optimal.build_optimal --sizes 100 --num_test 1000 --timeout 5
  python -m optimal.build_optimal --problem_types tsp cvrp --sizes 50 100 --workers 8
  LKH_BIN=/path/to/LKH python -m optimal.build_optimal --problem_types tsp   # TSP 用 LKH
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

# optimal/ 位于 UniCOP 顶层；问题生成逻辑(problems/、utils/)在 UniCOP-Reason 下。
# 同时把 UniCOP 根(供 `import optimal.*`)和 UniCOP-Reason(供 `import problems`)加到 path，
# 兼容 `python -m optimal.build_optimal` 与 `python optimal/build_optimal.py` 两种调用。
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../UniCOP/optimal
_UNICOP = os.path.dirname(_HERE)                            # .../UniCOP
_REASON = os.path.join(_UNICOP, "UniCOP-Reason")           # problems/ utils/ 所在
for _p in (_UNICOP, _REASON):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from problems import get_problem  # noqa: E402
from optimal.solvers import solve_instance, SUPPORTED, LKH_BIN  # noqa: E402

# Server 酱（长时间运行脚本通知规则）
_SCKEY = "SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
_DEFAULT_SEED = 9999  # 必须与 evaluate.py:evaluate_single 的 seed 一致


def _notify(title: str, desp: str = "") -> None:
    try:
        url = f"https://sctapi.ftqq.com/{_SCKEY}.send"
        data = urllib.parse.urlencode({"title": title[:100], "desp": desp[:2000]}).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:  # noqa: BLE001 — 通知失败不影响主流程
        print(f"[notify failed] {e}")


def _regenerate_instances(problem_type: str, n: int, num_test: int, seed: int) -> list[dict]:
    """复现 evaluate.py 的测试集：同 seed、同顺序调用 generate_instance。"""
    prob = get_problem(problem_type)
    rng = np.random.default_rng(seed=seed)
    return [prob.generate_instance(n, rng) for _ in range(num_test)]


def _solve_one(args):
    """ProcessPoolExecutor worker：求解单个实例，返回 (idx, cost, feasible, solver)。"""
    idx, problem_type, instance, timeout, lkh_bin, seed = args
    res = solve_instance(problem_type, instance, timeout=timeout,
                         lkh_bin=lkh_bin, seed=seed)
    return idx, res["cost"], res["feasible"], res["solver"]


def build_one(problem_type: str, n: int, num_test: int, seed: int,
              timeout: int, workers: int, lkh_bin: str, out_dir: str) -> dict:
    print(f"\n[{problem_type.upper()} n={n}] 复现 {num_test} 个实例 (seed={seed}) ...")
    instances = _regenerate_instances(problem_type, n, num_test, seed)

    costs = [None] * num_test
    feasibles = [False] * num_test
    solver_used = None
    t0 = time.time()

    tasks = [(i, problem_type, instances[i], timeout, lkh_bin, seed)
             for i in range(num_test)]

    if workers <= 1:
        for k, task in enumerate(tasks):
            idx, cost, feas, solver = _solve_one(task)
            costs[idx], feasibles[idx], solver_used = cost, feas, solver
            if (k + 1) % max(1, num_test // 20) == 0 or k + 1 == num_test:
                print(f"    {k + 1}/{num_test}  ({time.time() - t0:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_solve_one, t) for t in tasks]
            done = 0
            for fut in as_completed(futures):
                idx, cost, feas, solver = fut.result()
                costs[idx], feasibles[idx], solver_used = cost, feas, solver
                done += 1
                if done % max(1, num_test // 20) == 0 or done == num_test:
                    print(f"    {done}/{num_test}  ({time.time() - t0:.0f}s)")

    infeasible = [i for i, f in enumerate(feasibles) if not f]
    valid = [c for c in costs if c is not None]
    elapsed = time.time() - t0

    payload = {
        "problem_type": problem_type,
        "n": n,
        "seed": seed,
        "num_test": num_test,
        "solver": solver_used,
        "timeout": timeout,
        "lkh_bin": lkh_bin or "(none, pyvrp fallback)",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 1),
        "infeasible_count": len(infeasible),
        "infeasible_idx": infeasible[:50],  # 仅留前 50 个索引便于排查
        "mean_cost": float(np.mean(valid)) if valid else None,
        "costs": costs,           # 长度 num_test，失败处为 null
    }

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{problem_type}_n{n}_seed{seed}_N{num_test}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"    ✓ solver={solver_used}  mean_cost="
          f"{payload['mean_cost']:.4f}" if payload["mean_cost"] is not None else "    ✗ 全部失败")
    print(f"    infeasible={len(infeasible)}/{num_test}  耗时 {elapsed:.0f}s  →  {fpath}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="生成 optimality gap 的 LKH/HGS 基线缓存")
    ap.add_argument("--problem_types", nargs="+", default=list(SUPPORTED),
                    help=f"默认全部受支持类型: {list(SUPPORTED)}（TSPDL 暂不支持）")
    ap.add_argument("--sizes", nargs="+", type=int, default=[100],
                    help="问题规模列表（客户节点数），如 50 100")
    ap.add_argument("--num_test", type=int, default=1000,
                    help="测试实例数（须 >= evaluate.py 评测用的 num_test，前缀一致）")
    ap.add_argument("--seed", type=int, default=_DEFAULT_SEED,
                    help=f"必须与 evaluate.py 一致，默认 {_DEFAULT_SEED}")
    ap.add_argument("--timeout", type=int, default=5,
                    help="单实例求解时间上限（秒），n 越大建议越长")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行进程数（>1 启用多进程；Windows 需在 __main__ 下运行）")
    ap.add_argument("--lkh_bin", default=LKH_BIN,
                    help="LKH 二进制路径（仅 TSP 使用；留空则 TSP 回退 PyVRP）")
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "cache"))
    args = ap.parse_args()

    for pt in args.problem_types:
        if pt.lower() not in SUPPORTED:
            ap.error(f"不支持的问题类型: {pt}（支持 {list(SUPPORTED)}）")

    print("=" * 60)
    print(f"problem_types={args.problem_types}  sizes={args.sizes}")
    print(f"num_test={args.num_test}  seed={args.seed}  timeout={args.timeout}s  "
          f"workers={args.workers}")
    print(f"LKH_BIN={'(set)' if args.lkh_bin else '(none → TSP 用 PyVRP)'}")
    print(f"out_dir={args.out_dir}")
    print("=" * 60)

    summary = []
    t_start = time.time()
    try:
        for n in args.sizes:
            for pt in args.problem_types:
                p = build_one(pt, n, args.num_test, args.seed,
                              args.timeout, args.workers, args.lkh_bin, args.out_dir)
                summary.append(
                    f"{pt} n={n}: mean={p['mean_cost']}, "
                    f"infeas={p['infeasible_count']}/{p['num_test']}"
                )
    except KeyboardInterrupt:
        _notify("⚠️ optimal 基线生成被手动中断",
                "已完成:\n" + "\n".join(summary))
        print("\n[中断] 已发送通知")
        raise
    except Exception as e:  # noqa: BLE001
        _notify("❌ optimal 基线生成异常退出", f"{type(e).__name__}: {e}\n\n"
                "已完成:\n" + "\n".join(summary))
        print(f"\n[异常] {e}，已发送通知")
        raise

    total = time.time() - t_start
    _notify("✅ optimal 基线生成完成",
            f"总耗时 {total / 60:.1f} 分钟\n\n" + "\n".join(summary))
    print(f"\n{'=' * 60}\n全部完成，总耗时 {total / 60:.1f} 分钟\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
