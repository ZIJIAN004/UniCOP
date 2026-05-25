"""
optimal/build_optimal.py
阶段二：读取 generate_testset.py 冻结的实例，逐个求 (近) 最优解，缓存为 JSON，供
optimality gap 使用。本脚本【不再生成实例】，只读冻结文件——请先跑阶段一：
  python -m optimal.generate_testset

求解器：TSP→LKH（未配 LKH_BIN 回退 PyVRP），CVRP/TSPTW/VRPTW→PyVRP/HGS。
cost 与 evaluate.py 的 get_tour_distance 同口径（原始坐标欧氏边长）。

缓存 costs[i] 与冻结实例第 i 个一一对应；前缀一致：用 --num_test K 只求前 K 个。

用法（在 UniCOP 仓库根目录下）：
  python -m optimal.build_optimal --sizes 20 50 100 --timeout 5 --workers 8
  python -m optimal.build_optimal --problem_types tsp cvrp --sizes 50 --num_test 200
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

from optimal.solvers import solve_instance, SUPPORTED, LKH_BIN  # noqa: E402
from optimal.generate_testset import INSTANCES_DIR  # noqa: E402
from optimal.loader import load_instances  # noqa: E402

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


def _solve_one(args):
    """ProcessPoolExecutor worker：求解单个实例，返回 (idx, cost, feasible, solver)。"""
    idx, problem_type, instance, timeout, lkh_bin, seed = args
    res = solve_instance(problem_type, instance, timeout=timeout,
                         lkh_bin=lkh_bin, seed=seed)
    return idx, res["cost"], res["feasible"], res["solver"]


def build_one(problem_type: str, n: int, num_test, seed: int,
              timeout: int, workers: int, lkh_bin: str, out_dir: str,
              instances_dir: str) -> dict:
    # 读取阶段一冻结的实例（不再现场生成）；num_test=None 求解全部，否则取前 num_test 个
    instances = load_instances(problem_type, n, seed=seed,
                               num_test=num_test, inst_dir=instances_dir)
    num = len(instances)
    print(f"\n[{problem_type.upper()} n={n}] 读取冻结实例 {num} 个 (seed={seed}) ...")

    costs = [None] * num
    feasibles = [False] * num
    solver_used = None
    t0 = time.time()

    tasks = [(i, problem_type, instances[i], timeout, lkh_bin, seed)
             for i in range(num)]

    if workers <= 1:
        for k, task in enumerate(tasks):
            idx, cost, feas, solver = _solve_one(task)
            costs[idx], feasibles[idx], solver_used = cost, feas, solver
            if (k + 1) % max(1, num // 20) == 0 or k + 1 == num:
                print(f"    {k + 1}/{num}  ({time.time() - t0:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_solve_one, t) for t in tasks]
            done = 0
            for fut in as_completed(futures):
                idx, cost, feas, solver = fut.result()
                costs[idx], feasibles[idx], solver_used = cost, feas, solver
                done += 1
                if done % max(1, num // 20) == 0 or done == num:
                    print(f"    {done}/{num}  ({time.time() - t0:.0f}s)")

    infeasible = [i for i, f in enumerate(feasibles) if not f]
    valid = [c for c in costs if c is not None]
    elapsed = time.time() - t0

    payload = {
        "problem_type": problem_type,
        "n": n,
        "seed": seed,
        "num_test": num,
        "solver": solver_used,
        "timeout": timeout,
        "lkh_bin": lkh_bin or "(none, pyvrp fallback)",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 1),
        "infeasible_count": len(infeasible),
        "infeasible_idx": infeasible[:50],  # 仅留前 50 个索引便于排查
        "mean_cost": float(np.mean(valid)) if valid else None,
        "costs": costs,           # 长度 num，失败处为 null
    }

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{problem_type}_n{n}_seed{seed}_N{num}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"    ✓ solver={solver_used}  mean_cost="
          f"{payload['mean_cost']:.4f}" if payload["mean_cost"] is not None else "    ✗ 全部失败")
    print(f"    infeasible={len(infeasible)}/{num}  耗时 {elapsed:.0f}s  →  {fpath}")
    return payload


def main():
    ap = argparse.ArgumentParser(
        description="阶段二：读取冻结实例求 LKH/HGS 近最优基线（先跑 generate_testset）")
    ap.add_argument("--problem_types", nargs="+", default=list(SUPPORTED),
                    help=f"默认全部受支持类型: {list(SUPPORTED)}（TSPDL 暂不支持）")
    ap.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100],
                    help="问题规模列表（客户节点数），默认 20 50 100")
    ap.add_argument("--num_test", type=int, default=None,
                    help="只求冻结集前 N 个；默认 None=求全部")
    ap.add_argument("--seed", type=int, default=_DEFAULT_SEED,
                    help=f"须与冻结实例(generate_testset)一致，默认 {_DEFAULT_SEED}")
    ap.add_argument("--timeout", type=int, default=5,
                    help="单实例求解时间上限（秒），n 越大建议越长")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行进程数（>1 启用多进程；Windows 需在 __main__ 下运行）")
    ap.add_argument("--lkh_bin", default=LKH_BIN,
                    help="LKH 二进制路径（仅 TSP 使用；留空则 TSP 回退 PyVRP）")
    ap.add_argument("--instances_dir", default=INSTANCES_DIR,
                    help="冻结实例目录（generate_testset 的输出）")
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "cache"))
    args = ap.parse_args()

    for pt in args.problem_types:
        if pt.lower() not in SUPPORTED:
            ap.error(f"不支持的问题类型: {pt}（支持 {list(SUPPORTED)}）")

    print("=" * 60)
    print(f"problem_types={args.problem_types}  sizes={args.sizes}")
    print(f"num_test={args.num_test or 'ALL'}  seed={args.seed}  "
          f"timeout={args.timeout}s  workers={args.workers}")
    print(f"LKH_BIN={'(set)' if args.lkh_bin else '(none → TSP 用 PyVRP)'}")
    print(f"instances_dir={args.instances_dir}")
    print(f"out_dir={args.out_dir}")
    print("=" * 60)

    summary = []
    t_start = time.time()
    try:
        for n in args.sizes:
            for pt in args.problem_types:
                p = build_one(pt, n, args.num_test, args.seed,
                              args.timeout, args.workers, args.lkh_bin,
                              args.out_dir, args.instances_dir)
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
