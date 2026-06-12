#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_foarl_cvrp_data_mask1000.py — 用 UniCOP-Reason-Mask 完全相同的 1000 条 CVRP 实例
构建 FOARL RL 数据 (FOARL 自己的 prompt/输出格式 + PyVRP 参考距离), 多进程并行求解。

目的: FOARL 作为 baseline 与 Mask 做增益对比时, 两边必须在**同一批优化实例**上训练。
Mask 的训练实例是程序化生成的 (data/generate.py:build_dataset, 非固定文件):
    rng = np.random.default_rng(seed=42)
    for _ in range(num):  instance = CVRP.generate_instance(n=20, rng)
本脚本**逐字复刻**这一生成过程 (同一函数、同一 seed、同一调用顺序) → 得到与 Mask
逐条一致的 coords/demands/capacity。

并行不破坏对齐: 实例在主进程**顺序**生成 (保住 rng 顺序), 仅把"每条实例 PyVRP 求参考解"
这条独立的慢路 (~timeout 秒/条) 丢进进程池并行; PyVRP 求解对每条实例确定 (内部固定 seed),
并行与否结果一致。

与 Mask 的差异 (方法固有, 不是数据差异):
  - 输出格式: Mask 带 <think> 多行思维式; FOARL 单行非思维 "Routes: [[...]], Objective: X"。
  - 参考解: FOARL 的 R_o 需每条实例参考距离, 用团队 CVRP solver = PyVRP/HGS
    (lkh_solver.solve('cvrp',·) → _solve_cvrp; LKH 只用于 TSP)。

输出每条: {id, problem_type, n, instruction, input, output, instance:[coords,demands,capacity], solver_distance}

用法 (须先激活装了 pyvrp 的环境, 如 zhuoyi unicop; 纯 CPU, 别占 GPU 作业):
  python build_foarl_cvrp_data_mask1000.py \
    --mask_dir ../UniCOP-Reason-Mask --distill_dir ../UniCOP-Distill \
    --out data/foarl_cvrp20_mask1000.jsonl \
    --num 1000 --seed 42 --n 20 --k_nn 2 --timeout 5 --workers 16
"""
import argparse
import json
import os
import sys

import numpy as np

# ── 进程池 worker: 每个子进程独立 import 团队 solver, 求一条 CVRP 实例的参考解 ──
_W = {}   # worker 进程级缓存 (solve / n / timeout)


def _init_worker(distill_dir, n, timeout):
    if distill_dir not in sys.path:
        sys.path.insert(0, distill_dir)
    from lkh_solver import solve   # CVRP → PyVRP/HGS (import 在子进程内, 避免 fork/spawn 差异)
    _W["solve"] = solve
    _W["n"] = n
    _W["timeout"] = timeout


def _solve_one(packed):
    """packed = (i, coords_list, demands_list, capacity) → (i, sol_str|None)。"""
    i, coords_list, demands_list, capacity = packed
    inst = {"n": _W["n"], "coords": np.asarray(coords_list, dtype=float),
            "demands": np.asarray(demands_list, dtype=float), "capacity": capacity}
    try:
        sol_str = _W["solve"]("cvrp", inst, timeout=_W["timeout"])
    except Exception:
        sol_str = None
    return i, sol_str


def main():
    ap = argparse.ArgumentParser(description="用 Mask 同一批 1000 条 CVRP 实例构建 FOARL RL 数据 (多进程)")
    ap.add_argument("--mask_dir", default="../UniCOP-Reason-Mask",
                    help="UniCOP-Reason-Mask 根 (import problems 用)")
    ap.add_argument("--distill_dir", default="../UniCOP-Distill",
                    help="UniCOP-Distill 根 (import lkh_solver 用, CVRP 走 PyVRP)")
    ap.add_argument("--out", default="data/foarl_cvrp20_mask1000.jsonl")
    ap.add_argument("--num", type=int, default=1000, help="实例数 (Mask: NUM_TRAIN=1000)")
    ap.add_argument("--seed", type=int, default=42, help="必须 = Mask config.data_seed (默认 42)")
    ap.add_argument("--n", type=int, default=20, help="客户数 (不含 depot); CVRP20 → 20")
    ap.add_argument("--k_nn", type=int, default=2, help="FOARL input 每点 k 近邻 (与 SFT 数据一致)")
    ap.add_argument("--timeout", type=int, default=5, help="PyVRP 每条实例 MaxRuntime 秒 (n=20 几秒足够近优)")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数; 0=自动(os.cpu_count); 登录节点别用满")
    ap.add_argument("--obj_decimals", type=int, default=2)
    ap.add_argument("--max_records", type=int, default=0, help="只做前 N 条 (验证用), 0=全量")
    args = ap.parse_args()

    mask_dir = os.path.abspath(args.mask_dir)
    distill_dir = os.path.abspath(args.distill_dir)
    self_dir = os.path.dirname(os.path.abspath(__file__))
    for p in (mask_dir, distill_dir, self_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from problems import get_problem          # Mask: 与 build_dataset 同一实例生成
    except Exception as e:
        sys.exit(f"[FATAL] 无法 import Mask 的 problems (mask_dir={mask_dir}): {e}")
    try:
        import lkh_solver  # noqa: F401           # 主进程先验一遍 (子进程会各自再 import)
    except Exception as e:
        sys.exit(f"[FATAL] 无法 import lkh_solver (distill_dir={distill_dir}): {e}")
    from build_foarl_cvrp_data import (build_instruction, build_input, dist_matrix,
                                       parse_routes, route_distance, check_feasible)

    n_total = args.max_records if args.max_records else args.num
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)

    # ── 1) 顺序生成全部实例 (保住与 Mask 完全一致的 rng 顺序; 这步极快) ─────────
    prob = get_problem("cvrp")
    rng = np.random.default_rng(args.seed)
    instances = []   # (i, coords, demands, capacity)
    for i in range(args.num):
        inst = prob.generate_instance(args.n, rng)   # ★ 推进 rng, 中间不插别的 rng 调用
        if args.max_records and i >= args.max_records:
            break
        instances.append((i, np.asarray(inst["coords"], float),
                          np.asarray(inst["demands"], float), float(inst["capacity"]),
                          [list(r) for r in inst["feasible_routes"]]))
    print(f"[1/3] 顺序生成 {len(instances)} 条实例 (seed={args.seed}, n={args.n}) 完成")

    # ── 2) 并行 PyVRP 求参考解 (慢路, 独立, 进程池并行) ──────────────────────────
    from multiprocessing import Pool
    packed = [(i, c.tolist(), d.tolist(), cap) for (i, c, d, cap, _fr) in instances]
    print(f"[2/3] {workers} 进程并行 PyVRP 求解 {len(packed)} 条 (timeout={args.timeout}s/条)...")
    sol_by_id = {}
    done = 0
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(distill_dir, args.n, args.timeout)) as pool:
        for i, sol_str in pool.imap_unordered(_solve_one, packed, chunksize=4):
            sol_by_id[i] = sol_str
            done += 1
            if done % 100 == 0:
                print(f"      ... {done}/{len(packed)}")

    # ── 3) 按原顺序组装 FOARL 记录并写出 ───────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_out = n_solver_fail = feas_ok = regex_ok = 0
    dists = []
    import ast, re
    _re_routes = re.compile(r"Routes:\s*\[\s*(.*)\]", re.DOTALL)
    _re_obj = re.compile(r"Objective:\s*([\d.]+)")

    with open(args.out, "w", encoding="utf-8") as fout:
        for (i, coords, demands, capacity, feasible_routes) in instances:
            sol_str = sol_by_id.get(i)
            routes = parse_routes(sol_str) if sol_str else None
            if routes is None:
                n_solver_fail += 1                 # PyVRP 失败 → 回退实例自带 greedy 可行解 (弱参考)
                routes = feasible_routes

            D = dist_matrix(coords)
            dist = float(sum(route_distance(r, D) for r in routes))
            dists.append(dist)
            if check_feasible(routes, demands, capacity, tol=0.05):
                feas_ok += 1

            instruction = build_instruction(args.n, capacity, args.k_nn)
            inp = build_input(coords, demands, D, args.k_nn)
            output = f"Routes: {routes}, Objective: {dist:.{args.obj_decimals}f}"

            rm, om = _re_routes.search(output), _re_obj.search(output)
            if rm and om:
                try:
                    back = ast.literal_eval(f"[{rm.group(1).strip()}]")
                    if all(isinstance(r, list) for r in back):
                        regex_ok += 1
                except (SyntaxError, ValueError):
                    pass

            fout.write(json.dumps({
                "id": i, "problem_type": "cvrp", "n": args.n,
                "instruction": instruction, "input": inp, "output": output,
                "instance": [coords.tolist(), demands.tolist(), capacity],
                "solver_distance": dist,
            }, ensure_ascii=False) + "\n")
            n_out += 1

    # ── 定量报告 ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  FOARL CVRP 数据 (Mask 同 1000 实例) 构建报告")
    print(f"  out: {args.out}  (workers={workers})")
    print("-" * 60)
    print(f"  写出:                {n_out}")
    print(f"  PyVRP 失败回退greedy: {n_solver_fail}  ({n_solver_fail/max(n_out,1):.1%})  ← 应≈0, 否则调大 --timeout")
    print(f"  [自检1] 参考解可行(容0.05): {feas_ok}/{n_out} = {feas_ok/max(n_out,1):.2%}  ← 应≈100%")
    print(f"  [自检2] output 可被 FOARL 正则解析: {regex_ok}/{n_out} = {regex_ok/max(n_out,1):.2%}")
    if dists:
        d = np.array(dists)
        print(f"  [自检3] 参考距离: mean={d.mean():.4f} min={d.min():.4f} max={d.max():.4f}")
    print("=" * 60)
    print("  注: 实例顺序生成→并行求解, 同 seed 重跑结果一致 (确定性)。")


if __name__ == "__main__":
    main()
