#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_foarl_cvrp_data_mask1000.py — 用 UniCOP-Reason-Mask 完全相同的 1000 条 CVRP 实例
构建 FOARL RL 数据 (FOARL 自己的 prompt/输出格式 + PyVRP 参考距离)。

目的: FOARL 作为 baseline 与 Mask 做增益对比时, 两边必须在**同一批优化实例**上训练。
Mask 的训练实例是程序化生成的 (data/generate.py:build_dataset, 非固定文件):
    rng = np.random.default_rng(seed=42)
    for _ in range(num):  instance = CVRP.generate_instance(n=20, rng)
本脚本**逐字复刻**这一生成过程 (同一函数、同一 seed、同一调用顺序) → 得到与 Mask
逐条一致的 coords/demands/capacity。

为什么能对齐: build_dataset 里 rng 只被 generate_instance 推进 (build_prompt/to_json 不用 rng),
所以用同一个 default_rng(42) 顺序调用 generate_instance(20, rng) 1000 次, 必然复现同一序列。
solve() 内部 PyVRP 用自己的 RNG, 不碰这里的 numpy rng, 不破坏顺序。

与 Mask 的差异 (方法固有, 不是数据差异):
  - 输出格式: Mask 是带 <think> 的 "Route N: 0->..->0" 多行思维式; FOARL 用单行非思维
    "Routes: [[0,..,0],..], Objective: X" —— 各自方法自带, 这里走 FOARL 格式。
  - 参考解: Mask 在线用 POMO PRM 不需要 gold; FOARL 的 R_o 需要每条实例的参考距离,
    用团队 CVRP 既有 solver = PyVRP/HGS (lkh_solver.solve('cvrp',·) → _solve_cvrp), 与
    chains 的 solver_distance 同源口径。

输出每条 (与 build_foarl_cvrp_data.py 同 schema, 多一个 solver_distance):
  {id, problem_type, n, instruction, input, output, instance:[coords,demands,capacity], solver_distance}

用法 (须在装了 pyvrp 且能 import Mask 的 problems 的环境/主机上跑, 如 zhuoyi unicop env):
  python build_foarl_cvrp_data_mask1000.py \
    --mask_dir   ../UniCOP-Reason-Mask \
    --distill_dir ../UniCOP-Distill \
    --out data/foarl_cvrp20_mask1000.jsonl \
    --num 1000 --seed 42 --n 20 --k_nn 2 --timeout 5
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="用 Mask 同一批 1000 条 CVRP 实例构建 FOARL RL 数据")
    ap.add_argument("--mask_dir", default="../UniCOP-Reason-Mask",
                    help="UniCOP-Reason-Mask 根 (import problems / data.generate 用)")
    ap.add_argument("--distill_dir", default="../UniCOP-Distill",
                    help="UniCOP-Distill 根 (import lkh_solver 用, CVRP 会走 PyVRP)")
    ap.add_argument("--out", default="data/foarl_cvrp20_mask1000.jsonl")
    ap.add_argument("--num", type=int, default=1000, help="实例数 (Mask: config.num_train 经 NUM_TRAIN=1000)")
    ap.add_argument("--seed", type=int, default=42, help="必须 = Mask config.data_seed (默认 42)")
    ap.add_argument("--n", type=int, default=20, help="客户数 (不含 depot); Mask CVRP20 → 20")
    ap.add_argument("--k_nn", type=int, default=2, help="FOARL input 里每点 k 近邻 (与 SFT 数据一致)")
    ap.add_argument("--timeout", type=int, default=5, help="PyVRP 每条实例的 MaxRuntime 秒 (n=20 几秒足够近优)")
    ap.add_argument("--obj_decimals", type=int, default=2)
    ap.add_argument("--max_records", type=int, default=0, help="只做前 N 条 (验证用), 0=全量")
    args = ap.parse_args()

    # ── import: Mask 的实例生成 + 团队 solver + FOARL 数据格式工具 ───────────────
    mask_dir = os.path.abspath(args.mask_dir)
    distill_dir = os.path.abspath(args.distill_dir)
    self_dir = os.path.dirname(os.path.abspath(__file__))
    for p in (mask_dir, distill_dir, self_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from problems import get_problem            # Mask: 与 build_dataset 同一实例生成
    except Exception as e:
        sys.exit(f"[FATAL] 无法 import Mask 的 problems (mask_dir={mask_dir}): {e}\n"
                 f"        确认 --mask_dir 指向 UniCOP-Reason-Mask, 且当前环境能跑它 (numpy/datasets)。")
    try:
        from lkh_solver import solve                # 团队 solver: CVRP → PyVRP/HGS
    except Exception as e:
        sys.exit(f"[FATAL] 无法 import lkh_solver (distill_dir={distill_dir}): {e}")
    # FOARL 自己的 instruction/input/距离工具 (同目录 build_foarl_cvrp_data.py)
    from build_foarl_cvrp_data import (build_instruction, build_input, dist_matrix,
                                       parse_routes, route_distance, check_feasible)

    if not os.environ.get("LKH_BIN"):
        print("[WARN] 未设 LKH_BIN; CVRP 走 PyVRP 不需要 LKH, 可忽略 (TSP 才用)。")

    prob = get_problem("cvrp")
    rng = np.random.default_rng(args.seed)   # ★ 与 Mask build_dataset 同 seed/同顺序

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_out = 0
    n_solver_fail = 0      # PyVRP 没解出 → 回退 greedy feasible_routes 当参考
    feas_ok = 0
    regex_ok = 0
    dists = []
    import ast, re
    _re_routes = re.compile(r"Routes:\s*\[\s*(.*)\]", re.DOTALL)
    _re_obj = re.compile(r"Objective:\s*([\d.]+)")

    total = args.max_records if args.max_records else args.num
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in range(args.num):
            if args.max_records and i >= args.max_records:
                break   # 截断只为验证; 前 max_records 条与全量逐条一致 (rng 顺序未变)
            inst = prob.generate_instance(args.n, rng)   # ★ 推进 rng, 不要在中间插别的 rng 调用

            coords = np.asarray(inst["coords"], dtype=float)
            demands = np.asarray(inst["demands"], dtype=float)
            capacity = float(inst["capacity"])

            # ── 参考解: 团队 CVRP solver = PyVRP/HGS ────────────────────────────
            sol_inst = {"n": args.n, "coords": coords, "demands": demands, "capacity": capacity}
            sol_str = solve("cvrp", sol_inst, timeout=args.timeout)
            routes = parse_routes(sol_str) if sol_str else None
            if routes is None:
                # 回退: 用实例自带 greedy 可行解 (弱参考, 仅救少量 PyVRP 失败, 保证 1000 条不缺)
                n_solver_fail += 1
                routes = [list(r) for r in inst["feasible_routes"]]

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
                "id": i,
                "problem_type": "cvrp",
                "n": args.n,
                "instruction": instruction,
                "input": inp,
                "output": output,
                "instance": [coords.tolist(), demands.tolist(), capacity],
                "solver_distance": dist,
            }, ensure_ascii=False) + "\n")
            n_out += 1
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{total}  (PyVRP 失败回退 {n_solver_fail})")

    # ── 定量报告 ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  FOARL CVRP 数据 (Mask 同 1000 实例) 构建报告")
    print(f"  out: {args.out}")
    print("-" * 60)
    print(f"  写出:              {n_out}")
    print(f"  PyVRP 失败回退greedy: {n_solver_fail}  ({n_solver_fail/max(n_out,1):.1%})  ← 应≈0")
    print(f"  [自检1] 参考解可行(容0.05): {feas_ok}/{n_out} = {feas_ok/max(n_out,1):.2%}  ← 应≈100%")
    print(f"  [自检2] output 可被 FOARL 正则解析: {regex_ok}/{n_out} = {regex_ok/max(n_out,1):.2%}")
    if dists:
        d = np.array(dists)
        print(f"  [自检3] 参考距离: mean={d.mean():.4f} min={d.min():.4f} max={d.max():.4f}")
    print("=" * 60)
    print("  提醒: 用同 seed 重跑应逐字节一致 (确定性)。若 PyVRP 回退>0, 调大 --timeout 重跑。")


if __name__ == "__main__":
    main()
