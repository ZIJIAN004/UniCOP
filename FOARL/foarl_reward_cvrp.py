#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
foarl_reward_cvrp.py — FOARL CVRP 规则奖励 (Stage-2 RL 用), 自洽无重依赖。

忠实复现 FOARL (LLMCoSolver, NeurIPS 2025, arXiv:2509.16865) 官方 rewards.py 的
CVRP 奖励 (Summer142857/LLMCoSolver: feasibility_reward_func_cvrp +
optimality_reward_func_cvrp), 把官方拆成两个 reward_func 的逻辑合并到一个函数里:

    R^P = R_f + R_o
      R_f = ω_parse·ζ + ω_depot·c_depot + ω_cov·c_cov + ω_cap·c_cap
      R_o = α / (1 + gap)        仅当 R_f ≥ 0.99 (完全可行) 时给, 否则 0
      gap = (pred_cost − ref_cost) / ref_cost      ★不下截到0 (官方原样)

官方 CVRP 四个分量 (权重见 rewards.py 的 weights dict, 论文附录 A.3.3 同值):
  - ζ        parse  : 能否 ast 解出 routes; 解不出 → 整个 R^P = 0          ω=0.2
  - c_depot  depot  : 二元(全体路线都以 0 起止 → 1, 否则 0)                ω=0.1
  - c_cov    coverage: 访问客户集 == {1..n} → 1, 否则按交集比例给部分分      ω=0.1
  - c_cap    capacity: 二元(每条路线载重都 ≤ capacity → 1, 否则 0)          ω=0.6
注意与早期版本的差异(已对齐官方):
  · depot 不再并入 capacity, 是独立分量;
  · c_depot / c_cap 是"全体路线"的二元判定(任一违反即 0), 非 per-route 比例;
  · c_cov 用 route[1:-1] (剥首尾 depot) 统计;
  · R_o 只在完全可行(R_f≥0.99)时给, 且 gap 不下截 → 比参考解更优时 R_o>α 也允许;
  · α=1.0 (官方 1/(1+gap) 隐式系数), 不是早期的 0.5。

输入 instance = build_foarl_cvrp_data.py 写出的 [coords, demands, capacity]:
  coords:   list[[x,y]]  含 depot(0), 共 n+1 个点
  demands:  list[float]  含 depot demand=0, 长度 n+1 → 客户为 1..n
  capacity: float
ref_distance: 该实例 solver 参考目标值 (gold "Objective: X"), 供 gap; None → 只给 R_f。

所有权重/α 都可由 train_grpo_foarl.py 透参覆盖。
"""
import ast
import math
import re

# 官方 parse_solution_vrp 的正则: 贪婪吃到最后一个 ']', 再 ast 还原
_RE_ROUTES = re.compile(r"Routes:\s*\[\s*(.*)\]", re.DOTALL)


def parse_solution_vrp(response: str):
    """忠实复刻官方 parse_solution_vrp: 成功返回 list[list[int]], 失败返回 None。"""
    m = _RE_ROUTES.search(response)
    if not m:
        return None
    routes_str = m.group(1).strip()
    try:
        routes = ast.literal_eval(f"[{routes_str}]")
    except (SyntaxError, ValueError):
        return None
    if not isinstance(routes, list) or not all(isinstance(r, list) for r in routes):
        return None
    return routes


def _route_distance(routes, coords):
    """按 coords 重算所有路线总欧氏距离 (越界节点跳过该步, 不抛异常)。"""
    n_pts = len(coords)
    total = 0.0
    for r in routes:
        for a, b in zip(r[:-1], r[1:]):
            if isinstance(a, int) and isinstance(b, int) and 0 <= a < n_pts and 0 <= b < n_pts:
                ax, ay = coords[a]
                bx, by = coords[b]
                total += math.hypot(ax - bx, ay - by)
    return total


def compute_foarl_reward_cvrp(
    completion: str,
    instance,                 # [coords, demands, capacity]
    ref_distance,             # float | None
    alpha: float = 1.0,
    omega_parse: float = 0.2,
    omega_depot: float = 0.1,
    omega_coverage: float = 0.1,
    omega_capacity: float = 0.6,
):
    """返回 (scalar_reward, components_dict)。components 供 trainer 端按维 log 命中率。"""
    coords, demands, capacity = instance[0], instance[1], instance[2]

    routes = parse_solution_vrp(completion)
    zeta = 1.0 if routes is not None else 0.0
    if zeta == 0.0:
        return 0.0, {"parse": 0.0, "depot": 0.0, "coverage": 0.0, "capacity": 0.0,
                     "feasible": 0.0, "R_f": 0.0, "R_o": 0.0, "gap": None, "dist": None}

    score = omega_parse  # 能解析 → 先得 parse 分

    # ── c_depot: 二元, 全体路线都以 depot(0) 起止 ──────────────────────────
    depot_ok = True
    for r in routes:
        if not r or r[0] != 0 or r[-1] != 0:
            depot_ok = False
            break
    c_depot = 1.0 if depot_ok else 0.0
    score += omega_depot * c_depot

    # ── c_cap: 二元, 每条路线载重都 ≤ capacity (官方: 任一超载即整体 0) ──────
    c_cap = 0.0
    try:
        capacity_ok = True
        for r in routes:
            load = sum(demands[v] for v in r if v != 0)
            if load > capacity:
                capacity_ok = False
                break
        c_cap = 1.0 if capacity_ok else 0.0
    except (IndexError, TypeError):
        c_cap = 0.0
    score += omega_capacity * c_cap

    # ── c_cov: 访问客户集 == {1..n}; 否则按交集比例给部分分 ────────────────
    c_cov = 0.0
    try:
        n_customers = len(demands)                 # 含 depot, 客户为 1..n_customers-1
        required = set(range(1, n_customers))
        visited = set()
        for r in routes:
            visited.update(r[1:-1])                 # 剥首尾 depot
        if visited == required:
            c_cov = 1.0
        elif required:
            c_cov = len(visited & required) / len(required)
    except (IndexError, TypeError):
        c_cov = 0.0
    score += omega_coverage * c_cov

    R_f = score
    feasible = 1.0 if R_f >= 0.99 else 0.0

    # ── R_o: 仅完全可行时给; gap 不下截; score 下截到 0 (官方原样) ──────────
    R_o = 0.0
    gap = None
    dist = None
    if feasible == 1.0 and ref_distance is not None and abs(ref_distance) > 1e-10:
        dist = _route_distance(routes, coords)
        gap = (dist - ref_distance) / ref_distance
        R_o = max(0.0, alpha / (1.0 + gap))

    return R_f + R_o, {"parse": zeta, "depot": c_depot, "coverage": c_cov, "capacity": c_cap,
                       "feasible": feasible, "R_f": R_f, "R_o": R_o, "gap": gap, "dist": dist}


if __name__ == "__main__":
    # 自检: 3 客户玩具实例, 验证 可行最优 / 超载 / 覆盖遗漏 / 解析失败 四种情形
    coords = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.2, 0.2]]
    demands = [0.0, 0.5, 0.5, 0.5]
    capacity = 1.0
    inst = [coords, demands, capacity]
    ref = _route_distance([[0, 1, 2, 0], [0, 3, 0]], coords)

    good = "Routes: [[0, 1, 2, 0], [0, 3, 0]], Objective: %.2f" % ref  # 可行最优 → R_f=1.0, R_o≈1.0
    overload = "Routes: [[0, 1, 2, 3, 0]], Objective: 0.80"           # 一条装 1.5>1.0 → cap=0
    miss = "Routes: [[0, 1, 0]], Objective: 0.20"                     # 漏 2,3 → cov 部分分, 不可行
    bad = "the answer is 42"                                          # 解析失败 → 全 0

    for name, comp in [("可行最优", good), ("容量超载", overload),
                       ("覆盖遗漏", miss), ("解析失败", bad)]:
        r, c = compute_foarl_reward_cvrp(comp, inst, ref)
        print(f"[{name}] R={r:.3f}  ζ={c['parse']} depot={c['depot']} cov={c['coverage']:.2f} "
              f"cap={c['capacity']} feas={c['feasible']} "
              f"R_f={c['R_f']:.3f} R_o={c['R_o']:.3f} gap={c['gap']}")
