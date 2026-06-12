#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
foarl_reward_cvrp.py — FOARL CVRP 规则奖励 (Stage-2 RL 用), 自洽无重依赖。

复现 FOARL (LLMCoSolver, NeurIPS 2025, arXiv:2509.16865) 的可行性+最优性奖励:

    R^P = R_f + R_o
      R_f = ω0·ζ + Σ_i ω_i·c_i      (ζ != 0 时, 否则整个 R^P = 0)
      R_o = α / (1 + gap)           (ζ != 0 时, 否则 0)
      gap = max(0, (f(x̂) − f(x*)) / |f(x*)|)

CVRP 的两条约束 c_i (与本团队 UniCOP-Reason/terminal_reward.py 口径一致):
  - c_cov  覆盖率  : n_unique / max(n, n_total)   连续, 同时惩罚遗漏与重复
  - c_cap  容量    : valid_routes / total_routes  per-route 二元, 超载整条作废

ζ (format gate): completion 能否被 "Routes: [[...]], Objective: X" 正则+ast 解回。
  解不出 → ζ=0 → R^P=0 (格式不合规直接归零, 与论文一致)。

输入的 instance 是 build_foarl_cvrp_data.py 写出的 [coords, demands, capacity]:
  coords:   list[[x,y]]  含 depot(0) 共 n+1 个点, [0,1] 浮点尺度
  demands:  list[float]  含 depot demand=0
  capacity: float
ref_distance: 该实例 solver(near-opt) 目标值, 用于 gap; None 则只给 R_f。

默认权重沿用本团队既有 FOARL 配置 (UniCOP-Reason/foarl_reward.py):
  ω_parse(ζ)=0.2, ω_cov=0.3, ω_cap=0.3, ω_format=0.2(此处 format 并入 ζ, 见下), α=0.5
注: 本数据的解是单行 "Routes: [[...]]" (非 "Route N:" 多行), 故论文 CVRP 的
    "Route 编号正确率" 这一维退化为 ζ 的一部分, 不再单列, 把它的权重并回 ω_parse。
所有权重/α 都可由 train_grpo_foarl.py 透参覆盖, 便于对齐论文附录 A 的具体取值。
"""
import ast
import math
import re

_RE_ROUTES = re.compile(r"Routes:\s*(\[\[.*?\]\])", re.DOTALL)
_RE_OBJ = re.compile(r"Objective:\s*([-\d.]+)")


def parse_routes_and_obj(completion: str):
    """从 completion 解析 (routes:list[list[int]], obj:float|None)。失败返回 (None, None)。"""
    rm = _RE_ROUTES.search(completion)
    if not rm:
        return None, None
    try:
        routes = ast.literal_eval(rm.group(1))
    except (SyntaxError, ValueError):
        return None, None
    if not isinstance(routes, list) or not routes or not all(isinstance(r, list) for r in routes):
        return None, None
    # 路线元素必须全是整数节点
    try:
        routes = [[int(v) for v in r] for r in routes]
    except (TypeError, ValueError):
        return None, None
    om = _RE_OBJ.search(completion)
    obj = None
    if om:
        try:
            obj = float(om.group(1))
        except ValueError:
            obj = None
    return routes, obj


def _route_distance(routes, coords):
    """按 coords 重算所有路线总距离 (欧氏)。越界节点跳过该步, 不抛异常。"""
    n_pts = len(coords)
    total = 0.0
    for r in routes:
        for a, b in zip(r[:-1], r[1:]):
            if 0 <= a < n_pts and 0 <= b < n_pts:
                ax, ay = coords[a]
                bx, by = coords[b]
                total += math.hypot(ax - bx, ay - by)
    return total


def compute_foarl_reward_cvrp(
    completion: str,
    instance,                 # [coords, demands, capacity]
    ref_distance,             # float | None
    alpha: float = 0.5,
    omega_parse: float = 0.2,
    omega_coverage: float = 0.3,
    omega_capacity: float = 0.3,
    omega_format: float = 0.2,   # 并入 ζ (本数据无 "Route N:" 编号), 默认加到 parse 项
):
    """返回 (scalar_reward, components_dict)。components 供 trainer 端按维 log 命中率。"""
    coords, demands, capacity = instance[0], instance[1], instance[2]
    n = len(coords) - 1   # 客户数 (排除 depot)

    routes, _obj = parse_routes_and_obj(completion)
    zeta = 1.0 if routes is not None else 0.0
    if zeta == 0.0:
        return 0.0, {"parse": 0.0, "coverage": 0.0, "capacity": 0.0,
                     "feasible": 0.0, "R_f": 0.0, "R_o": 0.0, "gap": None, "dist": None}

    # ── c_cov: 覆盖率 (连续, 同时惩罚遗漏/重复) ──────────────────────────
    customer_visits = [v for r in routes for v in r if v != 0]
    n_unique = len(set(v for v in customer_visits if 1 <= v <= n))
    n_total = len(customer_visits)
    c_cov = n_unique / max(n, n_total) if max(n, n_total) > 0 else 0.0

    # ── c_cap: 容量满足率 (per-route 二元: 单条超载即整条作废) ───────────
    #    同时要求路线以 depot 起止, 否则该条不计为 valid (depot 闭合是 CVRP 硬约束)。
    if routes:
        valid = 0
        for r in routes:
            depot_ok = len(r) >= 2 and r[0] == 0 and r[-1] == 0
            load = sum(demands[v] for v in r if 0 <= v < len(demands) and v != 0)
            if depot_ok and load <= capacity + 1e-6:
                valid += 1
        c_cap = valid / len(routes)
    else:
        c_cap = 0.0

    # ω_format 并入 ζ 项 (见模块 docstring 说明)
    R_f = (omega_parse + omega_format) * zeta + omega_coverage * c_cov + omega_capacity * c_cap

    # ── R_o: 最优性 (gap 相对 solver 参考解), 仅 ζ!=0 时给 ──────────────
    R_o = 0.0
    gap = None
    dist = None
    if ref_distance is not None and abs(ref_distance) > 1e-10:
        dist = _route_distance(routes, coords)
        gap = max(0.0, (dist - ref_distance) / abs(ref_distance))
        R_o = alpha / (1.0 + gap)

    feasible = 1.0 if (abs(c_cov - 1.0) < 1e-9 and abs(c_cap - 1.0) < 1e-9) else 0.0
    return R_f + R_o, {"parse": zeta, "coverage": c_cov, "capacity": c_cap,
                       "feasible": feasible, "R_f": R_f, "R_o": R_o,
                       "gap": gap, "dist": dist}


if __name__ == "__main__":
    # 自检: 构造一个 3 客户玩具实例, 验证 可行/超载/解析失败 三种情形的奖励
    coords = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.2, 0.2]]
    demands = [0.0, 0.5, 0.5, 0.5]
    capacity = 1.0
    inst = [coords, demands, capacity]
    ref = _route_distance([[0, 1, 2, 0], [0, 3, 0]], coords)

    good = "Routes: [[0, 1, 2, 0], [0, 3, 0]], Objective: %.2f" % ref
    overload = "Routes: [[0, 1, 2, 3, 0]], Objective: 0.80"   # 一条路装 1.5 > 1.0
    miss = "Routes: [[0, 1, 0]], Objective: 0.20"             # 漏了 2,3
    bad = "the answer is 42"                                  # 解析失败

    for name, comp in [("可行最优", good), ("容量超载", overload),
                        ("覆盖遗漏", miss), ("解析失败", bad)]:
        r, c = compute_foarl_reward_cvrp(comp, inst, ref)
        print(f"[{name}] R={r:.3f}  ζ={c['parse']} cov={c['coverage']:.2f} "
              f"cap={c['capacity']:.2f} feas={c['feasible']} "
              f"R_f={c['R_f']:.3f} R_o={c['R_o']:.3f} gap={c['gap']}")
