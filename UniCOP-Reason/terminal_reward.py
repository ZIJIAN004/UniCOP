"""
Terminal Reward：completion 级标量 reward，捕获"对不对/匹不匹配"。

四维加权求和（默认等权 1:1:1:1），每维 ∈ [0, 1]：
    R_parse       — parse 是否成功
    R_coverage    — 覆盖完整性 (hinge：全覆盖 AND 无重复 → 1，否则 0)
    R_constraint  — 约束满足率（部分得分，沿用 problems/ 内 c_core 公式）
    R_format      — Route N 编号正确率（多车问题：correct/total；单车恒 1）

返回值范围：[0, Σweights]，默认 [0, 4]。

通过 GRPO 组归一化后广播到 token，与 PRM 信号合成（Method A，双 loss）。
"""

import numpy as np

from utils.parse import parse_single_route, parse_multi_route
from pomo_prm import parse_route_numbers

# 复用各问题内置的约束模拟器，避免重复实现
from problems.tsptw import _simulate as _tsptw_sim
from problems.tspdl import _simulate as _tspdl_sim
from problems.vrptw import _route_feasible as _vrptw_route_ok


SINGLE_ROUTE = {"tsp", "tsptw", "tspdl"}
MULTI_ROUTE  = {"cvrp", "vrptw"}


def compute_terminal_components(
    completion: str,
    instance: dict,
    problem_type: str,
) -> dict:
    """
    返回四维原始分数 dict（每维 ∈ [0, 1]）：
        {"parse": ..., "coverage": ..., "constraint": ..., "format": ...}

    parse 失败时其他三项无意义，全部返回 0.0。
    便于 trainer 端按维度 log 命中率。
    """
    n = instance["n"]
    is_multi = problem_type in MULTI_ROUTE

    # ── R_parse ──────────────────────────────────────────────────────
    if is_multi:
        routes = parse_multi_route(completion, n)
        parse_ok = routes is not None
    else:
        route = parse_single_route(completion, n)
        parse_ok = route is not None

    if not parse_ok:
        return {"parse": 0.0, "coverage": 0.0, "constraint": 0.0, "format": 0.0}

    # ── R_coverage（hinge：全覆盖 AND 无重复） ─────────────────────
    # parser 已过滤 node > n，v != 0 过滤 depot，剩下都是 [1, n] 内的客户。
    # 两条件合起来等价于"客户访问序列排序后恰好是 [1, 2, ..., n]"。
    if is_multi:
        all_customer_visits = [v for r in routes for v in r if v != 0]
    else:
        all_customer_visits = [v for v in route if v != 0]
    unique_customers = set(all_customer_visits)
    R_coverage = 1.0 if (
        len(unique_customers)    == n and
        len(all_customer_visits) == n
    ) else 0.0

    R_constraint = _constraint_score(
        problem_type, routes if is_multi else route, instance,
    )
    R_format = _format_score(completion, problem_type)

    return {
        "parse":      1.0,
        "coverage":   R_coverage,
        "constraint": R_constraint,
        "format":     R_format,
    }


def compute_terminal_reward(
    completion: str,
    instance: dict,
    problem_type: str,
    w_parse: float = 1.0,
    w_coverage: float = 1.0,
    w_constraint: float = 1.0,
    w_format: float = 1.0,
) -> float:
    """
    四维加权求和。返回 scalar ∈ [0, w_parse + w_coverage + w_constraint + w_format]。
    """
    c = compute_terminal_components(completion, instance, problem_type)
    return (
        w_parse      * c["parse"] +
        w_coverage   * c["coverage"] +
        w_constraint * c["constraint"] +
        w_format     * c["format"]
    )


def _constraint_score(problem_type: str, route_or_routes, instance: dict) -> float:
    """
    各问题的 c_core 公式：
      TSP   → 1.0（无约束）
      TSPTW → satisfied / n        （部分得分，per-customer 时间窗满足）
      TSPDL → satisfied / n        （部分得分，per-customer draft limit 满足）
      CVRP  → valid_routes / total （per-route 二元，容量超限即整条路线作废）
      VRPTW → valid_routes / total （per-route 二元，时窗违例即整条路线作废）
    """
    n = instance["n"]
    coords = instance["coords"]

    if problem_type == "tsp":
        return 1.0

    if problem_type == "tsptw":
        tw = instance["time_windows"]
        satisfied, _ = _tsptw_sim(route_or_routes, coords, tw)
        return satisfied / n

    if problem_type == "tspdl":
        demands = instance["demands"]
        dl      = instance["draft_limits"]
        cap     = instance["capacity"]
        satisfied, _ = _tspdl_sim(route_or_routes, coords, demands, dl, cap)
        return satisfied / n

    if problem_type == "cvrp":
        demands = instance["demands"]
        cap     = instance["capacity"]
        if not route_or_routes:
            return 0.0
        valid = sum(
            1 for r in route_or_routes
            if sum(demands[v] for v in r if v != 0) <= cap + 1e-6
        )
        return valid / len(route_or_routes)

    if problem_type == "vrptw":
        tw = instance["time_windows"]
        if not route_or_routes:
            return 0.0
        valid = sum(1 for r in route_or_routes if _vrptw_route_ok(r, coords, tw))
        return valid / len(route_or_routes)

    return 1.0


def is_fully_feasible(completion: str, instance: dict, problem_type: str) -> bool:
    """parse + coverage + constraint + format 全部满足才算可行。"""
    c = compute_terminal_components(completion, instance, problem_type)
    return (c["parse"] == 1.0 and c["coverage"] == 1.0
            and c["constraint"] == 1.0 and c["format"] == 1.0)


def _format_score(completion: str, problem_type: str) -> float:
    """
    Route N 编号正确率：
      单路线问题 (TSP/TSPTW/TSPDL)：恒为 1（无编号）
      多路线问题 (CVRP/VRPTW)    ：正确编号数 / 总编号数
        - 第 idx 条 Route 的期望编号 = idx + 1
        - 若 parse_route_numbers 抓不到任何 Route N: → 0（虽然 parse_multi_route
          已成功，但意味着模型用的不是 Route N: 这种格式，仍算违规）
    """
    if problem_type in SINGLE_ROUTE:
        return 1.0

    route_numbers = parse_route_numbers(completion)
    if not route_numbers:
        return 0.0

    correct = sum(
        1 for idx, (n_value, _) in enumerate(route_numbers)
        if n_value == idx + 1
    )
    return correct / len(route_numbers)
