"""
Terminal Reward：completion 级标量 reward，捕获"对不对/匹不匹配"。

四维加权求和（默认等权 1:1:1:1），每维 ∈ [0, 1]：
    R_parse       — parse 是否成功
    R_coverage    — 覆盖完整性，连续: n_unique / max(n, n_total)
                    全覆盖无重复 → 1.0; 遗漏或重复都按比例扣分
    R_constraint  — 约束满足率（部分得分，沿用 problems/ 内 c_core 公式）
    R_format      — Route N 编号正确率（多车问题：correct/total；单车恒 1）

A_feasibility 用 cov_gate 硬墙: cov 达不到阈值时 cons 信号完全关闭, 强制模型
优先把 cov 推到 gate, 再优化 cons. gate=1.0 时严格要求全覆盖+无重复.
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

    # ── R_coverage（连续: 同时惩罚遗漏和重复） ────────────────────
    # parser 已过滤 node > n，v != 0 过滤 depot，剩下都是 [1, n] 内的客户。
    # R_cov = n_unique / max(n, n_total):
    #   - 全覆盖无重复 (unique==n, total==n) → 1.0
    #   - 遗漏 K 客户 (unique=n-K, total=n-K)  → (n-K)/n
    #   - 重复 K 步   (unique=n,   total=n+K)  → n/(n+K)
    # 连续化让模型在 cov<1 时仍有 gradient 推进, 摆脱旧 hinge 0/1 卡死.
    if is_multi:
        all_customer_visits = [v for r in routes for v in r if v != 0]
    else:
        all_customer_visits = [v for v in route if v != 0]
    n_unique = len(set(all_customer_visits))
    n_total  = len(all_customer_visits)
    R_coverage = n_unique / max(n, n_total) if max(n, n_total) > 0 else 0.0

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


def compute_a_feasibility(
    completion: str,
    instance: dict,
    problem_type: str,
    w_p: float = 1.0,
    w_cov: float = 1.5,
    w_cons: float = 1.0,
    w_f: float = 0.5,
    cov_gate: float = 1.0,
) -> float:
    """
    A_feasibility = w_p*parse + w_cov*cov + w_cons*cons*gate + w_f*format

    cov_gate 硬墙: cov < gate 时 cons 信号置 0, 防"丢覆盖换约束"hack.
    gate=1.0 时严格要求全覆盖+无重复才解锁 cons (用户当前默认).
    gate=0.95 时允许 ≤1 客户误差就开门, 边界更平滑.

    满分 = w_p + w_cov + w_cons + w_f (默认 4.0, cov=1 且 cons=1 时拿满).
    """
    c = compute_terminal_components(completion, instance, problem_type)
    cons_signal = c["constraint"] if c["coverage"] >= cov_gate else 0.0
    return (w_p * c["parse"]
            + w_cov * c["coverage"]
            + w_cons * cons_signal
            + w_f * c["format"])


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


def is_fully_feasible(
    completion: str, instance: dict, problem_type: str,
    cov_gate: float = 1.0,
) -> bool:
    """parse + coverage(>=cov_gate) + constraint + format 全部满足才算可行。

    cov_gate=1.0 时跟旧版语义等价 (cov 连续后, ==1.0 仍要求 unique==n AND total==n).
    """
    c = compute_terminal_components(completion, instance, problem_type)
    return (c["parse"] == 1.0 and c["coverage"] >= cov_gate
            and c["constraint"] == 1.0 and c["format"] == 1.0)


## ══════════════════════════════════════════════════════════════════════
##  v4: Repaired Distance (simplified reward scheme)
##  把任意 trajectory (含漏访/违例/重复) 修复成完全可行解再算 distance,
##  让 outcome z-score 自然惩罚三类错误, 不依赖 hardgate / PRM cascade.
##  仅在 config.reward_scheme == "v4" 时被 trainer 调用.
## ══════════════════════════════════════════════════════════════════════


def repair_routes(routes, n: int, demands, capacity: float, dup_eps: float = 0.2):
    """把 routes 修复成完全可行: 违例贪心拆分 + 漏访补单条路线 + 重复去重.

    Args:
        routes: list[list[int]], 原始路线 (可能含违例/漏访/重复)
        n: 客户数 (不含 depot)
        demands: list[float] 长度 n+1, demands[0]=0
        capacity: float, 单路线容量上限
        dup_eps: 每次重复访问给的固定 distance 增量 (in repaired_distance 里加)

    Returns:
        (repaired_routes: list[list[int]], n_duplicates: int)
        - repaired_routes 完全可行: 每条 demand ≤ capacity, 全部客户 1..n 出现一次
        - n_duplicates: 重复次数 (1 个客户访问 K 次算 K-1 个 duplicate)
    """
    # 1. 统计重复 + 去重 (保留每个客户第一次出现)
    seen = set()
    n_duplicates = 0
    deduped = []
    for r in routes:
        new_r = []
        for v in r:
            if v == 0:
                new_r.append(v)
                continue
            if v in seen:
                n_duplicates += 1  # 重复, 不放进 deduped
                continue
            if v < 1 or v > n:
                # 越界, 视作 noise 跳过 (parse 已过滤但保险起见)
                continue
            seen.add(v)
            new_r.append(v)
        # 清理可能的连续 depot (0,0)
        cleaned = []
        for v in new_r:
            if v == 0 and cleaned and cleaned[-1] == 0:
                continue
            cleaned.append(v)
        if len(cleaned) > 0 and any(v != 0 for v in cleaned):
            deduped.append(cleaned)

    # 2. 违例贪心拆分: 累加 demand, 超 cap 就开新路线.
    #    内部 depot (中途 return depot) 也显式切路线 + reset load,
    #    防止"一条 list 内多个 0"被错误合并.
    repaired = []
    for r in deduped:
        current = [0]
        load = 0.0
        for v in r:
            if v == 0:
                # 内部 depot: 关闭当前路线, 起新路线 (reset load)
                if len(current) > 1:
                    current.append(0)
                    if len(current) > 2:
                        repaired.append(current)
                current = [0]
                load = 0.0
                continue
            d = demands[v] if v < len(demands) else 0.0
            if load + d > capacity + 1e-6:
                # 容量超: 强制关闭, 开新路线
                current.append(0)
                if len(current) > 2:
                    repaired.append(current)
                current = [0, v]
                load = d
            else:
                current.append(v)
                load += d
        # 关闭最后一条
        if len(current) > 1:
            current.append(0)
            if len(current) > 2:
                repaired.append(current)

    # 3. 漏访客户补单条路线 (保证 capacity 永远合规, 因为单 customer demand < cap)
    visited = {v for r in repaired for v in r if v != 0}
    missing = sorted(set(range(1, n + 1)) - visited)
    for c in missing:
        repaired.append([0, c, 0])

    return repaired, n_duplicates


def _route_distance(route, coords) -> float:
    """单条路线的几何 distance (欧氏距离). route 是 [0, c1, c2, ..., 0]."""
    if len(route) < 2:
        return 0.0
    coords = np.asarray(coords)
    total = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        total += float(np.linalg.norm(coords[a] - coords[b]))
    return total


def repaired_distance(
    routes,
    coords,
    n: int,
    demands,
    capacity: float,
    dup_eps: float = 0.2,
) -> float:
    """v4 outcome distance: 修复后几何 distance + n_duplicates × dup_eps.

    返回值越小越好 (跟原 get_tour_distance 方向一致).
    Trainer 端用 -repaired_distance 做 z-score, distance 短 → 正信号.

    设计 invariant:
        全访合规 distance < 全访违例 distance < 漏访 distance
        (因为拆分代价 < 漏访补全代价, 经验上 0.3-0.5 vs 0.5-1.0)
    """
    repaired, n_dup = repair_routes(routes, n, demands, capacity, dup_eps)
    geom_dist = sum(_route_distance(r, coords) for r in repaired)
    return geom_dist + n_dup * dup_eps


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
