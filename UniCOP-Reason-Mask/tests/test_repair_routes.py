"""
单元测试: terminal_reward.repair_routes + repaired_distance.

验证 v4 reward scheme 的核心 distance 排序 invariant:
    全访合规 < 全访违例 < 漏访
    重复访问通过 dup_eps 加固定罚 (不依赖几何位置)
    所有 trajectory 修复后都是完全可行解 (每条 demand ≤ cap, 全客户都访问)

跑法 (本地, 无需 GPU):
    cd UniCOP-Reason-Mask
    python -m tests.test_repair_routes
"""

import os
import sys
import types

# Allow running as standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub torch + pomo_prm (terminal_reward 顶层 import 链, 测试本身只用 numpy)
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
if "pomo_prm" not in sys.modules:
    _fake_pomo = types.ModuleType("pomo_prm")
    _fake_pomo.parse_route_numbers = lambda *a, **k: []
    sys.modules["pomo_prm"] = _fake_pomo

from terminal_reward import repair_routes, repaired_distance, _route_distance


def _make_instance(n=5):
    """简单 CVRP instance: depot 在原点, 5 客户在单位圆周等间隔, 各 demand=0.3."""
    import math
    coords = [(0.0, 0.0)]
    for i in range(n):
        angle = 2 * math.pi * i / n
        coords.append((math.cos(angle), math.sin(angle)))
    demands = [0.0] + [0.3] * n   # 每条路线最多装 cap/0.3 = 3 个客户 (cap=1.0)
    capacity = 1.0
    return coords, demands, capacity


def test_all_feasible_unchanged():
    """全访合规 routes, repair 后客户集合不变, distance 跟原 routes 一致."""
    coords, demands, cap = _make_instance(n=5)
    routes = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]   # demand 各 0.9, 0.6, 全合规
    repaired, n_dup = repair_routes(routes, n=5, demands=demands, capacity=cap)
    visited = {v for r in repaired for v in r if v != 0}
    assert visited == {1, 2, 3, 4, 5}, f"客户集合错: {visited}"
    assert n_dup == 0, f"无重复但 n_dup={n_dup}"
    # 每条路线 demand 合规
    for r in repaired:
        load = sum(demands[v] for v in r if v != 0)
        assert load <= cap + 1e-6, f"路线 {r} demand {load} 超容"
    dist = repaired_distance(routes, coords, n=5, demands=demands, capacity=cap)
    # 全访合规 distance 应该就是原 routes 的几何 distance
    orig_dist = sum(_route_distance(r, coords) for r in routes)
    assert abs(dist - orig_dist) < 1e-6, f"distance 不一致 dist={dist} orig={orig_dist}"
    print(f"[PASS] all_feasible_unchanged: dist={dist:.4f}")


def test_missing_customer():
    """漏访 1 客户, repair 后补一条 [0, c, 0] 路线, distance 增加."""
    coords, demands, cap = _make_instance(n=5)
    routes_full = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]
    routes_miss = [[0, 1, 2, 3, 0], [0, 4, 0]]  # 漏访 5
    repaired, n_dup = repair_routes(routes_miss, n=5, demands=demands, capacity=cap)
    visited = {v for r in repaired for v in r if v != 0}
    assert visited == {1, 2, 3, 4, 5}, f"漏访 5 应被补全, 实际 visited={visited}"
    assert n_dup == 0, f"无重复但 n_dup={n_dup}"
    dist_full = repaired_distance(routes_full, coords, n=5, demands=demands, capacity=cap)
    dist_miss = repaired_distance(routes_miss, coords, n=5, demands=demands, capacity=cap)
    assert dist_miss > dist_full, (
        f"漏访 distance 应 > 全访: miss={dist_miss:.4f} full={dist_full:.4f}"
    )
    print(f"[PASS] missing_customer: full={dist_full:.4f} miss={dist_miss:.4f} "
          f"增量={dist_miss - dist_full:.4f}")


def test_violate_capacity():
    """违例路线 (1 路线塞 4 客户共 1.2 demand) 应被拆成 2 条."""
    coords, demands, cap = _make_instance(n=5)
    routes = [[0, 1, 2, 3, 4, 0], [0, 5, 0]]   # 第 1 条 demand=1.2 超容
    repaired, n_dup = repair_routes(routes, n=5, demands=demands, capacity=cap)
    visited = {v for r in repaired for v in r if v != 0}
    assert visited == {1, 2, 3, 4, 5}, f"客户集合错: {visited}"
    assert n_dup == 0
    # 每条路线 demand 合规
    for r in repaired:
        load = sum(demands[v] for v in r if v != 0)
        assert load <= cap + 1e-6, f"路线 {r} demand {load} 仍超容"
    # 路线数应 ≥ 2 (原 1 违例条被拆)
    n_routes_with_customer = sum(1 for r in repaired if any(v != 0 for v in r))
    assert n_routes_with_customer >= 2
    print(f"[PASS] violate_capacity: {len(routes)} routes → {n_routes_with_customer} repaired")


def test_duplicate_customer():
    """重复访问 1 次, repair 后去重, n_dup=1, distance 加 dup_eps."""
    coords, demands, cap = _make_instance(n=5)
    routes_full = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]
    routes_dup  = [[0, 1, 2, 1, 0], [0, 4, 5, 3, 0]]  # 重复 1, 但其他不漏
    repaired, n_dup = repair_routes(routes_dup, n=5, demands=demands, capacity=cap)
    visited = {v for r in repaired for v in r if v != 0}
    assert visited == {1, 2, 3, 4, 5}, f"客户集合错 (去重 + 不漏): {visited}"
    assert n_dup == 1, f"重复 1 次但 n_dup={n_dup}"
    dist_full = repaired_distance(routes_full, coords, n=5, demands=demands,
                                   capacity=cap, dup_eps=0.2)
    dist_dup  = repaired_distance(routes_dup, coords, n=5, demands=demands,
                                   capacity=cap, dup_eps=0.2)
    # dup 至少多 dup_eps (实际可能再多因为修复后路线结构不同)
    assert dist_dup >= dist_full + 0.2 - 1e-6, (
        f"dup distance 应 ≥ full + dup_eps: dup={dist_dup} full={dist_full}"
    )
    print(f"[PASS] duplicate_customer: full={dist_full:.4f} dup={dist_dup:.4f} "
          f"增量={dist_dup - dist_full:.4f}")


def test_ordering_invariant():
    """核心 invariant: 全访合规 < 全访违例 < 漏访
    (其实违例可能 ≤ 漏访, 因为漏访补单条路线 distance 较高).
    """
    coords, demands, cap = _make_instance(n=5)
    routes_full = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]                # 全合规
    routes_violate = [[0, 1, 2, 3, 4, 0], [0, 5, 0]]             # 违例 1 条 (demand 1.2)
    routes_miss = [[0, 1, 2, 3, 0]]                              # 漏访 4,5
    d_full = repaired_distance(routes_full, coords, n=5, demands=demands, capacity=cap)
    d_violate = repaired_distance(routes_violate, coords, n=5, demands=demands, capacity=cap)
    d_miss = repaired_distance(routes_miss, coords, n=5, demands=demands, capacity=cap)
    assert d_full <= d_violate, f"全访合规 ({d_full}) 应 ≤ 违例 ({d_violate})"
    assert d_violate <= d_miss, f"违例 ({d_violate}) 应 ≤ 漏访 ({d_miss})"
    print(f"[PASS] ordering_invariant: full={d_full:.4f} ≤ "
          f"violate={d_violate:.4f} ≤ miss={d_miss:.4f}")


def test_mixed_errors():
    """同时存在 漏访 + 违例 + 重复, repair 应能处理."""
    coords, demands, cap = _make_instance(n=5)
    # 1 重复访问, 1 违例路线, 1 漏访 (5)
    routes = [[0, 1, 2, 3, 4, 0], [0, 1, 0]]  # 1 重复, 路线 1 demand=1.2 违例, 5 漏访
    repaired, n_dup = repair_routes(routes, n=5, demands=demands, capacity=cap)
    visited = {v for r in repaired for v in r if v != 0}
    assert visited == {1, 2, 3, 4, 5}, f"客户集合错: {visited}"
    assert n_dup == 1, f"应有 1 重复, 实际 n_dup={n_dup}"
    # 全部合规
    for r in repaired:
        load = sum(demands[v] for v in r if v != 0)
        assert load <= cap + 1e-6, f"路线 {r} demand {load} 仍超容"
    print(f"[PASS] mixed_errors: repaired into {len(repaired)} routes, n_dup={n_dup}")


def test_internal_depot_in_route():
    """单条 list 内有中间 depot ([0, 1, 2, 0, 3, 0]) 应被视作两条 sub-route.
    防止"1+2+3 demand 累加"被错误合并成一条 [0, 1, 2, 3, 0].
    """
    coords, demands, cap = _make_instance(n=5)
    # 1+2+3 demand 0.9 <= cap, 如果错误合并不会触发拆分但仍会丢失"路线 1 结束"语义
    # 正确处理: 切成 [0,1,2,0] + [0,3,0] 两条 sub-route
    routes = [[0, 1, 2, 0, 3, 0], [0, 4, 5, 0]]
    repaired, n_dup = repair_routes(routes, n=5, demands=demands, capacity=cap)
    visited = {v for r in repaired for v in r if v != 0}
    assert visited == {1, 2, 3, 4, 5}, f"客户集合错: {visited}"
    assert n_dup == 0, f"无重复但 n_dup={n_dup}"
    # 第一条 list 应被切成 2 条 sub-route, 总路线数 = 3
    assert len(repaired) == 3, f"内部 depot 应切路线, 期望 3 条, 实际 {len(repaired)}"
    # 找出含 customer 1 的路线, 应该不含 3 (因为 [0,1,2,0] 和 [0,3,0] 是两条)
    route_with_1 = next(r for r in repaired if 1 in r)
    assert 3 not in route_with_1, (
        f"customer 1 和 3 应分在两条路线, 实际同路线: {route_with_1}"
    )
    print(f"[PASS] internal_depot_in_route: 2 input lists → {len(repaired)} repaired "
          f"(切分对)")


def test_empty_routes():
    """空 routes (parse 后无任何客户) 应能处理, 全部当漏访补."""
    coords, demands, cap = _make_instance(n=5)
    routes = []
    repaired, n_dup = repair_routes(routes, n=5, demands=demands, capacity=cap)
    visited = {v for r in repaired for v in r if v != 0}
    assert visited == {1, 2, 3, 4, 5}, "空 routes 应全部补成单客户路线"
    assert n_dup == 0
    assert len(repaired) == 5, f"应是 5 条单客户路线, 实际 {len(repaired)}"
    print(f"[PASS] empty_routes: 5 routes补全")


if __name__ == "__main__":
    test_all_feasible_unchanged()
    test_missing_customer()
    test_violate_capacity()
    test_duplicate_customer()
    test_ordering_invariant()
    test_mixed_errors()
    test_internal_depot_in_route()
    test_empty_routes()
    print("\n全部 repair_routes 测试通过.")
