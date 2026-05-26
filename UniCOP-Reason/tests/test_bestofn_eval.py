"""
expected_best_of_k / bestofn_scaling_curve 纯逻辑单元测试 (无 torch/POMO/GPU).

用手算可验证的小用例:
  全可行 [1,2,3], N=3:  k=1→2.0, k=2→4/3, k=3→1.0
  含 1 条不可行, 可行 [1,2], N=3:  k=1→1.5(p_feas=2/3), k=2→4/3(p_feas=1)

跑法:  python tests/test_bestofn_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bestofn_eval import expected_best_of_k, bestofn_scaling_curve, dist_at_budget


def _close(a, b, eps=1e-9):
    return abs(a - b) < eps


def test_all_feasible():
    feas = [1.0, 2.0, 3.0]
    e1, p1 = expected_best_of_k(feas, 3, 1)
    assert _close(e1, 2.0) and _close(p1, 1.0), (e1, p1)        # 均值
    e2, p2 = expected_best_of_k(feas, 3, 2)
    assert _close(e2, 4 / 3) and _close(p2, 1.0), (e2, p2)      # {1,2}→1,{1,3}→1,{2,3}→2
    e3, p3 = expected_best_of_k(feas, 3, 3)
    assert _close(e3, 1.0) and _close(p3, 1.0), (e3, p3)        # 全取 → min=1
    print("test_all_feasible PASS")


def test_with_infeasible():
    feas = [1.0, 2.0]      # 第 3 条不可行
    e1, p1 = expected_best_of_k(feas, 3, 1)
    assert _close(e1, 1.5) and _close(p1, 2 / 3), (e1, p1)      # 条件均值(1+2)/2, 可行率2/3
    e2, p2 = expected_best_of_k(feas, 3, 2)
    assert _close(e2, 4 / 3) and _close(p2, 1.0), (e2, p2)      # k=2 必含≥1可行
    print("test_with_infeasible PASS")


def test_k_out_of_range():
    e, p = expected_best_of_k([1.0], 1, 2)   # k>n_total
    assert e is None and p == 0.0, (e, p)
    e, p = expected_best_of_k([], 3, 1)      # 无可行
    assert e is None, (e, p)
    print("test_k_out_of_range PASS")


def test_scaling_curve_monotone():
    # 2 实例, 各 N=4 全可行; best-of-k 应随 k 单调不增, k=N 时 = 各实例最小的均值
    per_instance = [([1.0, 2.0, 3.0, 4.0], 4), ([2.0, 4.0, 6.0, 8.0], 4)]
    curve = bestofn_scaling_curve(per_instance, mean_tokens=100.0,
                                  n_instances=2, N=4)
    dists = [pt["avg_best_dist"] for pt in curve]
    assert all(dists[i] >= dists[i + 1] - 1e-12 for i in range(len(dists) - 1)), dists
    assert _close(curve[-1]["avg_best_dist"], (1.0 + 2.0) / 2), curve[-1]  # k=4 → 各实例 min 均值
    assert curve[0]["compute"] == 1 * 100.0 * 2, curve[0]["compute"]       # k=1 总算力
    assert curve[-1]["compute"] == 4 * 100.0 * 2, curve[-1]["compute"]     # k=4 = 全量
    # dist_at_budget: 预算够 k=2 (compute=400) 但不够 k=3(600)
    at = dist_at_budget(curve, 500)
    assert _close(at, curve[1]["avg_best_dist"]), at
    print("test_scaling_curve_monotone PASS")


if __name__ == "__main__":
    test_all_feasible()
    test_with_infeasible()
    test_k_out_of_range()
    test_scaling_curve_monotone()
    print("\nALL TESTS OK")
