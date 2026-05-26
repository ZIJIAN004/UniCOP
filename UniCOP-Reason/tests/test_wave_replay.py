"""
simulate_wave 纯逻辑单元测试 (不依赖 torch/POMO/GPU).

构造一个 n=20、M=8 的合成场景, 检查点 5/10/15/20, halve@10/15, keep=0.5,
逐项断言: 硬过滤、POMO 留一半的淘汰顺序、token 记账、终点最优挑选、
以及 baseline anytime 曲线 + 同算力读数.

跑法:  python tests/test_wave_replay.py     (全部 PASS 会打印 OK)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_replay import (
    WaveConfig, TrajProfile, simulate_wave,
    baseline_anytime_curve, baseline_dist_at_budget,
)


def _build_profiles():
    """8 条合成链:
      0: 早期硬违约 (n_feas=3<5), 违约 token=100, full=1000  → 25% 删, cost=100
      1: 可行但没凑够 (n_feas=4<5, 未违约), full=1000         → 25% 删, cost=1000(跑满)
      2-7: 全程 (n_feas=20), POMO 值 traj2 最好…traj7 最差
           tokens_at {5:200,10:400,15:600}, full=1000
    POMO 淘汰: 10% 留 ceil(6*.5)=3 → {2,3,4}; 15% 留 ceil(3*.5)=2 → {2,3}
      → 5,6,7 cost=400(@10); 4 cost=600(@15); 2,3 cost=1000(满)
    """
    P = []
    P.append(TrajProfile(idx=0, full_tokens=1000, n_feasible_customers=3,
                         violated=True, violation_tokens=100,
                         tokens_at={}, pomo_at={},
                         final_feasible=False, final_distance=None))
    P.append(TrajProfile(idx=1, full_tokens=1000, n_feasible_customers=4,
                         violated=False, violation_tokens=1000,
                         tokens_at={}, pomo_at={},
                         final_feasible=False, final_distance=None))
    # POMO 值越大越好; final_distance 越小越好
    pomo10 = {2: -5, 3: -6, 4: -7, 5: -8, 6: -9, 7: -10}
    pomo15 = {2: -5, 3: -6, 4: -7, 5: -8, 6: -9, 7: -10}
    final = {2: 3.5, 3: 3.2, 4: 3.0, 5: 4.0, 6: 3.8, 7: 5.0}
    for i in range(2, 8):
        P.append(TrajProfile(
            idx=i, full_tokens=1000, n_feasible_customers=20,
            violated=False, violation_tokens=1000,
            tokens_at={5: 200, 10: 400, 15: 600},
            pomo_at={10: pomo10[i], 15: pomo15[i]},
            final_feasible=True, final_distance=final[i],
        ))
    return P


def test_simulate_wave():
    profiles = _build_profiles()
    cfg = WaveConfig(checkpoint_fracs=(0.25, 0.5, 0.75, 1.0),
                     halve_fracs=(0.5, 0.75), keep_fraction=0.5)
    res = simulate_wave(profiles, n=20, cfg=cfg)

    # ── token 记账逐条 ──
    assert res.consumed[0] == 100,  res.consumed[0]    # 违约点
    assert res.consumed[1] == 1000, res.consumed[1]    # 可行但没凑够 → 跑满
    assert res.consumed[5] == 400,  res.consumed[5]    # POMO@10 砍
    assert res.consumed[6] == 400,  res.consumed[6]
    assert res.consumed[7] == 400,  res.consumed[7]
    assert res.consumed[4] == 600,  res.consumed[4]    # POMO@15 砍
    assert res.consumed[2] == 1000, res.consumed[2]    # 活到终点
    assert res.consumed[3] == 1000, res.consumed[3]

    expected_C = 100 + 1000 + 400 * 3 + 600 + 1000 * 2
    assert res.total_tokens == expected_C, (res.total_tokens, expected_C)
    assert res.n_survivors == 2, res.n_survivors
    assert sorted(res.survivors) == [2, 3], res.survivors
    # 终点最优 = min(3.5, 3.2) = 3.2  (注: 真正最优 traj4=3.0 被 POMO 在 15% 误杀 → regret)
    assert abs(res.best_distance - 3.2) < 1e-9, res.best_distance

    # ── baseline 曲线 + 同算力读数 ──
    curve, (base_total, base_best) = baseline_anytime_curve(profiles)
    assert base_total == 8000, base_total
    assert abs(base_best - 3.0) < 1e-9, base_best     # 全跑完含 traj4=3.0
    at_C = baseline_dist_at_budget(curve, res.total_tokens)  # budget=4900
    assert abs(at_C - 3.2) < 1e-9, at_C               # ≤4900 只到 traj3, best=3.2

    print("test_simulate_wave PASS  "
          f"(wave_C={res.total_tokens}, wave_best={res.best_distance}, "
          f"baseline_C={base_total}, baseline_best={base_best}, base@C={at_C})")


def test_keep_fraction_one_no_pruning():
    """keep=1.0 + 全可行 → 不应淘汰任何 (除硬过滤), C 应等于 baseline."""
    profiles = [TrajProfile(idx=i, full_tokens=1000, n_feasible_customers=20,
                            violated=False, violation_tokens=1000,
                            tokens_at={5: 200, 10: 400, 15: 600},
                            pomo_at={10: -i, 15: -i},
                            final_feasible=True, final_distance=5.0 - i * 0.1)
                for i in range(4)]
    cfg = WaveConfig(keep_fraction=1.0)
    res = simulate_wave(profiles, n=20, cfg=cfg)
    assert res.total_tokens == 4000, res.total_tokens   # 全跑满
    assert res.n_survivors == 4, res.n_survivors
    print("test_keep_fraction_one_no_pruning PASS")


def test_all_infeasible():
    """全部早期违约 → 全在 25% 删, best=None."""
    profiles = [TrajProfile(idx=i, full_tokens=1000, n_feasible_customers=2,
                            violated=True, violation_tokens=80,
                            tokens_at={}, pomo_at={},
                            final_feasible=False, final_distance=None)
                for i in range(5)]
    res = simulate_wave(profiles, n=20, cfg=WaveConfig())
    assert res.total_tokens == 80 * 5, res.total_tokens
    assert res.best_distance is None
    assert res.n_survivors == 0
    print("test_all_infeasible PASS")


if __name__ == "__main__":
    test_simulate_wave()
    test_keep_fraction_one_no_pruning()
    test_all_infeasible()
    print("\nALL TESTS OK")
