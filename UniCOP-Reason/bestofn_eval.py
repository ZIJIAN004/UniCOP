"""
朴素 best-of-N 的 scaling 曲线 (推理期对照基线, 不依赖 POMO / torch).

作用
----
波次式回放 (wave_replay.py) 要打赢的就是这条曲线: 朴素 best-of-N 在固定算力
(= 解码 token 总数) 下能达到的质量. 现在 evaluate.py 只给固定 N 的单点
(avg_best_dist), 这里给【完整曲线】: 质量随 k (= 采样次数) / 算力变化.

只需 completion 文本 + tokenizer + prob (算距离/可行性), 不碰 POMO, 所以能在
POMO 还没接通时先单独跑, 也方便和 wave 的点叠在同一张 (算力, 质量) 图上对比.

best-of-k 期望: 顺序统计闭式
------------------------------
给定一个实例的 N 条样本 (m 条可行, 距离升序 d_(1)≤…≤d_(m); 其余不可行视为 +∞),
随机抽 k 条 (不放回) 的最优 (min) 距离期望:
  P(全 k 条都不可行) = C(N-m, k) / C(N, k)
  E[best-of-k | ≥1 可行]
     = [ Σ_{r=1}^{m} d_(r) · C(N-r, k-1) / C(N, k) ] / P(≥1 可行)
  (P(rank-r 样本是子集最小) = C(N-r, k-1)/C(N,k); 可行样本恒排在不可行前面)
闭式 → 无需 Monte Carlo, k 全程精确.

token 口径与 wave_replay 一致: tokenizer 对【完整 completion】计数. k=N 时总算力
等于 wave_replay 的 baseline_C_total, 两套可直接对齐.

bestofn_scaling_curve / expected_best_of_k 是纯函数, 可单元测试 (无 torch/GPU).
"""
from __future__ import annotations

import math
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# 纯逻辑 (可单元测试)
# ══════════════════════════════════════════════════════════════════════

def expected_best_of_k(sorted_feas: list[float], n_total: int, k: int):
    """单实例 best-of-k 期望.

    Args:
        sorted_feas: 该实例【可行】样本的距离, 升序.
        n_total:     该实例总样本数 (含不可行).
        k:           抽样次数 (1..n_total).
    Returns:
        (E[best-of-k | ≥1可行] or None, P(≥1可行)).
        None = 该 k 下抽不到任何可行 (或 k>n_total).
    """
    if k < 1 or k > n_total:
        return None, 0.0
    c_tot = math.comb(n_total, k)
    if c_tot == 0:
        return None, 0.0
    m = len(sorted_feas)
    # P(全不可行) = C(n_total-m, k)/C(n_total,k)
    c_none = math.comb(n_total - m, k) if (n_total - m) >= k else 0
    p_feas = 1.0 - c_none / c_tot
    if p_feas <= 0 or m == 0:
        return None, max(0.0, p_feas)
    num = 0.0
    for r in range(1, m + 1):                  # rank r = 第 r 小可行 (整体也排第 r)
        num += sorted_feas[r - 1] * math.comb(n_total - r, k - 1)
    e_best = (num / c_tot) / p_feas            # conditional on ≥1 feasible
    return e_best, p_feas


def bestofn_scaling_curve(per_instance, mean_tokens: float,
                          n_instances: int, N: int):
    """聚合所有实例 → best-of-k scaling 曲线.

    Args:
        per_instance: list of (sorted_feas, n_total) 每实例.
        mean_tokens:  全局每样本平均 token 数 (算力 = k · mean_tokens · n_instances).
        n_instances:  实例数.
        N:            最大 k (= num_samples).
    Returns:
        list[dict]: 每个 k 一行 {k, compute, avg_best_dist, feas_rate}.
                    avg_best_dist 在"有可行样本的实例"上平均 (与 evaluate.py 口径一致).
    """
    curve = []
    for k in range(1, N + 1):
        dists, feas_ps = [], []
        for sorted_feas, n_total in per_instance:
            if k > n_total:
                continue
            e, pf = expected_best_of_k(sorted_feas, n_total, k)
            feas_ps.append(pf)
            if e is not None:
                dists.append(e)
        curve.append({
            "k": k,
            "compute": k * mean_tokens * n_instances,   # 期望总算力 (token)
            "avg_best_dist": (sum(dists) / len(dists)) if dists else None,
            "feas_rate": (sum(feas_ps) / len(feas_ps)) if feas_ps else 0.0,
        })
    return curve


def dist_at_budget(curve, budget: float) -> Optional[float]:
    """best-of-N 曲线在给定算力预算下的最优距离 (取 compute≤budget 的最后一点)."""
    best = None
    for pt in curve:
        if pt["compute"] <= budget and pt["avg_best_dist"] is not None:
            best = pt["avg_best_dist"]
        elif pt["compute"] > budget:
            break
    return best


# ══════════════════════════════════════════════════════════════════════
# 从 completion 构建 (依赖 tokenizer + prob, 不依赖 POMO)
# ══════════════════════════════════════════════════════════════════════

def bestofn_replay(all_completions, instances, prob, tokenizer):
    """对所有实例算 best-of-N scaling 曲线.

    Args:
        all_completions: [n_instances][N] 的 completion 文本.
        instances:       [n_instances] instance dict.
        prob:            problem 对象 (get_tour_distance / is_feasible).
        tokenizer:       算每条完整 completion 的 token 数.
    Returns dict: mean_tokens_per_sample, N, n_instances, total_tokens, scaling_curve.
    """
    per_instance = []
    total_tokens = 0
    n_samp = 0
    for inst, comps in zip(instances, all_completions):
        feas = []
        for c in comps:
            total_tokens += len(tokenizer(c, add_special_tokens=False).input_ids)
            n_samp += 1
            d = prob.get_tour_distance(c, inst)
            if d is not None and prob.is_feasible(c, inst):
                feas.append(d)
        per_instance.append((sorted(feas), len(comps)))

    mean_tokens = (total_tokens / n_samp) if n_samp else 0.0
    N = max((nt for _, nt in per_instance), default=0)
    curve = bestofn_scaling_curve(per_instance, mean_tokens, len(instances), N)
    return {
        "n_instances":           len(instances),
        "N":                     N,
        "mean_tokens_per_sample": round(mean_tokens, 1),
        "total_tokens":          total_tokens,   # = best-of-N 全量算力 (k=N)
        "scaling_curve":         curve,
    }


__all__ = [
    "expected_best_of_k", "bestofn_scaling_curve", "dist_at_budget",
    "bestofn_replay",
]
