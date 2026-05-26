"""
波次式 (successive-halving) best-of-N 的【离线回放】模拟器。

目的
----
在固定算力 (= 解码 token 总数) 下,验证"用 POMO PRM 做剪枝的 best-of-N"
能否优于朴素 best-of-N。这一版是【离线回放】:照常一次性生成 M 条完整链,
然后事后按 1/4 客户检查点回放波次式淘汰,并精确累计每条链"实际会消耗"的
token,得到 (算力 C, 质量) 点,与朴素 best-of-N 在同一张图上对比。

为什么离线回放对波次式是【忠实】的
--------------------------------------
波次式 = 开一大池 M 条 → 逐检查点漏斗式淘汰,本身不需要"重分配新样本"
(那是流式 Flavor 2)。所以从一池完整链里回放淘汰、只累计每条走到被杀那一刻
的 token,得到的 (C, 质量) 就是 100% 真实的波次式结果,无任何近似。
唯一证明不了的是真实 wall-clock 加速 (仍把全部链生成完了才回放) —— 那属于
在线分段那一版的"系统层"主张,与这里的"算法层"主张分开证。

方案 A 的检查点调度 (见与用户的设计讨论)
------------------------------------------
检查点用【比例】,跨 n 自动成立: target_k = round(n * frac_k)。
  - 25% (选完 round(n/4) 个客户): 只【硬过滤】(违约/重复/越界/漏 → 删),
    不做 POMO 排名 —— POMO 前期信号弱 (项目已知 gap),这么早按值砍人会误杀。
  - 50% / 75%: 硬过滤 + POMO 排名【留一半】(keep_fraction)。
  - 100%: 终点选择 —— 漏节点/不可行的删除,存活者按【真实距离】挑最优。

Token 记账模型 (诚实、偏保守,不夸大节省)
------------------------------------------
每条链按"消耗到离开池子那一刻"的 token 计:
  - 硬违约 (容量/重复/越界): 违约 token 是廉价确定可检测的 (连续约束跟踪器),
    所以在【违约点】就杀,cost = 违约 token 数。剪枝的节省主要来自这里和 POMO。
  - 可行但没凑够 target 个客户 (提前 </think> / 跑飞但没违约): 它会一直生成到
    自然 EOS 才被检查点发现没达标,cost = 完整 token 数 (无节省,诚实)。
  - 被 POMO 在某检查点淘汰: 它生成到该检查点才被砍,cost = 检查点处 token 数。
  - 活到终点: cost = 完整 token 数。
总算力 C = Σ 各链 cost。朴素 best-of-N 的 C_baseline = Σ 完整 token 数。

依赖
----
- pomo_prm.POMOPRM (已实例化, 带 _validate_prefix / _batch_evaluate_prefixes)
- pomo_prm.parse_think_segments
- problems 的 prob 对象 (get_tour_distance / is_feasible)
- tokenizer (算各检查点前缀 token 数)

simulate_wave 是【纯函数】, 只吃 TrajProfile, 不碰 POMO/GPU, 便于单元测试。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# 注: parse_think_segments 在 build_profiles 内惰性 import (它连带 import torch),
# 这样 simulate_wave 等纯逻辑可在无 torch/GPU 的环境单独跑单元测试.


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class WaveConfig:
    """波次式回放的超参数 (全部可扫)."""
    checkpoint_fracs: tuple = (0.25, 0.5, 0.75, 1.0)   # 检查点比例
    halve_fracs: tuple = (0.5, 0.75)                   # 哪些检查点做 POMO 排名淘汰
    keep_fraction: float = 0.5                         # 每轮淘汰保留比例


@dataclass
class TrajProfile:
    """单条 completion 在各检查点的画像 (simulate_wave 的唯一输入)."""
    idx: int
    full_tokens: int                       # 完整链 token 数
    n_feasible_customers: int              # 合法前缀内 distinct 可行客户数
    violated: bool                         # 是否发生硬违约 (vs 仅"没凑够")
    violation_tokens: int                  # 违约检测点的 token 数 (violated=True 时有意义)
    tokens_at: dict                        # target_count → 到该检查点的 token 数 (仅 reached)
    pomo_at: dict                          # target_count → POMO 值 (越大越好, 仅 reached)
    final_feasible: bool                   # 答案段解析成功 + 可行 + 覆盖全 n
    final_distance: Optional[float]        # 真实 tour 距离 (final_feasible 时)


@dataclass
class WaveResult:
    total_tokens: int                      # 波次式总算力 C
    best_distance: Optional[float]         # 终点存活者中真实最优距离
    n_start: int
    n_survivors: int                       # 活到终点的链数
    consumed: dict = field(default_factory=dict)   # idx → 实际消耗 token
    survivors: list = field(default_factory=list)  # 活到终点的 idx


# ══════════════════════════════════════════════════════════════════════
# 纯逻辑: 波次式淘汰 + token 记账  (可单元测试, 不依赖 POMO/GPU)
# ══════════════════════════════════════════════════════════════════════

def simulate_wave(profiles: list[TrajProfile], n: int, cfg: WaveConfig) -> WaveResult:
    """对一组 TrajProfile 跑波次式淘汰, 返回总算力 C + 终点最优距离.

    检查点 target = round(n * frac); target < n 的是中途剪枝点, target == n 是终点选择.
    记账模型见模块 docstring.
    """
    pmap = {p.idx: p for p in profiles}
    checkpoints = sorted({max(1, round(n * f)) for f in cfg.checkpoint_fracs})
    halve_targets = {max(1, round(n * f)) for f in cfg.halve_fracs}

    consumed: dict = {}
    alive = [p.idx for p in profiles]

    for target in checkpoints:
        terminal = (target >= n)
        if terminal:
            break   # 终点单独处理 (下方)

        # ── 硬过滤: 必须可行地凑够 target 个客户 ──
        passed = []
        for i in alive:
            p = pmap[i]
            if p.n_feasible_customers >= target:
                passed.append(i)               # 暂活, token 先不结算 (可能后面被 POMO 砍)
            else:
                # 没凑够 target → 删
                if p.violated:
                    consumed[i] = p.violation_tokens   # 违约点检测, 有节省
                else:
                    consumed[i] = p.full_tokens         # 可行但提前结束, 跑到 EOS, 无节省

        # ── POMO 排名淘汰 (仅 halve 检查点) ──
        if target in halve_targets and len(passed) > 1:
            # 按 POMO 值降序 (越大越好), 留 top keep_fraction
            passed.sort(key=lambda i: pmap[i].pomo_at.get(target, float("-inf")),
                        reverse=True)
            n_keep = max(1, math.ceil(len(passed) * cfg.keep_fraction))
            for i in passed[n_keep:]:
                consumed[i] = pmap[i].tokens_at[target]   # 生成到检查点被砍
            passed = passed[:n_keep]

        alive = passed

    # ── 终点: 活到这里的都跑满, 按真实距离挑最优 (含漏节点过滤) ──
    for i in alive:
        consumed[i] = pmap[i].full_tokens
    feas = [i for i in alive
            if pmap[i].final_feasible and pmap[i].final_distance is not None]
    best = min((pmap[i].final_distance for i in feas), default=None)

    return WaveResult(
        total_tokens=sum(consumed.values()),
        best_distance=best,
        n_start=len(profiles),
        n_survivors=len(alive),
        consumed=consumed,
        survivors=alive,
    )


def baseline_anytime_curve(profiles: list[TrajProfile]):
    """朴素 best-of-N 的 anytime 曲线: 按生成顺序累计 token, 记 best-so-far 真实距离.

    Returns: list[(cumulative_tokens, best_dist_so_far_or_None)], 以及 (C_total, best_total).
    """
    curve = []
    cum = 0
    best = None
    for p in profiles:
        cum += p.full_tokens
        if p.final_feasible and p.final_distance is not None:
            if best is None or p.final_distance < best:
                best = p.final_distance
        curve.append((cum, best))
    return curve, (cum, best)


def baseline_dist_at_budget(curve, budget: int) -> Optional[float]:
    """朴素 best-of-N 在给定 token 预算下能达到的最优距离 (读 anytime 曲线).

    取累计 token ≤ budget 的最后一个点的 best-so-far. 用于"同算力下谁更好"的对比.
    """
    best = None
    for cum, b in curve:
        if cum <= budget:
            best = b
        else:
            break
    return best


# ══════════════════════════════════════════════════════════════════════
# 画像构建: 从 completion + POMO PRM 提取 TrajProfile  (依赖 POMO/GPU)
# ══════════════════════════════════════════════════════════════════════

def build_profiles(
    completions: list[str],
    instance: dict,
    prob,
    prm,                       # POMOPRM 实例
    tokenizer,
    problem_type: str,
    cfg: WaveConfig,
) -> list[TrajProfile]:
    """把单个 instance 的 M 条 completion 转成 TrajProfile 列表."""
    from pomo_prm import parse_think_segments   # 惰性 import (连带 torch)

    n = instance["n"]
    targets = sorted({max(1, round(n * f)) for f in cfg.checkpoint_fracs if f < 1.0})

    def _tok_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False).input_ids)

    profiles: list[TrajProfile] = []
    for idx, completion in enumerate(completions):
        full_tokens = _tok_len(completion)

        full_steps, customer_steps, anomaly_start = parse_think_segments(completion)
        valid_len = prm._validate_prefix(full_steps, instance, problem_type)
        # 合法前缀内的客户步 (full_step_idx < valid_len → 该步被约束校验接受)
        valid_cust = [s for s in customer_steps if s.full_step_idx < valid_len]
        n_feas = len(valid_cust)

        # 硬违约判定: think 链里有 step 被 _validate_prefix 截掉, 或检测到重复访问
        violated = (valid_len < len(full_steps)) or (anomaly_start is not None)
        # 违约检测点: 第一个非法客户步的字符末尾 (读完它才知道非法); 没有则取合法前缀末尾
        if violated:
            bad = next((s for s in customer_steps if s.full_step_idx >= valid_len), None)
            if bad is None and anomaly_start is not None and anomaly_start < len(customer_steps):
                bad = customer_steps[anomaly_start]
            if bad is not None:
                violation_tokens = _tok_len(completion[:bad.char_range[1]])
            elif valid_cust:
                violation_tokens = _tok_len(completion[:valid_cust[-1].char_range[1]])
            else:
                violation_tokens = full_tokens
        else:
            violation_tokens = full_tokens

        # 各 reached 检查点的 token 数 + POMO 值 (一次 batched rollout 拿全部)
        tokens_at: dict = {}
        pomo_at: dict = {}
        reached = [t for t in targets if t <= n_feas]
        if reached:
            for t in reached:
                tokens_at[t] = _tok_len(completion[:valid_cust[t - 1].char_range[1]])
            cust_indices = [valid_cust[t - 1].full_step_idx for t in reached]
            pomo_vals = prm._batch_evaluate_prefixes(
                full_steps[:valid_len], cust_indices, instance, problem_type,
            )
            for t, v in zip(reached, pomo_vals):
                pomo_at[t] = v

        # 终点真实指标 (答案段)
        dist = prob.get_tour_distance(completion, instance)
        final_feasible = bool(prob.is_feasible(completion, instance)) and dist is not None

        profiles.append(TrajProfile(
            idx=idx,
            full_tokens=full_tokens,
            n_feasible_customers=n_feas,
            violated=violated,
            violation_tokens=violation_tokens,
            tokens_at=tokens_at,
            pomo_at=pomo_at,
            final_feasible=final_feasible,
            final_distance=dist if final_feasible else None,
        ))
    return profiles


def wave_replay(
    all_completions: list[list[str]],   # [num_instances][M]
    instances: list[dict],
    prob,
    prm,
    tokenizer,
    problem_type: str,
    cfg: Optional[WaveConfig] = None,
):
    """对所有 instance 跑波次式回放, 聚合指标 + 与 baseline 同算力对比.

    Returns dict:
      wave_C_total, wave_avg_best_dist,
      baseline_C_total, baseline_avg_best_dist,
      baseline_avg_best_dist_at_wave_C,   # 朴素 best-of-N 在波次式同等算力下的成绩
      compute_saving_ratio,               # 1 - wave_C/baseline_C
      per_instance: [...]                 # 每实例明细
    """
    cfg = cfg or WaveConfig()

    per_instance = []
    wave_C = baseline_C = 0
    wave_dists, base_dists, base_at_C_dists = [], [], []

    for inst, comps in zip(instances, all_completions):
        n = inst["n"]
        profiles = build_profiles(comps, inst, prob, prm, tokenizer, problem_type, cfg)
        wres = simulate_wave(profiles, n, cfg)
        curve, (base_total, base_best) = baseline_anytime_curve(profiles)
        base_at_C = baseline_dist_at_budget(curve, wres.total_tokens)

        wave_C += wres.total_tokens
        baseline_C += base_total
        if wres.best_distance is not None:
            wave_dists.append(wres.best_distance)
        if base_best is not None:
            base_dists.append(base_best)
        if base_at_C is not None:
            base_at_C_dists.append(base_at_C)

        per_instance.append({
            "n": n,
            "wave_C": wres.total_tokens,
            "wave_best": wres.best_distance,
            "wave_survivors": wres.n_survivors,
            "baseline_C": base_total,
            "baseline_best": base_best,
            "baseline_best_at_wave_C": base_at_C,
        })

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "config": cfg,
        "n_instances": len(instances),
        "wave_C_total": wave_C,
        "baseline_C_total": baseline_C,
        "compute_saving_ratio": (1 - wave_C / baseline_C) if baseline_C else None,
        "wave_avg_best_dist": _mean(wave_dists),
        "baseline_avg_best_dist": _mean(base_dists),
        "baseline_avg_best_dist_at_wave_C": _mean(base_at_C_dists),
        "per_instance": per_instance,
    }


__all__ = [
    "WaveConfig", "TrajProfile", "WaveResult",
    "simulate_wave", "baseline_anytime_curve", "baseline_dist_at_budget",
    "build_profiles", "wave_replay",
]
