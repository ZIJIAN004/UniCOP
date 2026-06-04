#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_shards.py — 合并 evaluate.py --num_shards 的分片输出, 精确重算聚合指标。

核心指标(可行率/avg_best_dist/各 rate)从各分片的 _raw 计数 + per_instance_best 精确合并;
bestofn scaling 曲线按 n_eval 加权合并(best-effort)。输出与全量单跑同格式, 供下游 gap 消费。

用法:
    python merge_shards.py eval_dir/Model_*_shard0of4.json ... --out eval_dir/Model_merged.json
或:
    python merge_shards.py --glob "eval_dir/Model_*_shard*of4.json" --out merged.json
"""
import argparse
import glob as _glob
import json
from collections import defaultdict

import numpy as np


def _merge_bestofn(combos):
    n_evals = [c["_raw"]["n_eval"] for c in combos]
    tot_ne = sum(n_evals)
    # 每分片曲线点 avg_best_dist@k 的真实分母 = 该分片"有≥1可行样本的实例数"
    # (bestofn_eval.expected_best_of_k 对 m≥1 的实例在所有 k 都返回非 None, 与 k 无关)
    # = _raw.instance_has_feas → 用它加权可【精确】还原全量单跑的池化均值。
    # ⚠️ 旧实现用 feas_rate*n_eval = Σ p_feas (概率和): k<N 时 p_feas<1 的部分可行实例
    #    权重被低估 → 合并曲线偏向高可行分片, 与单跑系统性不一致 (workflow 审查确认)。
    w_feas = [c["_raw"]["instance_has_feas"] for c in combos]
    curves = [c["bestofn"]["scaling_curve"] for c in combos]
    out = []
    for pts in zip(*curves):                      # 同一 k 的各分片点
        compute = sum(p["compute"] for p in pts)
        vals = [(p["avg_best_dist"], w) for p, w in zip(pts, w_feas)
                if p["avg_best_dist"] is not None and w > 0]
        abd = (sum(v * w for v, w in vals) / sum(w for _, w in vals)) if vals else None
        fr = sum(p["feas_rate"] * ne for p, ne in zip(pts, n_evals))   # P(≥1可行) 按实例数池化
        out.append({"k": pts[0]["k"], "compute": compute,
                    "avg_best_dist": round(abd, 4) if abd is not None else None,
                    "feas_rate": round(fr / tot_ne, 4) if tot_ne else 0.0})
    base = dict(combos[0]["bestofn"])
    base["scaling_curve"] = out
    # 这些汇总量不能抄 shard0: 重算
    base["n_instances"] = sum(c["bestofn"]["n_instances"] for c in combos)
    base["total_tokens"] = sum(c["bestofn"]["total_tokens"] for c in combos)
    _N = base.get("N") or 1
    _n_samp = base["n_instances"] * _N
    base["mean_tokens_per_sample"] = round(base["total_tokens"] / _n_samp, 1) if _n_samp else 0.0
    return base


def _merge_wave(combos):
    """精确合并 wave 指标: per_instance 拼接, C 求和, 均值在非 None 的实例上重算。
    (与 wave_replay.py:318-330 的聚合口径一致 — 无可行解的实例不参与平均。)"""
    waves = [c["wave"] for c in combos]
    per = [pi for w in waves for pi in w["per_instance"]]
    wave_C = sum(w["wave_C_total"] for w in waves)
    base_C = sum(w["baseline_C_total"] for w in waves)

    def _m(key):
        xs = [p[key] for p in per if p.get(key) is not None]
        return round(float(np.mean(xs)), 4) if xs else None

    # 对齐口径: 与 wave_replay.py 一致, 在两边都有值的实例交集上算
    pairs = [(p["wave_best"], p["baseline_best_at_wave_C"]) for p in per
             if p.get("wave_best") is not None and p.get("baseline_best_at_wave_C") is not None]

    def _pm(i):
        return round(float(np.mean([t[i] for t in pairs])), 4) if pairs else None

    out = dict(waves[0])                       # 配置字段(checkpoint_fracs 等)取第一个分片
    out.update({
        "n_instances": sum(w["n_instances"] for w in waves),
        "wave_C_total": wave_C,
        "baseline_C_total": base_C,
        "compute_saving_ratio": round(1 - wave_C / base_C, 4) if base_C else None,
        "wave_avg_best_dist": _m("wave_best"),
        "baseline_avg_best_dist": _m("baseline_best"),
        "baseline_avg_best_dist_at_wave_C": _m("baseline_best_at_wave_C"),
        "n_aligned": len(pairs),
        "wave_avg_best_dist_aligned": _pm(0),
        "baseline_avg_best_dist_at_wave_C_aligned": _pm(1),
        "per_instance": per,
    })
    return out


def merge_combo(combos):
    """合并同一 (problem,size) 的各分片 result dict → 全量等价 result dict。"""
    raws = [c["_raw"] for c in combos]
    ts = sum(r["total_samples"] for r in raws)
    n_eval = sum(r["n_eval"] for r in raws)
    feas_inst = sum(r["instance_has_feas"] for r in raws)
    best_dists = [d for r in raws for d in r["best_dists"]]
    per_inst = sorted((tuple(x) for c in combos for x in c["per_instance_best"]),
                      key=lambda t: t[0])

    def rate(key):
        return sum(r[key] for r in raws) / ts if ts else 0.0

    merged = {k: v for k, v in combos[0].items()
              if k not in ("shard", "_raw", "per_instance_best", "bestofn", "wave")}
    merged.update({
        "num_test": n_eval, "n_eval": n_eval, "total_samples": ts,
        "truncation_rate": round(rate("total_truncated"), 4),
        "format_match_rate": round(rate("total_parsed"), 4),
        "global_feasibility_rate": round(rate("total_feasible"), 4),
        "coverage_rate": round(sum(r["sum_coverage"] for r in raws) / ts, 4) if ts else 0.0,
        "constraint_rate": round(sum(r["sum_constraint"] for r in raws) / ts, 4) if ts else 0.0,
        "avg_completion_tokens": round(sum(r["sum_comp_len"] for r in raws) / ts, 1) if ts else 0.0,
        # min/max 不能抄 shard0 (最长/最短 completion 可能在别的分片): 各分片 min/max 再聚合 = 精确
        "min_completion_tokens": min(c["min_completion_tokens"] for c in combos),
        "max_completion_tokens": max(c["max_completion_tokens"] for c in combos),
        "instance_feasibility_rate": round(feas_inst / n_eval, 4) if n_eval else 0.0,
        "feasible_instances": feas_inst,
        "avg_best_dist": round(float(np.mean(best_dists)), 4) if best_dists else None,
        "per_instance_best": [[gi, d] for gi, d in per_inst],
        "num_shards_merged": len(combos),
    })
    # 完整性检查: 全局下标应无缺漏/重复
    idxs = [gi for gi, _ in per_inst]
    if len(idxs) != len(set(idxs)):
        merged["_warn"] = "per_instance 全局下标有重复!"
    try:
        if all("bestofn" in c for c in combos):
            merged["bestofn"] = _merge_bestofn(combos)
    except Exception as e:
        merged["bestofn_merge_error"] = repr(e)
    # wave 指标精确合并 (之前直接抄第一个分片的, 分片跑 --wave 时合并结果只含 1/N 实例 → bug)
    try:
        if all("wave" in c for c in combos):
            merged["wave"] = _merge_wave(combos)
    except Exception as e:
        merged["wave_merge_error"] = repr(e)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="*", help="分片 JSON 文件路径")
    ap.add_argument("--glob", default=None, help="或用通配符指定分片")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = list(a.shards)
    if a.glob:
        paths += sorted(_glob.glob(a.glob))
    if not paths:
        raise SystemExit("没有分片文件 (传路径或 --glob)")
    print(f"合并 {len(paths)} 个分片: {paths}")
    files = [json.load(open(p, encoding="utf-8")) for p in paths]

    groups = defaultdict(list)
    for fdata in files:
        for c in fdata["results"]:
            groups[(c["problem_type"], c["problem_size"])].append(c)

    merged_results = []
    for cs in groups.values():
        ns = {c.get("shard", {}).get("num_shards") for c in cs}
        ids = sorted(c.get("shard", {}).get("shard_id") for c in cs)
        if len(cs) != (ns.pop() if len(ns) == 1 else -1):
            print(f"⚠️ {cs[0]['problem_type']} n={cs[0]['problem_size']}: 分片数={len(cs)}, "
                  f"shard_id={ids} — 可能缺分片!")
        merged_results.append(merge_combo(cs))

    out = {"hyperparams": files[0]["hyperparams"], "results": merged_results,
           "merged_from": paths}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("-" * 60)
    for r in merged_results:
        print(f"  {r['problem_type']} n={r['problem_size']}: "
              f"instance_feas={r['instance_feasibility_rate']:.2%}  "
              f"avg_best_dist={r['avg_best_dist']}  "
              f"({r['n_eval']} 实例, 合并 {r['num_shards_merged']} 分片)")
    print(f"已写: {a.out}")


if __name__ == "__main__":
    main()
