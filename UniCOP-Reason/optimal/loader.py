"""
optimal/loader.py
读取 build_optimal.py 生成的基线缓存，计算 optimality gap。

集成到 evaluate.py（示意）：
    from optimal.loader import load_costs, optimality_gap
    opt = load_costs(problem_type, n, num_test, seed=9999)   # 长度 num_test
    # best_dists[i] 为模型在第 i 个实例上的最优可行解距离（evaluate.py 已有）
    gap = optimality_gap(model_dists=best_dists, optimal_costs=opt)
    print(gap["mean_gap_pct"], gap["matched"])
"""

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CACHE = os.path.join(_HERE, "cache")


def cache_path(problem_type: str, n: int, num_test: int, seed: int = 9999,
               cache_dir: str = _DEFAULT_CACHE) -> str:
    return os.path.join(cache_dir, f"{problem_type}_n{n}_seed{seed}_N{num_test}.json")


def _find_cache(problem_type: str, n: int, num_test: int, seed: int,
                cache_dir: str) -> str:
    """
    精确文件不存在时，回退到“更大 N 的同 (type, n, seed) 缓存并取前缀”。
    依赖前缀一致性：实例序列由 seed 顺序决定，前 K 个与小集合一致。
    """
    exact = cache_path(problem_type, n, num_test, seed, cache_dir)
    if os.path.exists(exact):
        return exact

    prefix = f"{problem_type}_n{n}_seed{seed}_N"
    candidates = []
    if os.path.isdir(cache_dir):
        for fn in os.listdir(cache_dir):
            if fn.startswith(prefix) and fn.endswith(".json"):
                big_n = int(fn[len(prefix):-len(".json")])
                if big_n >= num_test:
                    candidates.append((big_n, os.path.join(cache_dir, fn)))
    if not candidates:
        raise FileNotFoundError(
            f"找不到 {problem_type} n={n} seed={seed} 且 N>={num_test} 的缓存于 {cache_dir}。"
            f"先运行: python -m optimal.build_optimal --problem_types {problem_type} "
            f"--sizes {n} --num_test {num_test} --seed {seed}"
        )
    return min(candidates, key=lambda x: x[0])[1]  # 取刚好够大的最小集合


def load_costs(problem_type: str, n: int, num_test: int, seed: int = 9999,
               cache_dir: str = _DEFAULT_CACHE) -> list:
    """返回前 num_test 个实例的近最优 cost 列表（失败处为 None）。"""
    path = _find_cache(problem_type, n, num_test, seed, cache_dir)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["costs"][:num_test]


def optimality_gap(model_dists, optimal_costs) -> dict:
    """
    计算 optimality gap。仅在“模型给出可行解 且 基线求解成功”的实例上统计。

    Args:
        model_dists:   list，模型每个实例的最优可行解距离；不可行/无解处传 None。
        optimal_costs: list，基线每个实例的近最优 cost；失败处为 None。
    Returns dict:
        mean_gap_pct:   平均 gap（百分比），gap_i = model_i / opt_i - 1
        median_gap_pct: 中位数 gap
        sem_pct:        gap 均值的标准误 = std/sqrt(matched)
        matched:        参与统计的实例数
        n_total:        总实例数
    """
    gaps = []
    n_total = max(len(model_dists), len(optimal_costs))
    for md, oc in zip(model_dists, optimal_costs):
        if md is None or oc is None or oc <= 0:
            continue
        gaps.append(md / oc - 1.0)

    if not gaps:
        return {"mean_gap_pct": None, "median_gap_pct": None, "sem_pct": None,
                "matched": 0, "n_total": n_total}

    g = np.asarray(gaps, dtype=float)
    sem = float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
    return {
        "mean_gap_pct": float(np.mean(g) * 100),
        "median_gap_pct": float(np.median(g) * 100),
        "sem_pct": sem * 100,
        "matched": len(g),
        "n_total": n_total,
    }
