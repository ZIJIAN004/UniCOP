"""
optimal/loader.py
读取冻结实例(generate_testset)与基线缓存(build_optimal)，计算 optimality gap。

  load_instances  读 optimal/instances/ 的冻结实例（求解、评测共用同一批）
  load_costs      读 optimal/cache/ 的近最优 cost
  optimality_gap  由模型距离与 cost 算 gap

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
_DEFAULT_INSTANCES = os.path.join(_HERE, "instances")


def _find_by_prefix(prefix: str, directory: str, min_n: int | None) -> str | None:
    """在 directory 找形如 {prefix}{N}.json 的文件；min_n=None 取最大 N，否则取 >=min_n 的最小 N。"""
    cands = []
    if os.path.isdir(directory):
        for fn in os.listdir(directory):
            if fn.startswith(prefix) and fn.endswith(".json"):
                big_n = int(fn[len(prefix):-len(".json")])
                if min_n is None or big_n >= min_n:
                    cands.append((big_n, os.path.join(directory, fn)))
    if not cands:
        return None
    return (max(cands)[1] if min_n is None
            else min(cands, key=lambda x: x[0])[1])


def load_instances(problem_type: str, n: int, seed: int = 9999,
                   num_test: int | None = None,
                   inst_dir: str = _DEFAULT_INSTANCES) -> list[dict]:
    """
    读取冻结实例。num_test=None 返回全部；否则返回前 num_test 个（前缀一致）。
    找不到精确 N 时回退到 N>=请求值的最小文件取前缀；num_test=None 时取最大文件。
    """
    prefix = f"{problem_type}_n{n}_seed{seed}_N"
    path = _find_by_prefix(prefix, inst_dir, min_n=num_test)
    if path is None:
        raise FileNotFoundError(
            f"找不到 {problem_type} n={n} seed={seed} 的冻结实例于 {inst_dir}。"
            f"先运行: python -m optimal.generate_testset "
            f"--problem_types {problem_type} --sizes {n} --seed {seed}"
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    insts = payload["instances"]
    return insts[:num_test] if num_test is not None else insts


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
