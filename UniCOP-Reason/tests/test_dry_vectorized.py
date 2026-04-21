"""验证 numpy 向量化版本 vs 原始纯 Python 版本数值一致，并测速。"""

from __future__ import annotations

import time

import numpy as np


# ── 参考实现（纯 Python，已在 test_dry_algorithm_pure.py 通过） ─────────────
def compute_penalties_ref(context, allowed_length, multiplier, base, max_match):
    n = len(context)
    if n < 2:
        return {}
    m = [0] * n
    for p in range(n):
        L = 0
        while (
            L < max_match
            and p - L >= 0
            and n - 1 - L >= 0
            and context[p - L] == context[n - 1 - L]
        ):
            L += 1
        m[p] = L
    best = {}
    for p in range(n - 1):
        tok = context[p]
        L_ext = m[p - 1] + 1 if p >= 1 else 1
        if tok not in best or L_ext > best[tok]:
            best[tok] = L_ext
    out = {}
    for tok, L in best.items():
        if L > allowed_length:
            out[tok] = multiplier * (base ** (L - allowed_length))
    return out


# ── 向量化实现（从 dry_processor 复制数值逻辑） ───────────────────────────
def compute_penalties_vec(context, allowed_length, multiplier, base, max_match):
    n = len(context)
    if n < 2:
        return {}
    effective_max = min(max_match, n)
    ctx = np.asarray(context, dtype=np.int64)

    A = np.zeros((n, effective_max), dtype=np.int8)
    for L in range(effective_max):
        right_val = ctx[n - 1 - L]
        A[L:, L] = (ctx[: n - L] == right_val).astype(np.int8)

    cum = np.cumprod(A, axis=1)
    m_arr = cum.sum(axis=1)

    L_ext = np.empty(n, dtype=np.int64)
    L_ext[0] = 1
    L_ext[1:] = m_arr[:-1] + 1
    tokens = ctx[: n - 1]
    L_ext_valid = L_ext[: n - 1]

    order = np.lexsort((-L_ext_valid, tokens))
    sorted_tokens = tokens[order]
    sorted_L = L_ext_valid[order]
    first_of_group = np.concatenate(([True], sorted_tokens[1:] != sorted_tokens[:-1]))
    best_tokens = sorted_tokens[first_of_group]
    best_L = sorted_L[first_of_group]

    mask = best_L > allowed_length
    pen_tokens = best_tokens[mask]
    pen_L = best_L[mask]
    if pen_tokens.size == 0:
        return {}
    pen_values = multiplier * np.power(base, (pen_L - allowed_length).astype(np.float64))
    return dict(zip(pen_tokens.tolist(), pen_values.tolist()))


# ── 数值一致性 ───────────────────────────────────────────────
def dicts_close(a, b, eps=1e-9):
    if set(a.keys()) != set(b.keys()):
        return False
    return all(abs(a[k] - b[k]) < eps for k in a)


rng = np.random.default_rng(42)
test_cases = [
    # (name, context, allowed, mult, base, max_match)
    ("long_repeat", [1, 2, 3, 4, 5, 1, 2, 3, 4], 2, 0.8, 1.75, 20),
    ("short_ok",    [1, 2, 3, 4, 1, 2], 4, 0.8, 1.75, 20),
    ("exp_scale",   [1, 2, 3, 4, 1, 2, 3], 2, 1.0, 2.0, 20),
    ("cyclic",      [42, 99] * 10, 4, 0.8, 1.75, 20),
    ("max_match_bound", list(range(50)) + list(range(49)), 2, 1.0, 2.0, 5),
    ("tiny",        [7, 7], 1, 0.8, 1.75, 20),
    ("empty",       [], 2, 0.8, 1.75, 20),
    ("single",      [5], 2, 0.8, 1.75, 20),
    ("random_500",  rng.integers(0, 50, size=500).tolist(), 4, 0.8, 1.75, 20),
    ("random_2000", rng.integers(0, 100, size=2000).tolist(), 4, 0.8, 1.75, 20),
]

print("一致性测试")
print("-" * 70)
for name, ctx, al, m, b, mm in test_cases:
    ref = compute_penalties_ref(ctx, al, m, b, mm)
    vec = compute_penalties_vec(ctx, al, m, b, mm)
    ok = dicts_close(ref, vec)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<20} ref={len(ref)} toks, vec={len(vec)} toks")
    if not ok:
        diff = {k: (ref.get(k), vec.get(k)) for k in set(ref) ^ set(vec)}
        print(f"    差异: {diff}")
        for k in set(ref) & set(vec):
            if abs(ref[k] - vec[k]) > 1e-9:
                print(f"    token {k}: ref={ref[k]} vec={vec[k]}")
        raise AssertionError(f"{name} 数值不一致")

print()
print("性能测试（N=5000，循环多次取平均）")
print("-" * 70)
ctx_big = rng.integers(0, 150, size=5000).tolist()

N_RUNS = 5
t0 = time.perf_counter()
for _ in range(N_RUNS):
    compute_penalties_ref(ctx_big, 4, 0.8, 1.75, 20)
t_ref = (time.perf_counter() - t0) / N_RUNS

t0 = time.perf_counter()
for _ in range(N_RUNS):
    compute_penalties_vec(ctx_big, 4, 0.8, 1.75, 20)
t_vec = (time.perf_counter() - t0) / N_RUNS

print(f"  纯 Python:   {t_ref * 1000:.2f} ms/call")
print(f"  向量化:      {t_vec * 1000:.2f} ms/call")
print(f"  加速比:      {t_ref / t_vec:.1f}x")

print()
print("性能测试（N=10000）")
ctx_huge = rng.integers(0, 150, size=10000).tolist()
t0 = time.perf_counter()
for _ in range(N_RUNS):
    compute_penalties_ref(ctx_huge, 4, 0.8, 1.75, 20)
t_ref = (time.perf_counter() - t0) / N_RUNS
t0 = time.perf_counter()
for _ in range(N_RUNS):
    compute_penalties_vec(ctx_huge, 4, 0.8, 1.75, 20)
t_vec = (time.perf_counter() - t0) / N_RUNS
print(f"  纯 Python:   {t_ref * 1000:.2f} ms/call")
print(f"  向量化:      {t_vec * 1000:.2f} ms/call")
print(f"  加速比:      {t_ref / t_vec:.1f}x")

# ── 病理场景：长循环已形成，每个 p 都能扩展到 max_match ─────────────
print()
print("病理场景（退化循环已形成，N=5000）")
print("-" * 70)
# 前 4000 是 50-token pattern 循环 80 次；后 1000 继续循环一部分
pattern = list(range(50))
bad_ctx = (pattern * 100)[:5000]

t0 = time.perf_counter()
for _ in range(N_RUNS):
    compute_penalties_ref(bad_ctx, 4, 0.8, 1.75, 20)
t_ref = (time.perf_counter() - t0) / N_RUNS
t0 = time.perf_counter()
for _ in range(N_RUNS):
    compute_penalties_vec(bad_ctx, 4, 0.8, 1.75, 20)
t_vec = (time.perf_counter() - t0) / N_RUNS

r = compute_penalties_ref(bad_ctx, 4, 0.8, 1.75, 20)
v = compute_penalties_vec(bad_ctx, 4, 0.8, 1.75, 20)
assert dicts_close(r, v)
max_pen = max(r.values()) if r else 0
print(f"  (最大惩罚 = {max_pen:.1f}, penalized tokens = {len(r)})")
print(f"  纯 Python:   {t_ref * 1000:.2f} ms/call")
print(f"  向量化:      {t_vec * 1000:.2f} ms/call")
print(f"  加速比:      {t_ref / t_vec:.1f}x")

print("\n所有测试通过。")
