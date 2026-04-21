"""纯 Python 验证 DRY 核心算法（不依赖 torch/transformers）。

直接从 dry_processor 复制核心 _compute_penalties 逻辑，独立跑数值断言。
"""


def compute_penalties(context, allowed_length, multiplier, base, max_match):
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

    penalties = {}
    for tok, L in best.items():
        if L > allowed_length:
            penalties[tok] = multiplier * (base ** (L - allowed_length))
    return penalties


def approx(a, b, eps=1e-6):
    return abs(a - b) < eps


# ── 长匹配被惩罚 ──────────────────────────────────────────
# context = [1,2,3,4,5,1,2,3,4]，追加 5 → 匹配 5-gram [1,2,3,4,5]
pen = compute_penalties([1, 2, 3, 4, 5, 1, 2, 3, 4], allowed_length=2, multiplier=0.8, base=1.75, max_match=20)
assert 5 in pen, f"token 5 应该被惩罚: {pen}"
expected = 0.8 * (1.75 ** (5 - 2))
assert approx(pen[5], expected), f"惩罚值错误 {pen[5]} vs {expected}"
print(f"[PASS] long repeat: token 5 -> penalty {pen[5]:.4f} (expected {expected:.4f})")


# ── 短匹配不惩罚 ──────────────────────────────────────────
# context = [1,2,3,4,1,2]，追加 3 → 匹配 3-gram，allowed=4 不惩罚
pen = compute_penalties([1, 2, 3, 4, 1, 2], allowed_length=4, multiplier=0.8, base=1.75, max_match=20)
assert 3 not in pen, f"3-gram < allowed=4 不应惩罚: {pen}"
print(f"[PASS] short match (len 3 < allow 4): no penalty")


# ── 指数增长 ─────────────────────────────────────────────
# 4-gram 匹配: ids=[1,2,3,4,1,2,3], 追加 4; mult=1, base=2, allow=2; penalty=2^(4-2)=4
pen = compute_penalties([1, 2, 3, 4, 1, 2, 3], allowed_length=2, multiplier=1.0, base=2.0, max_match=20)
assert 4 in pen and approx(pen[4], 4.0), f"期望 penalty=4.0, 得 {pen.get(4)}"
print(f"[PASS] exponential scaling: 4-gram -> penalty {pen[4]}")

# 3-gram 匹配: ids=[1,2,3,1,2], 追加 3; penalty = 2^(3-2) = 2
pen = compute_penalties([1, 2, 3, 1, 2], allowed_length=2, multiplier=1.0, base=2.0, max_match=20)
assert 3 in pen and approx(pen[3], 2.0)
print(f"[PASS] exponential scaling: 3-gram -> penalty {pen[3]}")


# ── 极端长循环：模拟 2000 次重复中的一小段 ──────────────────
# [42,99] * 10 → context，追加 42 应该匹配 19 长度
seq = [42, 99] * 10
seq = seq[:-1]   # [42,99,42,99,...,42] 长度 19
# 追加 99 会匹配长度 ~20 的 pattern吗？其实追加 99 后匹配多少个位置？
# 最后一个位置 p=0(42), p=1(99)... 哪个 = 99 且扩展最远？
# 实际 ids 尾部是 ...42, 追加 99 → 新尾部是 42,99。找 p 使 ids[p]=99 且 ids[p-1]=42。
# 多个位置都行。最长扩展 = m[最后合适 p-1] + 1。
pen = compute_penalties(seq, allowed_length=4, multiplier=0.8, base=1.75, max_match=20)
# 具体惩罚值不重要，关键是必须非零
if 99 in pen:
    print(f"[PASS] cyclic pattern: token 99 -> penalty {pen[99]:.4f} (matched length embedded)")
else:
    raise AssertionError(f"循环 pattern 未被识别: {pen}")


# ── 边界：短上下文 ────────────────────────────────────────
assert compute_penalties([], allowed_length=2, multiplier=0.8, base=1.75, max_match=20) == {}
assert compute_penalties([1], allowed_length=2, multiplier=0.8, base=1.75, max_match=20) == {}
print("[PASS] empty / single-token context -> no penalties")


# ── 匹配长度受 max_match 限制 ──────────────────────────────
# 非常长的重复，但 max_match=5 应该截断
seq = list(range(50)) + list(range(50))   # 两遍 0..49
seq = seq[:-1]  # 去掉最后一个
# 追加 49 会匹配长度 50，但 max_match=5 限制到 5
pen = compute_penalties(seq, allowed_length=2, multiplier=1.0, base=2.0, max_match=5)
# 语义：max_match 卡 m[p]（回溯），+1 给候选 token，所以有效 L 上限 = max_match+1 = 6
# 惩罚上限 = 2^(6-2) = 16
assert 49 in pen
assert pen[49] <= 16.0 + 1e-6, f"max_match 未生效: {pen[49]}"
print(f"[PASS] max_match=5 bound: L capped at 6 -> penalty {pen[49]:.2f} (max 16)")


print("\n所有测试通过。")
