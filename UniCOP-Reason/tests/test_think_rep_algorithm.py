"""ThinkOnlyRepPenaltyProcessor 算法逻辑校验（不依赖 torch 的 mock 版本）。"""

import math


def apply_rep_penalty(logits_dict, seen_tokens, exempt_ids, penalty):
    """模拟 HF 风格 rep_penalty 在单条序列上的效果。

    logits_dict: {token_id: float}
    seen_tokens: set of token ids seen so far
    exempt_ids: set of token ids to exempt
    """
    effective_seen = seen_tokens - exempt_ids
    out = dict(logits_dict)
    for tid, val in logits_dict.items():
        if tid in effective_seen:
            out[tid] = val / penalty if val > 0 else val * penalty
    return out


# ── 基础：seen 的 token 被扣分，其他不变 ─────────────────────────
def test_basic_penalty():
    logits = {0: 2.0, 1: 1.5, 2: -0.5, 3: 3.0}
    seen = {0, 1, 2}
    out = apply_rep_penalty(logits, seen, exempt_ids=set(), penalty=1.2)
    assert math.isclose(out[0], 2.0 / 1.2)
    assert math.isclose(out[1], 1.5 / 1.2)
    assert math.isclose(out[2], -0.5 * 1.2)   # 负值乘 penalty
    assert math.isclose(out[3], 3.0)          # 未见过，不变
    print("[PASS] basic penalty")


# ── 白名单豁免：exempt 里的 token 不被扣分 ───────────────────────
def test_exempt_zero():
    logits = {0: 5.0, 1: 2.0, 2: 1.0}
    seen = {0, 1, 2}
    out = apply_rep_penalty(logits, seen, exempt_ids={0}, penalty=1.5)
    assert math.isclose(out[0], 5.0)          # depot 0 不惩罚
    assert math.isclose(out[1], 2.0 / 1.5)
    assert math.isclose(out[2], 1.0 / 1.5)
    print("[PASS] exempt depot")


# ── 多个豁免 token ─────────────────────────────────────────────
def test_multi_exempt():
    logits = {10: 3.0, 20: 3.0, 30: 3.0, 40: 3.0}
    seen = {10, 20, 30, 40}
    out = apply_rep_penalty(logits, seen, exempt_ids={10, 20}, penalty=2.0)
    assert math.isclose(out[10], 3.0)
    assert math.isclose(out[20], 3.0)
    assert math.isclose(out[30], 1.5)
    assert math.isclose(out[40], 1.5)
    print("[PASS] multi exempt")


# ── penalty=1.0 等价于不惩罚 ──────────────────────────────────
def test_penalty_one_is_noop():
    logits = {0: 1.0, 1: 2.0, 2: -1.0}
    seen = {0, 1, 2}
    out = apply_rep_penalty(logits, seen, exempt_ids=set(), penalty=1.0)
    for k in logits:
        assert math.isclose(out[k], logits[k])
    print("[PASS] penalty=1.0 is noop")


# ── seen 为空时不改 logits ───────────────────────────────────
def test_empty_seen():
    logits = {0: 1.0, 1: 2.0}
    out = apply_rep_penalty(logits, set(), exempt_ids={999}, penalty=1.5)
    assert out == logits
    print("[PASS] empty seen")


if __name__ == "__main__":
    test_basic_penalty()
    test_exempt_zero()
    test_multi_exempt()
    test_penalty_one_is_noop()
    test_empty_seen()
    print("\n所有算法测试通过。")
