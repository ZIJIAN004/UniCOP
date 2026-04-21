"""DRY Processor 单元测试——纯数值校验，不依赖 GPU / 模型。"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dry_processor import ThinkOnlyDRYProcessor


class FakeTokenizer:
    """最小 tokenizer，用于测试；把字符串 token 化为固定 id 序列。"""

    VOCAB = {
        "</think>": [9990, 9991],   # 假装 </think> 是 2 个 token
        "\n\n": [20],
    }

    def encode(self, s: str, add_special_tokens: bool = False):
        return list(self.VOCAB.get(s, []))


def _build(tokenizer=None, **kwargs):
    tokenizer = tokenizer or FakeTokenizer()
    defaults = dict(multiplier=0.8, base=1.75, allowed_length=2, max_match=20)
    defaults.update(kwargs)
    return ThinkOnlyDRYProcessor(tokenizer=tokenizer, **defaults)


def _score_row(vocab_size=100):
    return torch.zeros(1, vocab_size)


# ── 基础：没出现 </think>，应该在 think 内启用，长重复被惩罚 ──────────────
def test_long_repeat_penalized():
    proc = _build()
    # context: 1,2,3,4,5,1,2,3,4 （后段与前段匹配 4 个 token）
    # 若下一个 token = 5，则匹配 5-gram；allowed_length=2，会被惩罚
    ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4]], dtype=torch.long)
    scores = _score_row()
    original = scores.clone()

    out = proc(ids, scores)
    # token 5 应该被惩罚
    assert out[0, 5] < original[0, 5], "长匹配的 token 5 应被惩罚"
    # token 99 (不在历史里) 不应变化
    assert out[0, 99] == original[0, 99]


# ── 短匹配不应被惩罚 ──────────────────────────────────────────────
def test_short_match_not_penalized():
    proc = _build(allowed_length=4)
    ids = torch.tensor([[1, 2, 3, 4, 1, 2]], dtype=torch.long)
    scores = _score_row()
    out = proc(ids, scores)
    # 匹配长度只有 3，低于 allowed=4，不惩罚
    assert out[0, 3].item() == 0.0, "短匹配不应被惩罚"


# ── </think> 出现后必须直接跳过 ────────────────────────────────────
def test_skip_after_think_end():
    proc = _build()
    # 9990,9991 = </think>
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 9990, 9991, 1, 2, 3, 4]], dtype=torch.long)
    scores = _score_row()
    original = scores.clone()
    out = proc(ids, scores)
    # </think> 之后应跳过，全部零变化
    assert torch.equal(out, original), "think 结束后不应再惩罚"


# ── 惩罚强度应随长度指数增长 ─────────────────────────────────────
def test_penalty_scales_exponentially():
    proc = _build(allowed_length=2, multiplier=1.0, base=2.0)
    # 匹配长度 3：penalty = 1.0 * 2^(3-2) = 2
    ids = torch.tensor([[1, 2, 3, 1, 2]], dtype=torch.long)
    scores = _score_row()
    out = proc(ids, scores)
    assert abs(out[0, 3].item() - (-2.0)) < 1e-6

    # 匹配长度 4：penalty = 1.0 * 2^(4-2) = 4
    proc2 = _build(allowed_length=2, multiplier=1.0, base=2.0)
    ids2 = torch.tensor([[1, 2, 3, 4, 1, 2, 3]], dtype=torch.long)
    scores2 = _score_row()
    out2 = proc2(ids2, scores2)
    assert abs(out2[0, 4].item() - (-4.0)) < 1e-6


# ── batch 多样本：每条独立判断 </think> ────────────────────────────
def test_batch_per_sample_gating():
    proc = _build()
    # 样本 0：think 内；样本 1：已过 think
    ids = torch.tensor([
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 0, 0, 0, 0],   # 未含 </think>，9990,9991 不在
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 9990, 9991, 0, 0],  # 已出现
    ], dtype=torch.long)
    scores = torch.zeros(2, 100)
    out = proc(ids, scores)
    assert out[0, 5] < 0, "样本 0 应被惩罚"
    assert out[1, 5] == 0, "样本 1 已过 think，不应惩罚"


# ── 短上下文不崩溃 ──────────────────────────────────────────────
def test_tiny_context():
    proc = _build()
    ids = torch.tensor([[1]], dtype=torch.long)
    scores = _score_row()
    original = scores.clone()
    out = proc(ids, scores)
    assert torch.equal(out, original)


if __name__ == "__main__":
    test_long_repeat_penalized()
    test_short_match_not_penalized()
    test_skip_after_think_end()
    test_penalty_scales_exponentially()
    test_batch_per_sample_gating()
    test_tiny_context()
    print("All tests passed.")
