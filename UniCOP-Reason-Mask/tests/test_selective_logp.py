"""验证 _selective_logp_chunked 与原 log_softmax().gather() 数值 + 梯度等价。

背景: grpo_prm_trainer._compute_loss 原本用 `torch.log_softmax(completion_logits, -1)`
对整块 [B, T, V] 做 softmax, 会再物化一个等大张量 → CUDA OOM (train_cvrp20_v5 第 193 步崩)。
改为分块 + fp32 + gradient-checkpoint 的逐 token logprob。本测试证明改动只省显存、不改语义。

在 unicop 环境运行:
    cd UniCOP-Reason-Mask && python -m pytest tests/test_selective_logp.py -v
或直接:
    python tests/test_selective_logp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.checkpoint as tc

from grpo_prm_trainer import _selective_logp_chunked, _LOGP_CHUNK_SIZE


def _reference(logits, index):
    """原始等价实现 (fp32 真值): log_softmax 再 gather。"""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def test_forward_equivalence():
    """前向数值与原 log_softmax().gather() 等价 (覆盖分块边界)。"""
    torch.manual_seed(0)
    B, V = 3, 1537
    # T 取值覆盖: 单块 / 恰好整块 / 跨界+1 / 多块 / 真实 completion 长度
    for T in [70, _LOGP_CHUNK_SIZE, _LOGP_CHUNK_SIZE + 1, 1100, 3215]:
        logits = torch.randn(B, T, V, dtype=torch.float32)
        index = torch.randint(0, V, (B, T))
        out = _selective_logp_chunked(logits, index)
        ref = _reference(logits, index)
        assert out.shape == (B, T), f"T={T}: shape {out.shape} != {(B, T)}"
        err = (out - ref).abs().max().item()
        assert err < 1e-5, f"T={T}: 前向最大误差 {err} 超阈值"


def test_grad_equivalence_through_checkpoint():
    """通过 torch.utils.checkpoint 反传, 梯度与原实现等价。"""
    torch.manual_seed(1)
    B, T, V = 3, 1100, 1537
    index = torch.randint(0, V, (B, T))
    base = torch.randn(B, T, V, dtype=torch.float32)

    lg1 = base.clone().requires_grad_(True)
    out1 = tc.checkpoint(_selective_logp_chunked, lg1, index, use_reentrant=False)
    out1.sum().backward()

    lg2 = base.clone().requires_grad_(True)
    _reference(lg2, index).sum().backward()

    err = (lg1.grad - lg2.grad).abs().max().item()
    assert err < 1e-5, f"梯度最大误差 {err} 超阈值"


def test_bf16_is_more_stable_than_old():
    """bf16 输入下, 新方法(块内 upcast fp32) 比原 bf16 log_softmax 更接近 fp32 真值。"""
    torch.manual_seed(2)
    B, T, V = 3, 1100, 1537
    logits_bf = torch.randn(B, T, V, dtype=torch.bfloat16)
    index = torch.randint(0, V, (B, T))

    truth = _reference(logits_bf.float(), index)                  # fp32 真值
    new_bf = _selective_logp_chunked(logits_bf, index)            # 新: 块内 fp32
    old_bf = (torch.log_softmax(logits_bf, dim=-1)
              .gather(-1, index.unsqueeze(-1)).squeeze(-1).float())  # 原: 纯 bf16

    new_err = (new_bf - truth).abs().max().item()
    old_err = (old_bf - truth).abs().max().item()
    # 新方法不应比原方法更差 (通常显著更准)
    assert new_err <= old_err + 1e-6, f"新方法 err={new_err} 反而劣于原 err={old_err}"


if __name__ == "__main__":
    test_forward_equivalence()
    print("✓ test_forward_equivalence")
    test_grad_equivalence_through_checkpoint()
    print("✓ test_grad_equivalence_through_checkpoint")
    test_bf16_is_more_stable_than_old()
    print("✓ test_bf16_is_more_stable_than_old")
    print("全部通过。")
