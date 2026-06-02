"""
batched LR forward 数值等价对拍 (correctness gate, CPU, 不需主模型).

验证: 把 n 个 latent 段从"逐段 LR forward"合并成"一次 batched forward (pad 到 max_k)"后,
      注入 embeds 与 align 段末位 hidden 与逐段路径逐元素一致。

复刻 compute_hlr_loss 里的两条真实下游消费:
  - 注入: up_proj(layer_hiddens[-1]) → student 序列 (model.py:726-727), 按真实 k 切片去 pad
  - align: 每层段末位 lh[:, -1, :] (model.py:787), batched 端切 [:k] 使 -1 落到真实 k-1 而非 pad

fp32 下做严格对拍 (训练是 bf16, 等价性是算子层面的代数恒等, 与 dtype 无关; fp32 更易暴露逻辑错)。
退出码 0=等价通过, 1=不等价 (smoke 据此决定是否保留 batched 配置)。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LatentReasoner


def _per_segment(lr, h_inputs, ks):
    """逐段路径: 对每段单独 lr(h, k=k), 收注入 embeds 与 align 段末位 (各层)。"""
    injs, aligns = [], []
    for h, k in zip(h_inputs, ks):
        layer_hiddens, _ = lr(h, k=k)                              # num_layers × [1, k, H_lr]
        injs.append(lr.up_proj(layer_hiddens[-1]))                 # [1, k, H_main]
        aligns.append(torch.stack([lh[:, -1, :] for lh in layer_hiddens], dim=0))  # [nlr, 1, H_lr]
    return injs, aligns


def _batched(lr, h_inputs, ks):
    """批量化路径: 一次 lr(stack, k=max_k), 再按真实 k 切片。"""
    max_k = max(ks)
    h_stack = torch.cat(h_inputs, dim=0)                           # [n, H_main]
    lh_b, _ = lr(h_stack, k=max_k)                                 # num_layers × [n, max_k, H_lr]
    inj_b = lr.up_proj(lh_b[-1])                                   # [n, max_k, H_main]
    injs, aligns = [], []
    for j, k in enumerate(ks):
        injs.append(inj_b[j:j + 1, :k, :])                         # [1, k, H_main] 去 pad
        lh_seg = [lh[j:j + 1, :k, :] for lh in lh_b]               # 各层 [1, k, H_lr] 去 pad
        aligns.append(torch.stack([lh[:, -1, :] for lh in lh_seg], dim=0))  # -1 = 真实 k-1
    return injs, aligns


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    # 小号 LR (满足三约束: heads×head_dim==hidden, main_layers%layers==0, heads%kv_heads==0)
    lr = LatentReasoner(
        main_hidden_size=256, num_main_layers=8,
        hidden_size=64, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=32,
        intermediate_size=128, max_latent_steps=16,
    ).eval()

    # 故意用异质 k (含 max_k 与各种短段), 覆盖 pad 切片逻辑
    ks = [3, 5, 8, 4, 6, 8, 3]
    h_inputs = [torch.randn(1, 256) for _ in ks]

    with torch.no_grad():
        inj_ref, align_ref = _per_segment(lr, h_inputs, ks)
        inj_bat, align_bat = _batched(lr, h_inputs, ks)

    ok = True
    atol, rtol = 1e-5, 1e-4
    for idx, (a, b) in enumerate(zip(inj_ref, inj_bat)):
        d = (a - b).abs().max().item()
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            ok = False
            print(f"  ❌ 注入 seg{idx} (k={ks[idx]}) 不等价: max|Δ|={d:.3e}")
    for idx, (a, b) in enumerate(zip(align_ref, align_bat)):
        d = (a - b).abs().max().item()
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            ok = False
            print(f"  ❌ align seg{idx} (k={ks[idx]}) 不等价: max|Δ|={d:.3e}")

    if ok:
        print(f"  ✓ batched LR ≡ 逐段 LR  (n={len(ks)} 段, k={ks}, atol={atol} rtol={rtol})")
        sys.exit(0)
    else:
        print("  ❌ batched LR 与逐段不等价 — 逻辑有 bug, 禁止上 batched 训练")
        sys.exit(1)


if __name__ == "__main__":
    main()
