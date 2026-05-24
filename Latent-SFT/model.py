"""
Hierarchical Latent Reasoner (HLR) 模型封装.

设计:
  - 主模型 (Qwen 7B / Qwen3 4B + LoRA) 负责 prompt / 显式段 / solution 的处理
  - 独立小 transformer (LatentReasoner) 负责 latent 段的内部自回归 (hidden chain)
  - 监督只在每个 latent 段末位 hidden 与 teacher 同位 hidden 对齐

Phase 1 限制:
  - LatentReasoner KV cache 由内部 attention 处理 (训练侧整段单次 forward, 无需接力)
  - 不带 cross-attention 到主模型 K/V (Phase 2 升级方向)
  - LR 顶层 hidden 经 up_proj 注入主模型 inputs_embeds (A'' 风, Phase 2 改严格 KV inject)
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Step-0-only diag stamp (定位 compute_hlr_loss 内部 hang) ───────────
# 只在前 N 次调用打印, 避免训练循环每步都打日志.
_HLR_LOSS_T0 = time.time()
_HLR_LOSS_RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
_HLR_LOSS_CALL_COUNT = 0
_HLR_LOSS_MAX_DIAG_CALLS = 1   # 只在第 1 次 compute_hlr_loss 打全 stamp


def _loss_stamp(msg: str) -> None:
    if _HLR_LOSS_CALL_COUNT > _HLR_LOSS_MAX_DIAG_CALLS:
        return
    elapsed = time.time() - _HLR_LOSS_T0
    print(f"[LSTAMP rank={_HLR_LOSS_RANK} +{elapsed:6.1f}s call={_HLR_LOSS_CALL_COUNT}] {msg}",
          flush=True)


class SimpleRMSNorm(nn.Module):
    """轻量 RMSNorm。bf16 训练下在 fp32 算 variance，避免数值问题。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x32).to(orig_dtype)


# --- RoPE (LLaMA/Qwen 风格) ---

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding，沿用 LLaMA/Qwen 实现。"""

    def __init__(self, dim: int, max_position: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    def forward(self, seq_len: int, device, dtype, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, H, L, D]，cos/sin: [L, D] → 广播 [1, 1, L, D]"""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# --- 核心组件 ---

class GQAAttention(nn.Module):
    """Grouped-Query Attention + RoPE，causal mask 走 PyTorch SDPA。"""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        if num_heads * head_dim != hidden_size:
            raise ValueError(
                f"num_heads × head_dim ({num_heads}×{head_dim}) ≠ hidden_size ({hidden_size})"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) 必须能被 num_kv_heads ({num_kv_heads}) 整除"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, L, hidden] — 本次输入 (新加的 L 步, 不含 past)
            cos, sin: [L, head_dim] — 已对齐 start_pos 的当前 step RoPE
            past_kv: (K_past, V_past), 形状 [B, num_kv_heads, past_len, head_dim], 或 None

        Returns:
            output: [B, L, hidden]
            new_kv: (K_full, V_full) — past + new, 未 repeat (保留 num_kv_heads 维度省内存)
        """
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 只对 Q 和新 K 应用 (past K 已经 RoPE 过)
        q, k_new = apply_rope(q, k_new, cos, sin)

        # 累积 KV
        if past_kv is not None:
            k_past, v_past = past_kv
            past_len = k_past.size(2)
            k_full = torch.cat([k_past, k_new], dim=2)
            v_full = torch.cat([v_past, v_new], dim=2)
        else:
            past_len = 0
            k_full, v_full = k_new, v_new

        # GQA: 仅 attention 计算时 repeat (cache 保留 unrepeated)
        if self.num_kv_groups > 1:
            k_attn = k_full.repeat_interleave(self.num_kv_groups, dim=1)
            v_attn = v_full.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_attn, v_attn = k_full, v_full

        # Causal mask
        if past_len == 0:
            # Initial forward: 标准 causal
            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=True)
        else:
            # Incremental: Q[i] 可看 K[0 .. past_len + i]
            total_len = past_len + L
            row_idx = torch.arange(L, device=x.device).unsqueeze(1)
            col_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
            mask = (col_idx <= past_len + row_idx)  # bool [L, total_len], True=keep
            attn_out = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=mask, is_causal=False
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), (k_full, v_full)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (Qwen/Llama 风格): down(SiLU(gate) * up)。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LatentReasonerBlock(nn.Module):
    """
    Pre-norm transformer block: SwiGLU + GQA + RoPE。
    结构和主模型 Qwen2 block 一致。
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate_size: int):
        super().__init__()
        self.attn_norm = SimpleRMSNorm(hidden_size)
        self.attn = GQAAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.ffn_norm = SimpleRMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv_new = self.attn(self.attn_norm(x), cos, sin, past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv_new


# --- LatentReasoner 主类 ---

class LatentReasoner(nn.Module):
    """
    Hierarchical Latent Reasoner — 最终设计。

    架构:
        - num_layers (默认 7) 个独立 SwiGLU+GQA+RoPE block
        - down_proj: 主模型 hidden (3584) → LR hidden (896)
        - up_proj:   LR hidden (896) → 主模型 hidden (3584)，共享给所有主模型层
        - layer_emb: 28 个调制向量，区分同一 LR hidden 被注入到不同主模型层时的语义

    Forward 行为:
        1) h_in_main → down_proj → h [B, hidden]
        2) 复制 k 次成 [B, k, hidden]，RoPE 在 attention 内部区分位置
        3) 跑 num_layers 层独立 block，每层输出收集到 list
        4) 返回 list[num_layers × [B, k, hidden]]

    Hidden sharing (1:ratio):
        ratio = num_main_layers / num_layers   (默认 28/7 = 4)
        每个 LR 层 hidden 喂给主模型 ratio 个连续层的 K_proj/V_proj
        在 project_for_main_layer() 里做 layer_emb 调制 + up_proj。
    """

    def __init__(
        self,
        main_hidden_size: int,
        num_main_layers: int = 28,
        hidden_size: int = 896,
        num_layers: int = 7,
        num_heads: int = 7,
        num_kv_heads: int = 1,
        head_dim: int = 128,
        intermediate_size: int = 4736,
        max_latent_steps: int = 128,
    ):
        super().__init__()

        if num_main_layers % num_layers != 0:
            raise ValueError(
                f"num_main_layers ({num_main_layers}) 必须能被 num_layers ({num_layers}) 整除 "
                f"(hidden sharing 比例必须整齐)"
            )

        self.main_hidden_size = main_hidden_size
        self.num_main_layers = num_main_layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.share_ratio = num_main_layers // num_layers

        # 入口降维
        self.down_proj = nn.Linear(main_hidden_size, hidden_size, bias=False)

        # 7 层独立 block (无 weight sharing)
        self.layers = nn.ModuleList([
            LatentReasonerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        self.final_norm = SimpleRMSNorm(hidden_size)

        # 注入主模型用：layer_emb 调制 + 共享 up_proj
        self.layer_emb = nn.Embedding(num_main_layers, hidden_size)
        self.up_proj = nn.Linear(hidden_size, main_hidden_size, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(head_dim, max_position=max_latent_steps)

    def forward(
        self,
        h_in_main: torch.Tensor,
        k: int,
        past_kv: list | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            h_in_main: [B, main_hidden_size] 入口 (主模型显式段末位 hidden)
                       past_kv 为 None 时是 latent 段起点; 不为 None 时同一向量复用为新 step 的 input
            k:         本次 forward 的 latent step 数 (训练时 = 段长, 推理时通常 = 1)
            past_kv:   list of num_layers × (K, V), 或 None (initial forward)
                       K, V 形状 [B, num_kv_heads, past_len, head_dim]

        Returns:
            layer_hiddens: list of num_layers × [B, k, hidden_size]
                           只含本次新算的 k 个 step (不含 past)
            new_past_kv:   list of num_layers × (K_full, V_full)
                           K_full, V_full 形状 [B, num_kv_heads, past_len + k, head_dim]
        """
        B = h_in_main.size(0)

        # 头一个 block 的 attn 用于读 num_kv_heads/head_dim 在 k=0 分支构造空 KV
        sample_attn = self.layers[0].attn

        if k <= 0:
            empty_h = h_in_main.new_zeros(B, 0, self.hidden_size)
            empty_kv = (
                h_in_main.new_zeros(B, sample_attn.num_kv_heads, 0, sample_attn.head_dim),
                h_in_main.new_zeros(B, sample_attn.num_kv_heads, 0, sample_attn.head_dim),
            )
            return (
                [empty_h.clone() for _ in range(self.num_layers)],
                [empty_kv for _ in range(self.num_layers)],
            )

        past_len = 0 if past_kv is None else past_kv[0][0].size(2)

        # 降维 → 复制 k 次作为初始 sequence
        h = self.down_proj(h_in_main)                                    # [B, hidden]
        x = h.unsqueeze(1).expand(B, k, self.hidden_size).contiguous()   # [B, k, hidden]

        # RoPE: 当前 step 位置 = past_len .. past_len + k - 1
        cos, sin = self.rope(k, device=x.device, dtype=x.dtype, start_pos=past_len)

        layer_hiddens: list[torch.Tensor] = []
        new_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for L, layer in enumerate(self.layers):
            past_kv_L = past_kv[L] if past_kv is not None else None
            x, kv_new = layer(x, cos, sin, past_kv=past_kv_L)
            layer_hiddens.append(self.final_norm(x))
            new_past_kv.append(kv_new)

        return layer_hiddens, new_past_kv

    def project_for_main_layer(
        self,
        layer_hiddens: list[torch.Tensor],
        main_layer_idx: int,
    ) -> torch.Tensor:
        """
        给定主模型 layer L (0-indexed)，返回该层 K_proj/V_proj 的输入 hidden。

        逻辑:
            L_lr = L // share_ratio                       # 1:ratio hidden 复用
            base = layer_hiddens[L_lr]                    # [B, k, hidden]
            signal = layer_emb(L)                         # [hidden]
            h_modulated = base + signal                   # [B, k, hidden]
            return up_proj(h_modulated)                   # [B, k, main_hidden]
        """
        L_lr = min(main_layer_idx // self.share_ratio, self.num_layers - 1)
        base = layer_hiddens[L_lr]
        signal = self.layer_emb(
            torch.tensor(main_layer_idx, device=base.device, dtype=torch.long)
        )
        h_modulated = base + signal
        return self.up_proj(h_modulated)


def build_latent_reasoner_from_main(main_model, cfg) -> LatentReasoner:
    """
    根据 HLRConfig 与主模型 config 构造 LatentReasoner.

    自动 1/4 缩放策略 (基座模型无关):
        lr_num_layers     = main_layers // 4
        lr_num_heads      = main_heads // 4
        lr_num_kv_heads   = main_kv_heads // 4
        lr_head_dim       = main_head_dim   (保持 head 维度不变)
        lr_hidden_size    = lr_num_heads × lr_head_dim
        lr_intermediate   = lr_hidden × (main_intermediate / main_hidden)
                            (保持 SwiGLU intermediate 比例, 取 64 的倍数)

    HLRConfig 字段 = 0 时走 auto 推断, > 0 时用 cfg 值 (override).

    适配示例:
      R1-Distill-Qwen-7B  (hidden=3584, L=28, H=28, KV=4, dh=128, FFN=18944)
        → LR L=7, H=7, KV=1, dh=128, hidden=896,  FFN=4736   (~108M)
      Qwen3-4B-Thinking    (hidden=2560, L=36, H=32, KV=8, dh=128, FFN=9728)
        → LR L=9, H=8, KV=2, dh=128, hidden=1024, FFN=3904   (~125M)
        (具体数字以模型真实 config 为准, 这里只是示意)
    """
    main_cfg = main_model.config if hasattr(main_model, "config") else main_model.module.config

    main_hidden = getattr(main_cfg, "hidden_size")
    main_layers = getattr(main_cfg, "num_hidden_layers")
    main_heads = getattr(main_cfg, "num_attention_heads")
    main_kv_heads = getattr(main_cfg, "num_key_value_heads", main_heads)
    main_head_dim = getattr(main_cfg, "head_dim", main_hidden // main_heads)
    main_intermediate = getattr(main_cfg, "intermediate_size", main_hidden * 4)

    # 1/4 缩放 (cfg 字段 > 0 时尊重 user override)
    lr_num_layers = cfg.lr_num_layers if cfg.lr_num_layers > 0 else max(1, main_layers // 4)
    lr_num_heads = cfg.lr_num_heads if cfg.lr_num_heads > 0 else max(1, main_heads // 4)
    lr_num_kv_heads = cfg.lr_num_kv_heads if cfg.lr_num_kv_heads > 0 else max(1, main_kv_heads // 4)
    lr_head_dim = cfg.lr_head_dim if cfg.lr_head_dim > 0 else main_head_dim
    lr_hidden_size = cfg.lr_hidden_size if cfg.lr_hidden_size > 0 else lr_num_heads * lr_head_dim

    # SwiGLU intermediate 保持主模型 ratio, 圆整到 64 倍数
    if cfg.lr_intermediate_size > 0:
        lr_intermediate = cfg.lr_intermediate_size
    else:
        intermediate_ratio = main_intermediate / main_hidden
        raw = int(lr_hidden_size * intermediate_ratio)
        lr_intermediate = ((raw + 63) // 64) * 64   # 向上取整到 64 倍数

    # 三个核心约束
    if lr_num_heads * lr_head_dim != lr_hidden_size:
        raise ValueError(
            f"LR num_heads × head_dim ({lr_num_heads}×{lr_head_dim}={lr_num_heads*lr_head_dim}) "
            f"!= hidden_size ({lr_hidden_size}). 调整 cfg.lr_num_heads / lr_head_dim / lr_hidden_size."
        )
    if lr_num_heads % lr_num_kv_heads != 0:
        raise ValueError(
            f"GQA 不齐: lr_num_heads ({lr_num_heads}) 不能被 lr_num_kv_heads ({lr_num_kv_heads}) 整除."
        )
    if main_layers % lr_num_layers != 0:
        raise ValueError(
            f"hidden sharing 不齐: main_layers ({main_layers}) 不能被 lr_num_layers ({lr_num_layers}) 整除. "
            f"显式设 cfg.lr_num_layers 为 main_layers 的因子 ({main_layers} 的因子)."
        )

    print(f"  [build_latent_reasoner] auto-inferred from main config:")
    print(f"    main: hidden={main_hidden} layers={main_layers} heads={main_heads}/{main_kv_heads} "
          f"head_dim={main_head_dim} intermediate={main_intermediate}")
    print(f"    LR:   hidden={lr_hidden_size} layers={lr_num_layers} heads={lr_num_heads}/{lr_num_kv_heads} "
          f"head_dim={lr_head_dim} intermediate={lr_intermediate}")
    print(f"    hidden sharing ratio = {main_layers // lr_num_layers} (每个 LR 层服务主模型 {main_layers // lr_num_layers} 层)")

    return LatentReasoner(
        main_hidden_size=main_hidden,
        num_main_layers=main_layers,
        hidden_size=lr_hidden_size,
        num_layers=lr_num_layers,
        num_heads=lr_num_heads,
        num_kv_heads=lr_num_kv_heads,
        head_dim=lr_head_dim,
        intermediate_size=lr_intermediate,
        max_latent_steps=cfg.max_latent_steps,
    )


# ====================================================================
# compute_hlr_loss — HLR 训练的核心 loss 函数
# ====================================================================
#
# Loss 构成:
#   L_total = α · L_student_ce + β · L_align + γ · L_teacher_ce
#
#   - Teacher CE (γ): 主模型完整 forward + solution 区域 CE
#                     防止 LoRA 训歪、保留显式 CoT 能力
#   - Student CE (α): 分段流水 forward，explicit/solution 段累加 next-token CE
#   - Align loss (β): 每个 latent 段, 7 个 LR 层 hidden 段末位 ↔
#                     teacher 主模型 (4L+3) 层 hidden 在 teacher_align_pos
#                     L1 + std-normalize, teacher 端 stop-gradient
#
# Phase 1 注入策略 (TODO_INJECT, Phase 2 升级):
#   latent 段用 LR 顶层 hidden 作为主模型 inputs_embeds (A'' 风),
#   主模型自己 forward 这 k 个位置, 累积 KV cache 由 HF 处理 (RoPE 等)。
#   监督仍是 B2 风 (7 hidden 都 align)，注入路径退化到 A''。
#   Phase 2 改成严格 KV inject 时, 只需替换 latent 段 forward 5 行代码。
#
# Phase 1 简化:
#   - batch_size = 1 (HLRDataset.collate_hlr 已经强制)
#   - 段内独立 CE, 忽略段间 cross-CE
#   - 不按 latent 段长 k 加权 align loss


def _get_input_embedding_layer(model):
    """安全访问 PEFT-wrapped 主模型的 input embeddings。"""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    return model.module.get_input_embeddings()


def _segment_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    对一段 forward 的 logits 算 next-token CE。
    返回 (sum_loss, num_targets)。
    若该段没有 valid label 返回 (0, 0)。

    注意: bf16 logits 需先转 fp32 再算 CE, 否则 log_softmax under/overflow 易 NaN
          (HF 内部 model(labels=) 自动做这一步, 我们手动算时也得做)
    """
    # next-token shift: logits[:, :-1] 预测 labels[:, 1:]
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    loss = loss_fn(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
    )
    num_targets = (shifted_labels != -100).sum().item()
    return loss, num_targets


def compute_hlr_loss(
    model,
    latent_reasoner: LatentReasoner,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_labels: torch.Tensor,
    prompt_ids: torch.Tensor,
    segments: list,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
):
    """
    HLR 训练 loss (Phase 1, GC-兼容单次拼接 forward 版).

    设计 (从"逐段 KV cache 接力"改为"单次拼接 inputs_embeds"):
      use_cache=True 与 gradient_checkpointing 互斥 (HF 会强制 use_cache=False),
      原 "prompt → 段1 → 段2 ..." KV cache 接力路径在 GC 下静默失效.
      新设计:
        1. Teacher: 完整 forward 一次 → teacher_loss + teacher_hidden_states (供 LR 用)
        2. LR: 每个 latent 段独立 forward, 输入 = teacher hidden(段起始前一个 token).detach()
           输出 = layer_hiddens (各层 × [1, k, lr_hidden])
        3. Student: prompt + 各段 (含 latent 段 LR输出·up_proj) 拼成完整 inputs_embeds,
           一次 model(inputs_embeds=..., labels=...) → student_loss
           GC 兼容, 不依赖 KV cache, 跨段语义靠 self-attention 自然连接.
        4. Align: 每个 latent 段末位 LR 输出 vs teacher 段末位 hidden (每主模型层 1 个对齐点).
           teacher hidden 端 detach, 不污染主模型 teacher 路径.
           LR input 也用 detach 的 teacher hidden, align 梯度不回传主模型.

    梯度路径:
      α · student CE → 主模型 LoRA (直接) + LR 全部参数 (通过 inputs_embeds 拼接)
      β · align L1   → LR 全部 (含 layer_emb / up_proj), 不更新主模型 (双向 detach)
      γ · teacher CE → 主模型 LoRA

    Args:
        prompt_ids:          [P] 无 batch 维 (token id 序列)
        segments:            list[HLRSegment], explicit/solution 含 ids+labels,
                              latent 含 k + teacher_align_pos + teacher_input_pos

    Returns: (total_loss, teacher_loss[d], student_loss[d], align_loss[d])
    """
    global _HLR_LOSS_CALL_COUNT
    _HLR_LOSS_CALL_COUNT += 1
    _loss_stamp(f"ENTER compute_hlr_loss (teacher_len={teacher_input_ids.size(-1)})")

    device = next(model.parameters()).device
    teacher_input_ids = teacher_input_ids.to(device)
    teacher_attention_mask = teacher_attention_mask.to(device)
    teacher_labels = teacher_labels.to(device)
    prompt_ids_dev = prompt_ids.to(device)
    T_teacher = teacher_input_ids.size(1)
    _loss_stamp(f"after .to(device) / read T_teacher={T_teacher}")

    # ── (1) Teacher forward 一次 ──
    _loss_stamp("BEFORE teacher model.forward (ZeRO-3 first all_gather here)")
    teacher_out = model(
        input_ids=teacher_input_ids,
        attention_mask=teacher_attention_mask,
        labels=teacher_labels,
        output_hidden_states=True,
    )
    _loss_stamp("AFTER teacher model.forward")
    teacher_loss = teacher_out.loss
    teacher_hidden_states = teacher_out.hidden_states  # tuple of (L+1) × [1, T, H]
    # hidden_states[0] = embedding, hidden_states[i] = main layer (i-1) 输出 (0-indexed main)

    # ── (2)+(3) 准备拼接 student inputs_embeds ──
    # Unwrap DeepSpeedEngine / DDP / PEFT 直到能调 get_input_embeddings
    inner_model = model
    while not hasattr(inner_model, "get_input_embeddings") and hasattr(inner_model, "module"):
        inner_model = inner_model.module
    embed_module = inner_model.get_input_embeddings()

    # ZeRO-3 下 embedding.weight 被 partition, 直接调 embed_module(ids) 拿不到完整 weight.
    # 用 deepspeed.zero.GatheredParameters 临时 gather (非 ZeRO-3 下为 no-op).
    try:
        import deepspeed
        gather_ctx = deepspeed.zero.GatheredParameters(
            [embed_module.weight], enabled=True
        )
    except ImportError:
        from contextlib import nullcontext
        gather_ctx = nullcontext()

    student_embeds_parts: list[torch.Tensor] = []
    student_labels_parts: list[torch.Tensor] = []
    align_records: list = []  # [(layer_hiddens, teacher_align_pos)]

    n_segs = len(segments)
    n_lat = sum(1 for s in segments if s.type == "latent")
    _loss_stamp(f"BEFORE gather_ctx (n_segs={n_segs} n_latent={n_lat})")
    with gather_ctx:
        _loss_stamp("inside gather_ctx, before prompt embed")
        # prompt embeds
        prompt_embeds = embed_module(prompt_ids_dev).unsqueeze(0).clone()  # [1, P, H]
        student_embeds_parts.append(prompt_embeds)
        student_labels_parts.append(torch.full(
            (prompt_embeds.size(1),), -100, dtype=torch.long, device=device
        ))
        _loss_stamp("after prompt embed, entering segments loop")

        _seg_explicit = 0
        _seg_latent = 0
        _N_SEG = len(segments)
        for seg_idx, seg in enumerate(segments):
            # 每 5 个 + 最后 15 个全打 (精确定位 hang 在第几个)
            _hit_stamp = (seg_idx % 5 == 0) or (seg_idx >= _N_SEG - 15)
            if seg.type in ("explicit", "solution"):
                if _hit_stamp:
                    _loss_stamp(f"seg {seg_idx}/{_N_SEG} type={seg.type} BEFORE embed")
                seg_ids = seg.ids.to(device)
                seg_labels = seg.labels.to(device)
                # clone() 让 embed 在 context 退出后仍有效
                seg_embeds = embed_module(seg_ids).unsqueeze(0).clone()  # [1, T_seg, H]
                student_embeds_parts.append(seg_embeds)
                student_labels_parts.append(seg_labels)
                _seg_explicit += 1
                if _hit_stamp:
                    _loss_stamp(f"seg {seg_idx}/{_N_SEG} type={seg.type} AFTER embed (len={seg_ids.size(0)})")

            elif seg.type == "latent":
                k = seg.k
                if _hit_stamp:
                    _loss_stamp(f"seg {seg_idx}/{_N_SEG} type=latent k={k} BEFORE LR forward")
                # LR 输入: teacher 段起始前 token 最后一层 hidden (双向 detach)
                in_pos = seg.teacher_input_pos
                in_pos = min(max(in_pos, 0), T_teacher - 1)
                h_input = teacher_hidden_states[-1][:, in_pos, :].detach()  # [1, H]

                layer_hiddens, _ = latent_reasoner(h_input, k=k)
                # list of N_lr × [1, k, lr_hidden]

                # 注入: up_proj(顶层 hidden)
                top_hidden = layer_hiddens[-1]
                latent_inputs_embeds = latent_reasoner.up_proj(top_hidden)  # [1, k, H]
                student_embeds_parts.append(latent_inputs_embeds)
                student_labels_parts.append(torch.full(
                    (k,), -100, dtype=torch.long, device=device
                ))

                align_records.append((layer_hiddens, seg.teacher_align_pos))
                _seg_latent += 1
                if _hit_stamp:
                    _loss_stamp(f"seg {seg_idx}/{_N_SEG} type=latent k={k} AFTER LR forward")

            else:
                raise ValueError(f"未知 segment 类型: {seg.type}")

        # 退出 gather_ctx 前打 stamp (gather_ctx 退出是 ZeRO-3 partition collective,
        # 4 rank 必须同时退出. 如果 rank 1/2 到这里但 rank 0/3 已经退出 → 矛盾说明
        # GatheredParameters 退出不是阻塞 collective, 或者是 async pending 后续 hang)
        _loss_stamp(f"END of segments loop, BEFORE exit gather_ctx")

    _loss_stamp("AFTER segments loop (exited gather_ctx)")
    student_embeds = torch.cat(student_embeds_parts, dim=1)        # [1, T_student, H_main]
    student_labels = torch.cat(student_labels_parts, dim=0).unsqueeze(0)  # [1, T_student]

    # ── 防御: 强制 rank 同步到 student forward 入口 ──
    # 不同样本 segments 数异质 (rank 0:42 latent vs rank 1:48 latent), 导致
    # rank 0/3 先到 student forward 触发 ZeRO-3 all_gather collective,
    # rank 1/2 还在 segments loop 跑 LR forward 没到, all_gather 死锁.
    # barrier 让 4 rank 同时进 model.forward, ZeRO-3 collective 顺序对齐.
    if torch.distributed.is_initialized():
        _loss_stamp(f"BEFORE barrier (rank-sync before student forward)")
        torch.distributed.barrier()
        _loss_stamp(f"AFTER barrier (all ranks aligned)")

    _loss_stamp(f"BEFORE student forward (student_len={student_embeds.size(1)})")

    # ── (3) Student forward 一次 (GC 友好, 不需 use_cache) ──
    student_out = model(
        inputs_embeds=student_embeds,
        labels=student_labels,
        # 不开 output_hidden_states (Phase 1 不用 student hidden), 省显存
    )
    _loss_stamp("AFTER student forward")
    student_loss = student_out.loss
    if student_loss is None:
        # 全 -100 边界情况 (理论上不会发生, 至少有 solution segment)
        student_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # ── (4) Align loss: 每段末位 × 每主模型层 ──
    align_loss_sum = torch.tensor(0.0, device=device)
    align_seg_count = 0
    num_main_layers = latent_reasoner.num_main_layers

    for layer_hiddens, teacher_align_pos in align_records:
        teacher_align_pos = min(max(teacher_align_pos, 0), T_teacher - 1)
        seg_align = torch.tensor(0.0, device=device)
        for L_main in range(num_main_layers):
            teacher_hs_idx = L_main + 1   # skip embedding [0]
            h_lr_main = latent_reasoner.project_for_main_layer(
                layer_hiddens, L_main
            )[:, -1, :]   # [1, H_main]  段末位 LR 输出
            h_teacher = teacher_hidden_states[teacher_hs_idx][
                :, teacher_align_pos, :
            ].detach()    # [1, H_main]  teacher 段末位 (stop-grad)
            std = h_teacher.std(dim=-1).mean().clamp(min=0.1)
            seg_align = seg_align + (h_lr_main - h_teacher).abs().mean() / std

        seg_align = seg_align / num_main_layers
        align_loss_sum = align_loss_sum + seg_align
        align_seg_count += 1

    if align_seg_count > 0:
        align_loss = align_loss_sum / align_seg_count
    else:
        align_loss = torch.tensor(0.0, device=device)

    total_loss = alpha * student_loss + beta * align_loss + gamma * teacher_loss

    return (
        total_loss,
        teacher_loss.detach(),
        student_loss.detach(),
        align_loss.detach(),
    )
