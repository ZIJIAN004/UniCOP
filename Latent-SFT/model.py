"""
CODI 模型封装：teacher/student 双路 forward + 多点 hidden state 对齐。

核心机制：
  - 同一模型跑两次 forward（共享参数）
  - Teacher: 标准输入（完整 CoT）
  - Student: latent 段的 embedding 被替换为单一可学习连续向量
  - 对齐: 每个 latent→explicit 边界 + solution 边界，逐层 L1 对齐
         Teacher 端 stop-gradient，梯度只流向 Student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEmbeddings(nn.Module):
    """单一可学习 latent embedding，所有 latent 位置共享。"""

    def __init__(self, hidden_size: int, init_std: float = 0.02):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(hidden_size) * init_std)

    def forward(self):
        return self.embedding


def compute_codi_loss(
    model,
    latent_emb: LatentEmbeddings,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_labels: torch.Tensor,
    student_input_ids: torch.Tensor,
    student_attention_mask: torch.Tensor,
    student_labels: torch.Tensor,
    latent_positions: list[list[int]],
    align_pairs: list[list[tuple[int, int]]],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
):
    """
    CODI 单步训练：两次 forward + 多点对齐 loss。

    Returns:
        total_loss, teacher_loss, student_loss, align_loss
    """
    # ── Teacher forward ──
    teacher_out = model(
        input_ids=teacher_input_ids,
        attention_mask=teacher_attention_mask,
        labels=teacher_labels,
        output_hidden_states=True,
    )
    teacher_loss = teacher_out.loss
    teacher_hidden = teacher_out.hidden_states

    # ── Student forward: 注入 latent embedding ──
    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        embed_layer = model.module.get_input_embeddings()
    student_embeds = embed_layer(student_input_ids).clone()

    latent_vec = latent_emb()
    batch_size = student_embeds.size(0)
    for b in range(batch_size):
        for pos in latent_positions[b]:
            student_embeds[b, pos, :] = latent_vec

    student_out = model(
        inputs_embeds=student_embeds,
        attention_mask=student_attention_mask,
        labels=student_labels,
        output_hidden_states=True,
    )
    student_loss = student_out.loss
    student_hidden = student_out.hidden_states

    # ── 多点对齐 loss: 逐层 L1，teacher 端 stop-gradient ──
    align_loss = torch.tensor(0.0, device=student_embeds.device)
    num_align_layers = 0

    for layer_idx in range(1, len(teacher_hidden)):
        h_t_list, h_s_list = [], []
        for b in range(batch_size):
            for t_pos, s_pos in align_pairs[b]:
                if t_pos < 0 or t_pos >= teacher_hidden[layer_idx].size(1):
                    continue
                if s_pos < 0 or s_pos >= student_hidden[layer_idx].size(1):
                    continue
                h_t_list.append(teacher_hidden[layer_idx][b, t_pos, :].detach())
                h_s_list.append(student_hidden[layer_idx][b, s_pos, :])

        if not h_t_list:
            continue

        h_t_batch = torch.stack(h_t_list)
        h_s_batch = torch.stack(h_s_list)
        std = h_t_batch.std(dim=-1).mean().clamp(min=0.1)
        layer_loss = (h_t_batch - h_s_batch).abs().mean() / std

        align_loss = align_loss + layer_loss
        num_align_layers += 1

    if num_align_layers > 0:
        align_loss = align_loss / num_align_layers

    total_loss = alpha * student_loss + beta * align_loss + gamma * teacher_loss

    return total_loss, teacher_loss.detach(), student_loss.detach(), align_loss.detach()


# ====================================================================
# Hierarchical Latent Reasoner (HLR) — Phase 1 框架
# ====================================================================
#
# 设计:
#   - 主模型 (Qwen 7B + LoRA) 负责 prompt / 显式段 / solution 的处理
#   - 独立小 transformer 负责 latent 段的内部自回归 (hidden chain)
#   - 监督只在每个 latent 段末位 hidden 与 teacher 同位 hidden 对齐
#
# Phase 1 限制:
#   - 小 transformer 不带 KV cache (O(k^2) 复杂度，k 较小时可接受)
#   - 不引入位置编码 (依赖 hidden chain 本身的位置信息)
#   - 不带 cross-attention 到主模型 K/V (TODO_3 attend_scope = "internal")


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
    """根据 HLRConfig 与主模型 config 构造 LatentReasoner。"""
    main_cfg = main_model.config if hasattr(main_model, "config") else main_model.module.config
    main_hidden = getattr(main_cfg, "hidden_size", 3584)
    num_main_layers = getattr(main_cfg, "num_hidden_layers", 28)

    return LatentReasoner(
        main_hidden_size=main_hidden,
        num_main_layers=num_main_layers,
        hidden_size=cfg.lr_hidden_size or 896,
        num_layers=cfg.lr_num_layers,
        num_heads=cfg.lr_num_heads or 7,
        num_kv_heads=cfg.lr_num_kv_heads or 1,
        head_dim=cfg.lr_head_dim or 128,
        intermediate_size=cfg.lr_intermediate_size or 4736,
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
    HLR 训练 loss (Phase 1)。

    Args:
        model:               PEFT-wrapped 主模型 (LoRA + base)
        latent_reasoner:     LatentReasoner 实例
        teacher_input_ids:   [1, T_teacher]
        teacher_attention_mask: [1, T_teacher]
        teacher_labels:      [1, T_teacher]，solution 区间外为 -100
        prompt_ids:          [P]  无 batch 维, prompt token id 序列
        segments:            list[HLRSegment]，HLRDataset 输出的分段
        alpha, beta, gamma:  loss 加权系数

    Returns:
        total_loss, teacher_loss, student_loss, align_loss  (后三者 detach)
    """
    device = next(model.parameters()).device
    teacher_input_ids = teacher_input_ids.to(device)
    teacher_attention_mask = teacher_attention_mask.to(device)
    teacher_labels = teacher_labels.to(device)
    prompt_ids_dev = prompt_ids.to(device).unsqueeze(0)  # [1, P]

    # ── Teacher forward (完整一次) ──
    teacher_out = model(
        input_ids=teacher_input_ids,
        attention_mask=teacher_attention_mask,
        labels=teacher_labels,
        output_hidden_states=True,
    )
    teacher_loss = teacher_out.loss
    teacher_hidden_states = teacher_out.hidden_states  # tuple of (L+1) × [1, T, H]
    # hidden_states[0] = embedding, hidden_states[i] = main layer (i-1) 输出 (0-indexed main)

    # ── Student 分段流水 ──
    student_ce_sum = torch.tensor(0.0, device=device)
    student_token_count = 0

    align_loss_sum = torch.tensor(0.0, device=device)
    align_seg_count = 0

    # Prefill prompt
    prompt_out = model(
        input_ids=prompt_ids_dev,
        use_cache=True,
        output_hidden_states=True,
    )
    past_kv = prompt_out.past_key_values
    last_main_hidden = prompt_out.hidden_states[-1][:, -1, :]  # [1, main_hidden]

    for seg in segments:
        if seg.type in ("explicit", "solution"):
            seg_ids = seg.ids.to(device).unsqueeze(0)
            seg_labels = seg.labels.to(device).unsqueeze(0)

            seg_out = model(
                input_ids=seg_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = seg_out.past_key_values

            seg_ce, n_targets = _segment_ce(seg_out.logits, seg_labels)
            if n_targets > 0:
                student_ce_sum = student_ce_sum + seg_ce
                student_token_count += n_targets

            last_main_hidden = seg_out.hidden_states[-1][:, -1, :]

        elif seg.type == "latent":
            k = seg.k

            # LR forward (训练: 一次性整段, 丢弃 KV cache)
            layer_hiddens, _ = latent_reasoner(last_main_hidden, k=k)
            # list of 7 × [1, k, lr_hidden=896]

            # ── Align loss: 28 个对齐点 (主模型每层各 1 个) ──
            # 同一 LR layer hidden 在不同 layer_emb 调制下应贴近主模型每层 hidden,
            # 这样 layer_emb 28 个向量全都有梯度信号。
            seg_align = torch.tensor(0.0, device=device)
            num_main_layers = latent_reasoner.num_main_layers
            for L_main in range(num_main_layers):
                teacher_hs_idx = L_main + 1   # HF index (skip embedding [0])

                # LR hidden 段末位, 通过 project_for_main_layer 加 layer_emb(L_main) 调制 + up_proj
                h_lr_main = latent_reasoner.project_for_main_layer(
                    layer_hiddens, L_main
                )[:, -1, :]   # [1, main_hidden]

                # Teacher 主模型 layer L_main 在 teacher_align_pos 的 hidden (stop-grad)
                h_teacher = teacher_hidden_states[teacher_hs_idx][
                    :, seg.teacher_align_pos, :
                ].detach()    # [1, main_hidden]

                # L1 + std normalize
                std = h_teacher.std(dim=-1).mean().clamp(min=0.1)
                layer_align = (h_lr_main - h_teacher).abs().mean() / std

                seg_align = seg_align + layer_align

            seg_align = seg_align / num_main_layers
            align_loss_sum = align_loss_sum + seg_align
            align_seg_count += 1

            # ── 注入主模型 (Phase 1: A'' 风, 用 LR 顶层 hidden 作 inputs_embeds) ──
            # TODO_INJECT (Phase 2): 替换为严格 B2 KV inject
            top_hidden = layer_hiddens[-1]                                # [1, k, 896]
            latent_inputs_embeds = latent_reasoner.up_proj(top_hidden)    # [1, k, main_hidden]

            latent_out = model(
                inputs_embeds=latent_inputs_embeds,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = latent_out.past_key_values
            last_main_hidden = latent_out.hidden_states[-1][:, -1, :]

        else:
            raise ValueError(f"未知 segment 类型: {seg.type}")

    # ── 汇总 ──
    if student_token_count > 0:
        student_loss = student_ce_sum / student_token_count
    else:
        student_loss = torch.tensor(0.0, device=device)

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
