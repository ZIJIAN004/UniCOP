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
                if t_pos >= teacher_hidden[layer_idx].size(1):
                    continue
                if s_pos >= student_hidden[layer_idx].size(1):
                    continue
                h_t_list.append(teacher_hidden[layer_idx][b, t_pos, :].detach())
                h_s_list.append(student_hidden[layer_idx][b, s_pos, :])

        if not h_t_list:
            continue

        h_t_batch = torch.stack(h_t_list)
        h_s_batch = torch.stack(h_s_list)
        std = h_t_batch.std(dim=-1).mean().clamp(min=1e-6)
        layer_loss = (h_t_batch - h_s_batch).abs().mean() / std

        align_loss = align_loss + layer_loss
        num_align_layers += 1

    if num_align_layers > 0:
        align_loss = align_loss / num_align_layers

    total_loss = alpha * student_loss + beta * align_loss + gamma * teacher_loss

    return total_loss, teacher_loss.detach(), student_loss.detach(), align_loss.detach()
