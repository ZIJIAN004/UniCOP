"""
Latent-SFT 配置：CODI 式隐式推理训练。

Pipeline 定位：
  ① UniCOP-Distill (SFT) → ② GRPO (显式 CoT) → ③ Latent-SFT (本阶段) → ④ Latent-GRPO (待定)

本阶段目标：让模型学会用 latent token 替代显式 <think> 链，
           通过 hidden state 对齐保证隐式推理编码了等价信息。
"""

from dataclasses import dataclass, field


@dataclass
class LatentSFTConfig:
    # ── 模型 ──
    model_name: str = "./output_grpo/final_model"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # ── 数据 ──
    data_path: str = "../UniCOP-Distill/data/chains_self_cvrp20.jsonl"
    filter_problems: list[str] = field(default_factory=lambda: ["cvrp"])
    filter_sizes: list[int] = field(default_factory=lambda: [20])
    max_length: int = 8192

    # ── Latent 配置 ──
    num_latent_tokens: int = 16
    latent_init_std: float = 0.02

    # ── CODI Loss 权重 ──
    alpha: float = 1.0   # student CE (答案预测)
    beta: float = 1.0    # hidden state 对齐
    gamma: float = 1.0   # teacher CE (保持显式推理能力)

    # ── 训练 ──
    seed: int = 42
    num_epochs: int = 3
    lr: float = 2e-5
    latent_lr: float = 1e-3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # ── 多卡 ──
    zero_stage: int = 0
    gradient_checkpointing: bool = False

    # ── 输出 ──
    output_dir: str = "./output_latent_sft"
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3

    # ── 推理（inference.py 用）──
    entropy_window: int = 3
    max_latent_steps: int = 48
