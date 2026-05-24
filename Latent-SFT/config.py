"""
Latent-SFT 配置 (HLR 路径).

Pipeline 定位:
  ① UniCOP-Distill (SFT) → ② GRPO (显式 CoT) → ③ Latent-SFT (本阶段) → ④ Latent-GRPO (待定)

本阶段目标: 用独立小 transformer (LatentReasoner) 在 CoT 的低熵确定性段做隐式推理,
           保留高熵段的显式推理, 通过多层 hidden state 对齐保证等价信息编码.
"""

from dataclasses import dataclass, field


@dataclass
class HLRConfig:
    """
    Hierarchical Latent Reasoner 配置 (最终设计).

    LatentReasoner 架构 (B2 + hidden sharing 1:4):
      - 7 层独立 SwiGLU + GQA + RoPE block (无 weight sharing)
      - down_proj 把主模型 hidden (3584) 降到 LR hidden (896)
      - up_proj  把 LR hidden 升回主模型 hidden (共享, 所有主模型层公用)
      - layer_emb (28 × 896): 调制同一 LR hidden 被注入不同主模型层时的语义
      - hidden sharing: 1 个 LR 层 hidden 喂主模型 4 层 K_proj / V_proj

    监督:
      - 每个 latent 段末位 hidden ↔ teacher 主模型对应层 hidden 做 L1 对齐
      - student answer CE + teacher CE 一起加权
    """

    # ── 主模型 ──
    model_name: str = "./output_grpo/final_model"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # ── 数据 (路径相对 UniCOP/ 根目录, 与 sbatch cwd 对齐) ──
    # raw_chains_path: Distill 阶段产出的模板化 CoT, Latent-SFT 共用
    # data_path:       本阶段 entropy_profile 后的派生数据, 放 Latent-SFT/data/ 下
    data_path: str = "Latent-SFT/data/profiled_cvrp20.jsonl"
    raw_chains_path: str = "UniCOP-Distill/data/chains_template_cvrp20.jsonl"
    auto_rebuild_data: bool = False        # train.py main() 检测到 args.data is None 时置 True,
                                            # train_hlr 启动时用 base model 跑 entropy profile
                                            # 覆盖 data_path, 确保数据与本次训练用的基座一致
    filter_problems: list[str] = field(default_factory=lambda: ["cvrp"])
    filter_sizes: list[int] = field(default_factory=lambda: [20])
    max_length: int = 8192
    latent_compression_ratio: int = 4
    min_latent_segment: int = 8

    # ── Latent Reasoner (SwiGLU + GQA + RoPE, 1/4 主模型尺寸) ──
    # 默认全 0 = auto 推断 (build_latent_reasoner_from_main 从主模型 config 自动算 1/4 缩放)
    # 兼容 R1-Distill-Qwen-7B / Qwen3-4B-Thinking 等不同尺寸基座
    # 显式设值 > 0 时覆盖 auto, 用于精细调控
    lr_num_layers: int = 0          # 0 → main_layers // 4 (R1-7B → 7, Qwen3-4B 36 层 → 9)
    lr_hidden_size: int = 0         # 0 → lr_num_heads × lr_head_dim
    lr_num_heads: int = 0           # 0 → main_heads // 4
    lr_num_kv_heads: int = 0        # 0 → main_kv_heads // 4 (保持 GQA 比例)
    lr_head_dim: int = 0            # 0 → main_head_dim (主模型 head_dim 不变)
    lr_intermediate_size: int = 0   # 0 → lr_hidden × (main_intermediate / main_hidden), 64 倍数
    lr_init_method: str = "random"  # TODO: "copy_first_layers" / "svd_compress"

    # ── Loss 权重 ──
    alpha: float = 1.0        # student CE (显式段 + solution 的 next-token)
    beta: float = 1.0         # 段末 hidden L1 对齐
    gamma: float = 1.0        # teacher CE
    align_normalize_by_seglen: bool = True  # 按 1/k 归一化避免长段主导

    # ── 训练 ──
    seed: int = 42
    num_epochs: int = 3
    lr: float = 2e-5                  # 主模型 LoRA
    latent_reasoner_lr: float = 5e-5  # 小 transformer 单独 lr
    per_device_batch_size: int = 1    # Phase 1: 强制 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # ── 多卡 ──
    zero_stage: int = 0
    gradient_checkpointing: bool = False

    # ── 输出 ──
    output_dir: str = "./output_hlr"
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3

    # ── Latent 进出 trigger (新方向: 压缩低熵确定性段以省算力) ──
    # 训练侧 entropy_profile.py 和未来推理侧需共享这些参数, 必须完全一致
    entropy_window: int = 3                   # 趋势窗口 (连续 K 步)
    entropy_quantile: float = 0.5             # 50 分位数 = 中位数, 作为"低熵"阈值
    min_latent_steps: int = 3                 # 进入 latent 后至少走的 step 数 (≈12 显式 token)
    max_latent_steps: int = 8                 # latent step 上限 (≈32 显式 token)
    latent_cooldown: int = 24                 # 退出 latent 后, 至少 24 个显式 token 才允许再进
    min_entropy_samples: int = 10             # 推理时至少观察到 K 个熵, 才用 median 触发
