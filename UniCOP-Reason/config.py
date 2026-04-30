from dataclasses import dataclass


@dataclass
class Config:
    # ── 问题 ─────────────────────────────────────────────────────────
    # 可选: tsp | tsptw | cvrp | vrptw
    # TSP/CVRP/VRPTW 走 POMO-PRM; TSPTW 走 PIP-D (NeurIPS 2024) backbone
    # TSPDL 暂未启用 (没有 POMO/PIP-D ckpt)
    problem_type: str  = "tsp"
    problem_size: int  = 10      # 客户节点数（不含 depot）

    # ── 模型 ─────────────────────────────────────────────────────────
    model_name: str   = "/home/ntu/lzj/Model/model/DeepSeek-R1-Distill-Qwen-7B"
    use_lora: bool    = True
    lora_rank: int    = 64
    lora_alpha: int   = 128

    # ── 数据集 ───────────────────────────────────────────────────────
    num_train: int    = 20000
    num_test: int     = 1000
    data_seed: int    = 42

    # ── GRPO 训练 ────────────────────────────────────────────────────
    # num_generations=8: 朴素扩量，缓解 gradient starvation（替代 DAPO Dynamic Sampling）
    # max_completion_length=4096: 平衡显存与截断率，不够再调
    num_generations: int               = 8
    max_prompt_length: int             = 768
    max_completion_length: int         = 4096
    learning_rate: float               = 5e-6   # GRPO + LoRA rank 64 推荐 5e-6~2e-5 (原 1e-6 过低,SFT 已升到 1e-4)
    per_device_train_batch_size: int   = 4
    gradient_accumulation_steps: int   = 8
    num_train_epochs: int              = 3
    warmup_ratio: float                = 0.05
    kl_coef: float                     = 0.01
    # n-gram 硬禁，0=关闭；与 evaluate.py 同义，抑制 think 段长程退化循环
    no_repeat_ngram_size: int          = 6

    # ── DAPO Clip-Higher (非对称 ratio clipping，缓解熵坍缩) ─────
    # ε_high > ε_low：加概率快、扣概率慢，让低概率"探索 token"指数累积逃出
    # DAPO 论文推荐 ε_low=0.20, ε_high=0.28；ε_high == ε_low 时退化为标准对称 clip
    clip_epsilon_low: float            = 0.20
    clip_epsilon_high: float           = 0.28

    # ── 多卡训练 ─────────────────────────────────────────────────────
    # num_gpus 需与启动命令的 --num_processes 保持一致，脚本本身不控制进程数。
    # 启动示例（4卡）：accelerate launch --num_processes 4 train.py --num_gpus 4
    #
    # zero_stage 选择建议：
    #   0 → 单卡，不启用 DeepSpeed
    #   2 → 多卡 1.5B，优化器+梯度分片，显存够用时推荐
    #   3 → 多卡 7B，完全分片（模型权重也拆分），3090×N 跑 7B 必须用此模式
    num_gpus:               int  = 1
    zero_stage:             int  = 2
    gradient_checkpointing: bool = False   # 以重计算换显存，ZeRO-3 + 7B 时建议开启

    # ── POMO PRM ─────────────────────────────────────────────────────
    pomo_ckpt_dir: str      = ""    # POMO checkpoint 根目录，子目录格式: {type}_n{size}/MODEL_BEST.pt
    pomo_baseline_dir: str  = ""    # POMO-Baseline 项目根目录（用于导入模型/环境代码）
    pomo_device: str        = "cuda"
    # TSPTW 走 PIP-D (NeurIPS 2024) backbone,与 POMO 参数并存
    pipd_ckpt_dir: str      = ""    # 指向 {PIP-D baseline}/POMO+PIP/pretrained/TSPTW
    pipd_dir: str           = ""    # 指向 {PIP-D baseline}/POMO+PIP (代码目录,用于 sys.path 注入)

    # ── 双信号合成（terminal "对不对" + PRM "好不好"，都走 GRPO 组归一化） ────
    # loss = α · L_terminal + β · L_prm
    # 两个 loss 各自按"自己的有效 token 数"归一，量纲对齐，α=β=1.0 即平衡
    terminal_alpha: float    = 1.0
    prm_beta: float          = 1.0
    # Terminal 4 维等权（parse + coverage + constraint + format），可单独调
    terminal_w_parse: float      = 1.0
    terminal_w_coverage: float   = 1.0
    terminal_w_constraint: float = 1.0
    terminal_w_format: float     = 1.0

    # ── 输出 ─────────────────────────────────────────────────────────
    output_dir: str    = "./output"
    logging_steps: int = 10
    save_steps: int    = 100
    use_wandb: bool    = False

    # ── 评估（evaluate.py 专用） ──────────────────────────────────────
    # model_type 决定评估时的生成长度上限：
    #   reasoning → 推理模型（DeepSeek-R1-Distill 系列），需长 completion 容纳 <think> 链
    #   instruct  → 普通指令模型（Qwen2.5-Instruct 等），直接输出答案，无需长 completion
    eval_model_type: str = "reasoning"
    eval_max_completion_length_reasoning: int = 4096
    eval_max_completion_length_instruct:  int = 512

    # ── 评估后端 ─────────────────────────────────────────────────────
    eval_backend: str = "local"           # local | api

    # ── Vertex AI Gemini（backend=api 时使用） ────────────────────────
    gcp_project: str     = "keen-oasis-489308-m8"
    gcp_location: str    = "us-central1"
    gcp_credentials: str = ""             # 服务账号 JSON 密钥路径，命令行 --gcp_credentials 指定
    api_model: str       = "gemini-2.5-flash"
    api_max_concurrency: int = 5          # API 最大并发请求数


config = Config()
