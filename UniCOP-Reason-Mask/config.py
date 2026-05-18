import os
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
    model_name: str   = os.environ.get("BASE_MODEL", "")
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
    learning_rate: float               = 1e-5   # v5 用 2e-5 第 1 step grad_norm explode (3.87 clip→1.0,
                                                # Δθ=2e-5, 比 v4 第 1 step Δθ=2e-8 大 100,000x), 模型 push 过头进 plateau.
                                                # 降回 1e-5: Δθ 减半, 配合 warmup 0.02 (10 step) 避免第 1 step explode.
    per_device_train_batch_size: int   = 4
    gradient_accumulation_steps: int   = 8
    num_train_epochs: int              = 3
    warmup_ratio: float                = 0.02   # v5 grad spike 修复: warmup 从 5 step (0.01) 延长到 10 step (0.02),
                                                # 让 LR 在 reward outlier 触发 grad spike 时仍小, 避免第 1 step 过冲.
                                                # 仍比 v4 默认 0.05 (25 step) 短一半, 保留 v5 设计 "早看效果" 意图.
    kl_coef: float                     = 0.0    # DAPO 标准 (2025): KL anchor 移除, 用 Clip-Higher 控 entropy.
                                                # 之前 0.01 比 DeepSeek-R1 (0.001) 高 10x, 比 DAPO/trl 默认 (0.0) 大 ∞.
                                                # long-CoT RL (3000+ token) + Clip-Higher 0.20/0.28 已生效, KL 反而 hamper exploration.
                                                # 兜底: 若 step 30+ 看到 entropy collapse / policy degenerate, 升到 0.001 (DeepSeek-R1 安全值).

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

    # ── 奖励模式 ──────────────────────────────────────────────────────
    # prm:   三信号解耦（feasibility gate + outcome + POMO PRM process）
    # foarl: Feasibility-and-Optimality-Aware RL（无 PRM，纯 completion 级奖励）
    #        参考: Jiang et al., NeurIPS 2025, arXiv:2509.16865
    reward_mode: str               = "prm"  # "prm" | "foarl"

    # ── 三信号解耦奖励（reward_mode=prm） ────────────────────────────
    # A_out = A_feasibility + A_outcome（所有 token 共享）
    # A_proc = per-customer 增量 PRM（段内广播到步骤推理段 token）
    # per-token advantage = A_out + α · A_proc
    disable_prm: bool              = False # ablation: 关闭 process reward，只用 outcome
    proc_alpha: float              = 0.5   # process 信号权重；段内广播已覆盖较多 token，0.5 起步
    infeasible_margin: float       = 1.0   # (legacy) 惩罚低于历史最小可行 advantage 的幅度
    infeasible_default_penalty: float = -5.0  # (legacy) 训练初始还没见过可行解时的 fallback
    # A_feasibility 权重（可行解拿满 = w_p + w_cov + w_cons + w_f = 4.0）
    # 改造历史:
    #   v1 各 1.0 独立加权 → 模型优化 cons 牺牲 cov, coverage 反向漂移
    #   v2 w_cc=2.0 cov×cons 乘积 → cov 是离散 hinge (0/1) + cov=0.5/cons=1.0 vs
    #       cov=1.0/cons=0.5 等高, 模型卡 "保守少走" 策略, cov 停在 0.5
    #   v3 (当前) cov 连续化 (n_unique/max(n, n_total)) + cov_gate 硬墙:
    #       - cov 连续 → cov<1 有 gradient, 摆脱 hinge 卡死
    #       - cov_gate 硬墙 → cov < gate 时 cons 信号 = 0,
    #                         强迫模型先把 cov 拉到 gate 才能拿 cons 分
    w_p: float                     = 1.0   # R_parse 权重
    w_cov: float                   = 1.5   # R_coverage 权重 (略高于 cons, 主导)
    w_cons: float                  = 1.0   # R_constraint 权重 (硬墙后才生效)
    w_f: float                     = 0.5   # R_format 权重
    # 硬墙阈值: cov >= cov_gate 时 cons 信号才开门
    #   1.0  → 严格全覆盖+无重复才给 cons (用户默认, 最强 cov 压力)
    #   0.95 → 允许 ≤1 客户误差, 边界更平滑, cold start 友好
    #   0.0  → 等价独立加权 (无硬墙)
    cov_gate: float                = 1.0
    # 异常步固定在 "min - k*std" 位置 (原始 reward 空间), 归一化后等价于
    # z(min) - k, 与 std 绝对大小脱钩, 避免早期 rollout 同质化时
    # margin/std → 10^4 的爆炸. k=3 = 3σ 落在正常分布之外, 惩罚显著且稳定.
    abnormal_sigma: float          = 3.0   # 异常步比最差正常步低多少个 σ
    # 单可行 / 孤儿 anomaly fallback: 当某 (prompt, customer) 上 normal reward
    # 不足 2 条无法 z-score 时, 给绝对常数信号 (量级粗略对齐 ≥2 可行场景的
    # z-score 输出: normal z 量级 ±1~2, anomaly z 量级 -3).
    fallback_normal_value: float   = 1.0   # 单条 normal 时的信号
    fallback_anomaly_value: float  = -3.0  # 对应的 anomaly 信号
    # resample 门控: 前 N step 跳过可行性重采样
    # 训练初期可行率低 (<20%), resample 也大概率失败, 反复 vLLM 调用浪费时间.
    # step >= N 之后模型应已学到基本可行模式, resample 救剩余 outlier 才有意义.
    resample_start_step: int       = 100

    # ── 奖励方案开关 (v3 = 当前 hardgate+PRM cascade, v4 = simplified+absolute PRM) ──
    # v3: 默认, 一字不改原逻辑. hardgate (cov_gate=1.0) + PRM 跨 trajectory z-score
    #     + fallback + anomaly cascade. 已知问题: 7362 run 中 PRM cascade 让模型
    #     选漏访避险, fullcov 65 步后崩.
    # v4: 简化设计.
    #     - A_feas 只剩 parse + format (w_p_v4 + w_f_v4), 没 cov/cons/hardgate
    #     - A_outcome 用 repaired distance: 漏访补单条路线 / 违例贪心拆分 /
    #       重复去重 + dup_distance_eps 固定罚. 所有 parse=1 trajectory 进 outcome 子集.
    #     - PRM 用 absolute: normal step a_proc = prm_base + tanh(R_step) (永远>0),
    #       违例/重复及之后 step 全部游离 a_proc=0 (机会成本通过 base 缺失体现).
    #       不再 z-score / fallback / anomaly cascade.
    #     - 违例惩罚 = PRM 机会成本 + outcome distance 增量
    #     - 漏访惩罚 = PRM 机会成本 (少 K 个 base) + outcome distance 增量
    #     - 重复惩罚 = PRM 机会成本 (cascade 游离) + outcome dup_eps
    # v5: v4 + hardgate distance (修 v4 7414 run 信号弱 + 冷启动).
    #     - A_feas 加回 parse+cov+cons+format (cov_gate=1.0 hardgate, cov=1 才给 cons),
    #       cov+cons 权重加大让 A_feas 主导冷启动信号 (parse/format 几乎恒 1.0 不贡献差异).
    #     - A_outcome 用 raw prob.get_tour_distance (不 repair), 严格 fully_feasible 子集.
    #       前期 fully_feasible<2 时全 0, 完全靠 A_feas + PRM 机会成本推可行性.
    #     - PRM 同 v4: absolute base+tanh, 违例/重复 step 之后游离.
    #     - 配 LR 2e-5 + warmup_ratio 0.01 (5 step) 加快收敛.
    reward_scheme: str             = "v5"   # "v3" | "v4" | "v5"

    # ── v4 专用参数 (v3 时被忽略) ─────────────────────────────────────
    # prm_base: PRM normal step 基础奖励. 必须 > |tanh(R_step)|_max = 1, 给 margin 取 1.5.
    # normal a_proc = prm_base + tanh(R_step) ∈ (0.5, 2.5), 永远 > 违例/重复 (=0).
    prm_base_v4: float             = 1.5
    # proc_alpha_v4: v4 段广播权重. v3 用 proc_alpha=0.5 是因为 sum 模式段贡献 ∝ seg_len;
    # v4 用 mean 模式段贡献 = α × a_proc 与段长解耦, 需要 α 调大才能让 PRM 信号在
    # trajectory loss 中可见. 50 量级让 PRM trajectory 贡献 (≈20段 × 50 × 1.5 = 1500)
    # 跟 A_out trajectory 贡献 (≈z-score × T = 1.5 × 3000 = 4500) 同量级, PRM 占比 25-30%.
    proc_alpha_v4: float           = 50.0
    # dup_distance_eps: outcome 上重复客户的固定 distance 增量 (per duplicate).
    # 设计上 < 漏访 distance 增量 (~0.7) 防止重复罚过重, > 0 让模型感知重复不好.
    dup_distance_eps: float        = 0.2
    # A_feas v4 权重 (只剩 parse + format), 跟 v3 的 w_p/w_f 隔离避免互相影响.
    w_p_v4: float                  = 1.0
    w_f_v4: float                  = 0.5

    # ── v5 专用参数 (v3/v4 时被忽略) ──────────────────────────────────
    # PRM 部分跟 v4 共用 prm_base_v4 / proc_alpha_v4 (PRM 设计相同, 共参数避免重复).
    # A_feas 权重 cov 强主导, cons 减权 (用户决定 2026-05-18):
    #   - v5 初版 hardgate (cov_gate=1.0) + 2.5/2.0 失败: 模型在 cov=1 outlier
    #     主导下没学到细粒度 cov, fullcov 反向漂移.
    #   - 改加法 (cov_gate=0) + cov 强主导: cov 信号始终有效, 权重 3.5:1 让
    #     cov 改善对 A_feas 贡献是 cons 的 3.5x, 避免模型"一昧提升 cons 不改 cov".
    # 权重总和 5.5 (不变), cov:cons = 78%:22% (vs 之前 56%:44%).
    w_p_v5: float                  = 0.5
    w_cov_v5: float                = 3.5
    w_cons_v5: float               = 1.0
    w_f_v5: float                  = 0.5
    # cov_gate_v5: cov >= cov_gate_v5 时 cons_signal = cons else 0.
    # 0.0 (当前) = 纯加法, cov 任何值都给 cons (cov 任何 trajectory 都拿 cons 信号);
    # 1.0 (废) = hardgate, 只有 cov=1 才给 cons (跷跷板模式, 已验证失败).
    cov_gate_v5: float             = 0.0
    # PRM 只对 fully_feasible trajectory 算 (用户决定 2026-05-18):
    # 原 v5 PRM 给所有 trajectory 算 (含 partially feasible), 但 PRM a_proc ∈ (0.5, 2.5)
    # 永远正 → 所有 trajectory advantage 偏正 +0.37 → 破坏 GRPO 零均值假设, 削弱 contrastive.
    # True (当前): PRM 只对 fully_feas (parse+cov=1+cons=1+format=1) 算
    #   - 整体 PRM 偏置 +0.37 → +0.115 (减 70%, GRPO baseline 更稳)
    #   - fully_feas vs partially PRM 差距 +0.13 → +0.46 (强 3.5x, push fully_feas)
    #   - 跟 outcome (只在 fully_feas subset z-score) 设计对称
    # False: 退回原 v5 行为 (所有 trajectory 都算 PRM)
    prm_only_fully_feas_v5: bool   = True

    # ── CVRP constrained-decoding mask (跟 reward_scheme 正交) ────────
    # use_mask=True 时:
    #   1. vLLM server 端: utils/vllm_serve_logprobs.py 必须用 --mask_enabled
    #      --mask_n {N} 启动 (run script 要传); 否则 server 不会真的 mask, 这里
    #      只是个 trainer 端开关.
    #   2. Trainer 端: 日志加 train/use_mask=1 字段区分实验, 第一次没收到
    #      mask_hits 时打 warning 提醒 server 配置不一致.
    #   3. mask 跟 reward_scheme=v3/v4 都兼容. 推荐组合:
    #      - v4 + mask: simplified reward + mask 强制 cov=1, 探索面收窄但更稳
    #      - v3 + mask: hardgate reward + mask 让 cov_gate 自动满足 (cons 总开)
    # mask 强制规则: CVRP-LLM-Mask-完整规则.md
    use_mask: bool                 = False
    mask_n: int                    = 0          # 0 → 跟 problem_size 同步
    mask_debug: bool               = False      # vLLM stderr 详细 mask 触发日志

    # ── FOARL 奖励（reward_mode=foarl） ──────────────────────────────
    # R = R_f + R_o
    # R_f = omega 加权的约束满足度，R_o = alpha / (1 + optimality_gap)
    foarl_alpha: float             = 0.5   # optimality reward scaling
    foarl_omega_parse: float       = 0.2   # 格式可解析权重
    foarl_omega_coverage: float    = 0.3   # 客户覆盖权重
    foarl_omega_constraint: float  = 0.3   # 约束满足权重
    foarl_omega_format: float      = 0.2   # Route 编号正确性权重

    # ── 输出 ─────────────────────────────────────────────────────────
    output_dir: str    = "./output"
    logging_steps: int = 1    # transformers Trainer 父类每 N step log 一次 {'loss','grad_norm','learning_rate'}.
                              # 调试期间 = 1 让 grad_norm 每 step 可见 (核心调参指标).
                              # 注意: self.log() 自定义字段 (reward_v5/*, diag/*) 不受此控制, 每 micro-batch 都触发.
                              # grad_norm 解读: < 0.1 信号弱 (LR/KL 抑制), > 1.0 被 clip, 0.1-0.5 正常.
    # save_steps=50: 配合 train.py 的 resume_from_checkpoint + auto_train.sh
    # 的 vLLM 自动重启. 平衡 IO 开销 vs 崩溃丢失:
    # - ZeRO-3 + LoRA 单次保存 ~1.3 GB (LoRA adapter + AdamW state + sharded grads)
    # - 每 50 step 保存一次 ≈ 14 小时 (实测每 step ~17 min)
    # - 99% 情况下 vLLM 闪挂会被 retry patch 接住, 不需要 resume
    # - 即使罕见 supervisor 也失败, 损失上限 50 step ≈ 14 h, 远好于之前 100 step
    save_steps: int    = 50
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
