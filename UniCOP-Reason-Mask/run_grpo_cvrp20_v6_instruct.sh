#!/bin/bash
# run_grpo_cvrp20_v6_instruct.sh
# ── yangzhihan(A*STAR-Zhihan, 直连 SSH 无 SLURM) 专用 v6 RL 启动器 ──────────────
#   基座 = Qwen3-4B-Instruct-2507 的 SFT 产物(非 thinking, 对齐 FOARL instruct 范式):
#     $DISTILL_DIR/output_sft_qwen3_instruct_template_cvrp20/final_model
#     (zhihan 实际路径: /Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/output_sft_qwen3_instruct_template_cvrp20/final_model)
#
#   本脚本是 run_grpo_cvrp20_v6.sh(纯净 v6 wrapper)的薄封装, 只额外:
#     - 把 RL 起点切到 instruct SFT 产物 (BASE_MODEL_TYPE=qwen3_instruct);
#       paths.sh 据此自动设采样参数 T=0.7/top_p=0.8/top_k=20 (Qwen3-Instruct-2507 官方 & FOARL 受控对比口径);
#     - 设温和超参 (LR/EPOCHS/SAVE_STEPS/NUM_TRAIN/PROC_ALPHA_V6), 与 zhuoyi submit_grpo_cvrp20_v6.sh 同口径,
#       便于与 thinking 版 v6 直接对照;
#     - 输出目录带 instruct 标识 + 超参标注, 与 thinking v6 (output_v6_*) 物理隔离, 避免误 resume。
#   reward 信号: REWARD_SCHEME=v6 (PRM 批级截尾标准化 + sigmoid, proc_alpha_v6) 由 run_grpo_cvrp20_v6.sh 设;
#               A_out (A_feas+A_outcome) 完全复用 v5; use_mask=False(纯净)。
#
#   GPU(zhihan 8×3090 24G): run_grpo_cvrp20_v5.sh 在 astar-zhihan 上自动挑空闲卡 (1 vLLM + ≤6 训练),
#     NCCL 默认开 P2P/SHM 提速 (单机), vLLM gpu_memory_utilization=0.80 正是 24G 卡 + 4B 模型的甜点值。
#
#   ⚠️ zhihan 无 SLURM, 直连 SSH——必须在 tmux 里跑, 否则 SSH 一断 SIGHUP 连带杀光 torchrun/vLLM:
#       tmux new -s v6_instruct
#       cd /Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason-Mask
#       bash run_grpo_cvrp20_v6_instruct.sh
#     (submit_grpo_cvrp20_v6.sh 是 zhuoyi 专用 SLURM 脚本, zhihan 不能用。)
#
#   常用覆盖(扫参/省算力): LR=1e-5 EPOCHS=2 PROC_ALPHA_V6=400 NUM_TRAIN=500 bash run_grpo_cvrp20_v6_instruct.sh

set -euo pipefail
_D="$(cd "$(dirname "$0")" && pwd)"

# ── tmux 提醒 (非强制): 检测到非 tmux/screen 的交互式会话时告警, 不阻断 ─────────────
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ -t 1 ]; then
    echo "⚠️  [警告] 当前不在 tmux/screen 会话中! zhihan 直连 SSH, 会话一断会 SIGHUP 杀光训练+vLLM。"
    echo "    强烈建议先: tmux new -s v6_instruct  再跑本脚本。 (5s 后继续, Ctrl-C 中止)"
    sleep 5
fi

# ── RL 起点: instruct SFT 产物 ──────────────────────────────────────────────
export BASE_MODEL_TYPE=qwen3_instruct          # paths.sh + run_v5 case 据此切 SFT 产物路径与采样参数

# ── 温和超参 (env 可覆盖; 与 zhuoyi submit_grpo_cvrp20_v6.sh 同口径) ─────────────
export REWARD_SCHEME="${REWARD_SCHEME:-v6}"     # PRM 批级截尾标准化+sigmoid (run_grpo_cvrp20_v6.sh 也会兜底设)
export LR="${LR:-2e-5}"                          # 对齐 v5 (1e-6 训练不足); train.py 经 env 覆盖 config
export EPOCHS="${EPOCHS:-1}"                     # 单 epoch
export SAVE_STEPS="${SAVE_STEPS:-20}"           # 每 20 step 存档 (短跑存 step20/40 + final)
export NUM_TRAIN="${NUM_TRAIN:-1000}"           # 一个 epoch 的训练实例数; 扫参可降到 500 提速
export PROC_ALPHA_V6="${PROC_ALPHA_V6:-1000}"   # v6 PRM 段注入权重 (扫参主轴); 默认 1000 (用户决定 2026-06-15, 同 config.py 默认)

# ── A_feas 权重对齐 FOARL 设计 (capacity 主导), 保 v5 总量 5.5 → PRM/A_outcome 标定无需变 ──
#   FOARL CVRP R_f 比例 parse:depot:cov:cap = 0.2:0.1:0.1:0.6 (你的 FOARL/foarl_reward_cvrp.py:75-79)。
#   Mask 无独立 depot 分量、但有 format → 把 depot 的 0.1 位给 format; 按 5.5 总量缩放:
#     parse 0.2→1.1  coverage 0.1→0.55  capacity(=constraint) 0.6→3.3  format 0.1→0.55。
#   train.py 经 W_*_V5 env 覆盖 config.w_*_v5 (v6 复用 v5 A_feas: _compute_a_out_v5)。
export W_P_V5="${W_P_V5:-1.1}"                   # parse
export W_COV_V5="${W_COV_V5:-0.55}"             # coverage (FOARL 让位给 capacity, 从 3.5 降到 0.55)
export W_CONS_V5="${W_CONS_V5:-3.3}"           # constraint = CVRP 容量满足率 (FOARL 主导, 从 1.0 升到 3.3)
export W_F_V5="${W_F_V5:-0.55}"                 # format

# 输出目录带 instruct + FOARL 权重(fw) 标识 + 超参标注 → 与 thinking v6 / 默认权重 v6 互不覆盖, 避免误 resume
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$_D/output_v6_instruct_fw_lr${LR}_ep${EPOCHS}_pa${PROC_ALPHA_V6}_nt${NUM_TRAIN}}"

echo "[v6-instruct] BASE_MODEL_TYPE=$BASE_MODEL_TYPE REWARD_SCHEME=$REWARD_SCHEME LR=$LR EPOCHS=$EPOCHS PROC_ALPHA_V6=$PROC_ALPHA_V6 NUM_TRAIN=$NUM_TRAIN"
echo "[v6-instruct] A_feas(FOARL对齐) parse=$W_P_V5 cov=$W_COV_V5 cap/cons=$W_CONS_V5 format=$W_F_V5 (总量 $(awk "BEGIN{print $W_P_V5+$W_COV_V5+$W_CONS_V5+$W_F_V5}"))"
echo "[v6-instruct] OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"

# run_grpo_cvrp20_v6.sh: 兜底设 REWARD_SCHEME=v6 + 沿用默认 vLLM 端口 8004, 再 exec run_grpo_cvrp20_v5.sh
exec bash "$_D/run_grpo_cvrp20_v6.sh"
