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
export PROC_ALPHA_V6="${PROC_ALPHA_V6:-200}"    # v6 PRM 段注入权重 (扫参主轴); train.py 用此 env 覆盖 config

# 输出目录带 instruct 标识 + 超参标注 → 与 thinking v6 (output_v6_*) 互不覆盖, 也避免误 resume 旧超参 ckpt
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$_D/output_v6_instruct_lr${LR}_ep${EPOCHS}_pa${PROC_ALPHA_V6}_nt${NUM_TRAIN}}"

echo "[v6-instruct] BASE_MODEL_TYPE=$BASE_MODEL_TYPE REWARD_SCHEME=$REWARD_SCHEME LR=$LR EPOCHS=$EPOCHS PROC_ALPHA_V6=$PROC_ALPHA_V6 NUM_TRAIN=$NUM_TRAIN"
echo "[v6-instruct] OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"

# run_grpo_cvrp20_v6.sh: 兜底设 REWARD_SCHEME=v6 + 沿用默认 vLLM 端口 8004, 再 exec run_grpo_cvrp20_v5.sh
exec bash "$_D/run_grpo_cvrp20_v6.sh"
