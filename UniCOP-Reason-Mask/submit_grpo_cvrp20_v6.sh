#!/bin/bash
# submit_grpo_cvrp20_v6.sh
# ── 实验: 纯净 v6 (PRM 批级截尾标准化 + sigmoid, proc_alpha_v6=200) + 温和超参 ──
#   目的: 验证 v6 的 PRM 信号变换 (相对 v5 的 base+tanh) 能否压低 CVRP20 optimality gap。
#   配置:
#     - reward_scheme = v6        (由 run_grpo_cvrp20_v6.sh 设)
#     - use_mask      = False     (纯净, 不传 --use_mask, 跟纯净 v5 同口径)
#     - 基座          = qwen3_thinking  (Qwen3-4B SFT 产物)
#     - LR            = 2e-5      (对齐 v5; 1e-6 训练不足; train.py 经 env 覆盖)
#     - epochs        = 1         (单 epoch)
#     - 其余 Mask 超参不动: kl_coef=0.0 / clip 0.20-0.28 / num_generations=8 / num_train=1000
#     - 输出隔离 output_v6, vLLM 端口沿用 v5 默认 8004 (独占节点, 不会与 v5 撞)
#   ⚠️ 注意: 本次同时变了"信号(v6)"和"温和超参(LR/epoch)"两个轴, 若 gap 改善
#            无法单独归因; 如需干净归因, 另跑一组"v6 + Mask 默认超参(LR=1e-5/epoch=2)"。
#
#   提交: sbatch submit_grpo_cvrp20_v6.sh

#SBATCH --qos=large
#SBATCH --gpus=7
#SBATCH --job-name=zijia_cvrp20_v6
#SBATCH --comment="zijianliu, pure v6 PRM sigmoid, do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v6_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v6_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 本实验覆盖项 ───────────────────────────────────────────────
export BASE_MODEL_TYPE=qwen3_thinking   # Qwen3-4B SFT 产物作为 RL 起点
export LR="${LR:-2e-5}"                   # 对齐 v5 (1e-6 训练不足; train.py env 覆盖); 可被 sbatch --export 覆盖
export EPOCHS="${EPOCHS:-1}"              # 单 epoch; 可被 sbatch --export 覆盖
export SAVE_STEPS="${SAVE_STEPS:-20}"     # 每 20 step 存档 (~41 步的短跑, 存 step20/40 + final)
export NUM_TRAIN="${NUM_TRAIN:-1000}"     # 一个 epoch 的训练样本数 (run_grpo_v5 经 env 读); 扫参可传 500 提速
export PROC_ALPHA_V6="${PROC_ALPHA_V6:-200}"   # v6 PRM 段注入权重 (扫参主轴; train.py:180 用此 env 覆盖 config)
# 输出目录带超参标注 → 不同 lr/epoch/proc_alpha/num_train 互不覆盖, 也避免误 resume 旧超参的 checkpoint
export OUTPUT_DIR_BASE="output_v6_lr${LR}_ep${EPOCHS}_pa${PROC_ALPHA_V6}_nt${NUM_TRAIN}"   # 如 output_v6_lr2e-5_ep1_pa400_nt500
# REWARD_SCHEME=v6 由 run_grpo_cvrp20_v6.sh 设

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

# ── GPU 占用预检: 分到的卡若被别人占着 → 把本节点 exclude 后重投, 本 job 退出 ──
#    无默认 exclude (用户决定: 不预排任何节点), 完全靠预检动态排除被占的坏节点。
export SUBMIT_SCRIPT="$(pwd)/submit_grpo_cvrp20_v6.sh"
export BASE_EXCLUDE=""
source "$(pwd)/preflight_gpu.sh"
preflight_gpu_or_resubmit

bash run_grpo_cvrp20_v6.sh
