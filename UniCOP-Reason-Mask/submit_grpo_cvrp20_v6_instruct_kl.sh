#!/bin/bash
# submit_grpo_cvrp20_v6_instruct_kl.sh
# ── zhuoyi(astar-zhuoyi, SLURM) 专用: instruct 基座 v6 GRPO + KL anchor sweep ──────
#   目的: instruct 基座 (Qwen3-4B-Instruct SFT 产物) 在 DAPO 无 KL (kl_coef=0) 下后期
#         策略漂出 SFT 流形 → 内容+长度双崩 (2026-06-23 v6_instruct log: step28 健康 →
#         step37 全截断/parse=0, clip_high_hit_rate 全程 0 拦不住慢漂)。补小 KL 锚回 SFT 止崩。
#   受控: 除 KL_COEF 外, 其余一律沿用 run_grpo_cvrp20_v6_instruct.sh 默认 (与之前崩掉那次同口径):
#         BASE=qwen3_instruct / LR=2e-5 / EPOCHS=1 / PROC_ALPHA_V6=1000 / NUM_TRAIN=1000 /
#         MAX_COMPLETION=4096 / A_feas FOARL 权重 / clip 0.20-0.28 / num_generations=8。
#
#   提交 (两个 KL 值, 复用本脚本, --export 传值):
#       sbatch --export=ALL,KL_COEF=0.005 submit_grpo_cvrp20_v6_instruct_kl.sh
#       sbatch --export=ALL,KL_COEF=0.01  submit_grpo_cvrp20_v6_instruct_kl.sh
#   (经验值 1e-3~5e-3: DeepSeek-R1=1e-3, HH-RLHF=1e-2; 0.005/0.01 各取下界稳态与上界兜顶。)
#
#   ⚠️ 提交前先在登录节点跑 python verify_kl.py 确认 KL 链路接线 (秒级, 无 GPU);
#      跑起来后用 python verify_kl.py --log <本 job 的 grpo_*_%j.log> 做运行时确认。

#SBATCH --qos=large
#SBATCH --gpus=7
#SBATCH --job-name=zijia_cvrp20_v6_instruct_kl
#SBATCH --comment="zijianliu, v6 instruct KL anchor sweep, do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_v6_instruct_kl_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_v6_instruct_kl_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 本实验唯一变量: KL anchor 系数 (train.py 经 env 覆盖 config.kl_coef → GRPOConfig.beta) ──
#   默认 0.005; sbatch --export=ALL,KL_COEF=... 覆盖。0 = 复现崩溃 (对照基线)。
export KL_COEF="${KL_COEF:-0.005}"

# ── instruct 受控超参 (与 run_grpo_cvrp20_v6_instruct.sh 默认一致, 显式写出便于 OUTPUT_DIR 命名) ──
export BASE_MODEL_TYPE=qwen3_instruct
export LR="${LR:-2e-5}"
export EPOCHS="${EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-20}"
export NUM_TRAIN="${NUM_TRAIN:-1000}"
export PROC_ALPHA_V6="${PROC_ALPHA_V6:-1000}"

# 输出目录带 kl 标识 → 两个 KL 值 + 与之前崩掉那次 (无 kl 标识) 物理隔离, 避免误 resume
export OUTPUT_DIR_BASE="/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/output_v6_instruct_fw_kl${KL_COEF}_lr${LR}_ep${EPOCHS}_pa${PROC_ALPHA_V6}_nt${NUM_TRAIN}"

# 全流程 train→merge→BO1 eval (launcher 已支持)。RUN_PREFIX 带 kl 标识 → 两个 job 的
# eval json/log 不撞名 (launcher 已改为 ${RUN_PREFIX:-...} 可覆盖)。merge 走 merge_lora.py
# 落盘 $OUT_DIR/merge.log, 不再"失败无 log"。
export RUN_EVAL=1
export RUN_PREFIX="v6_instruct_kl${KL_COEF}_"

echo "[submit-kl] KL_COEF=$KL_COEF  BASE=$BASE_MODEL_TYPE  LR=$LR  EPOCHS=$EPOCHS  PROC_ALPHA_V6=$PROC_ALPHA_V6  NUM_TRAIN=$NUM_TRAIN"
echo "[submit-kl] OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE  RUN_EVAL=$RUN_EVAL  RUN_PREFIX=$RUN_PREFIX"

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

# ── GPU 占用预检: 分到的卡若被占 → exclude 本节点重投, 本 job 退出 (无静态 exclude, 全靠预检) ──
export SUBMIT_SCRIPT="$(pwd)/submit_grpo_cvrp20_v6_instruct_kl.sh"
export BASE_EXCLUDE=""
source "$(pwd)/preflight_gpu.sh"
preflight_gpu_or_resubmit

# run_grpo_cvrp20_v6_instruct.sh: 设 instruct 基座 + FOARL 权重 + MAX_COMPLETION=4096, 再 → v6 → v5。
# 非 TTY (sbatch) 下其 tmux 警告自动跳过; GPU 走 run_v5 非 zhihan 默认 (vLLM=6 / 训练=0-5 / 7卡);
# NCCL 走 paths.sh HOST_ID=astar-zhuoyi 自动禁 P2P/SHM。
bash run_grpo_cvrp20_v6_instruct.sh
