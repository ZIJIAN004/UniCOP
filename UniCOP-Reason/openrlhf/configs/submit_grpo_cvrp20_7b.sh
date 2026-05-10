#!/bin/bash
# zhuoyi 主机 sbatch 提交脚本: CVRP n=100 · 7B · GRPO · LoRA · 5 卡
#
# 用法:
#   sbatch openrlhf/configs/submit_grpo_cvrp20_7b.sh
#
# QOS=long: 5 天上限,优先级 5 (与现有 SFT 任务相同档位)
# 5 张 GPU: 1 vLLM (含 reward server colocate) + 4 ZeRO-3 训练
#
#SBATCH --qos=long
#SBATCH --gpus=5
#SBATCH --job-name=grpo_cvrp20
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/openrlhf/logs/grpo_cvrp20_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/openrlhf/logs/grpo_cvrp20_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

mkdir -p /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/openrlhf/logs

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason
bash openrlhf/configs/train_grpo_cvrp20_7b.sh
