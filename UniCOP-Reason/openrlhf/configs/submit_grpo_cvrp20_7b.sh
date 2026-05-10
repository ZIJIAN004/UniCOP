#!/bin/bash
# zhuoyi 主机 sbatch 提交脚本: CVRP n=20 · 7B · GRPO · LoRA · 5 卡
#
# 用法:
#   sbatch openrlhf/configs/submit_grpo_cvrp20_7b.sh
#
# QOS=large: 8 天上限,优先级 0,单 job ≤24 GPU。
# 选 large 不选 long: 实测 long 单 job ≤2 GPU (QOSMaxGRESPerJob),5 卡被拒
# 选 large 不选 normal: normal 单 job ≤4 GPU,本任务需 1 vLLM + 4 训练 = 5 卡
# (训练侧 batch_size=32 与 train_gpus=4 绑定,grad_accum=8,降卡数会破坏整除关系)
# 风险: 优先级 0,可能被高优先级作业抢占 (terminated and requeued)
#
#SBATCH --qos=large
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
