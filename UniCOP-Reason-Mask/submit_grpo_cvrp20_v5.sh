#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=3
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_%j.log

# GPU 拓扑: sweep 实测甜点位 = 1 vLLM + 2 卡训练 (无 NVLink 下梯度同步中转最少, 每-completion 最低)。
export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── GPU 甜点位: 1 vLLM (GPU 2) + 2 卡训练 (GPU 0,1), B=4, num_gen=8, ZeRO-2 (整除 4×2=8 % 8 ✓) ──
export VLLM_GPU=2
export TRAIN_GPUS_CSV=0,1
export TRAIN_PROC=2
export PER_DEVICE_BATCH=4
export ZERO_STAGE=2

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
bash run_grpo_cvrp20_v5.sh
