#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=5
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/grpo_cvrp20_%j.log

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason
bash run_grpo_cvrp20.sh
