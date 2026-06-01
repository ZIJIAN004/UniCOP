#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=7
#SBATCH --exclude=canele1                 # 跳过易挂节点 canele1, SLURM 自动挑空闲节点
#SBATCH --no-requeue                      # canele1 易挂; 故障直接 FAIL 不重排, 避免日志截断/白跑
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_mask_%j.log

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
bash run_grpo_cvrp20_v5_mask.sh
