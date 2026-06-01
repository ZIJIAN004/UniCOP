#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=7
#SBATCH --exclude=canele1                 # 易挂节点(用户称 cancel1)，排除避免训练中途挂掉
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_%j.log

# GPU: 1 vLLM + 6 卡训练 (run 脚本默认动态挑卡)。实测吞吐随卡数提升 (per-completion: 6卡5.0s<4卡6.8s<2卡11.7s),
# 卡越多总训练越快; 仅当 SU/卡时受限时才降卡数 (2 卡最省算力但墙钟慢 2.34×)。
export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
bash run_grpo_cvrp20_v5.sh
