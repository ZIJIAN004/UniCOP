#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/sft_hybrid_%j.log

_SLURM_CUDA="${CUDA_VISIBLE_DEVICES:-}"

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA"

cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill
bash run_sft_hybrid_cvrp20.sh
