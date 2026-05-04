#!/bin/bash
#SBATCH --qos express
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/sft_template_%j.log

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp

source /homes/zhuoyi/.bashrc
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill
bash run_sft_template_cvrp20.sh
