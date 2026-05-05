#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/diagnose_%j.log

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill
source ../paths.sh
python diagnose_memory.py
