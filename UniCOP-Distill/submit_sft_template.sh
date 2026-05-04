#!/bin/bash
#SBATCH --qos express
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/sft_template_%j.log

source ~/.bashrc
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill
bash run_sft_template_cvrp20.sh
