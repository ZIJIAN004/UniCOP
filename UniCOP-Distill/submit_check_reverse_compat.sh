#!/bin/bash
#SBATCH --qos=express
#SBATCH --gpus=1
#SBATCH --job-name=zijia_check_reverse
#SBATCH --comment="zijianliu, do not cancel"
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/slurm_%x_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/slurm_%x_%j.err
#SBATCH --no-requeue
#SBATCH --open-mode=append
#
# reverse 数据兼容性自检（CPU 级，~2min；express 最高优先级，秒级排队）。
# 用 trainer 真实 load_sft_dataset + 真 tokenizer 跑一遍，报告最终样本数。
# 用法: sbatch UniCOP-Distill/submit_check_reverse_compat.sh
#   然后: cat UniCOP-Distill/slurm_zijia_check_reverse_<jobid>.log
# 判读: "最终训练样本数: 10000" + "✓ 完全兼容" → 可以 sbatch 训练脚本。

export HOME=/homes/zhuoyi
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP

MODEL=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_template_cvrp20/final_model
DATA=UniCOP-Distill/data/chains_reverse_cvrp20_10k.jsonl

echo "============================================================"
echo "  reverse 数据兼容性自检  $(date)"
echo "  MODEL = $MODEL"
echo "  DATA  = $DATA"
echo "============================================================"

python UniCOP-Distill/check_reverse_compat.py \
    --model "$MODEL" \
    --data  "$DATA" \
    --expect 10000

echo ">>> 自检结束 $(date)"
