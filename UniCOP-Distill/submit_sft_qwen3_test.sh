#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/sft_qwen3_test_%j.log

# Stage 2 SFT — Qwen3-4B-Thinking 测试跑: 单卡 / 1 epoch / chains_template_cvrp20
# 用于在跑 full 之前确认 pipeline 端到端通顺(数据加载 / template / labels mask /
# bf16 forward+backward / save_pretrained 全跑通)。
# 用法: sbatch UniCOP-Distill/submit_sft_qwen3_test.sh

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE=qwen3_thinking
source paths.sh

echo "============================================================"
echo "  Stage 2 SFT [TEST] — Qwen3-4B-Thinking (单卡 sanity)"
echo "  BASE_MODEL = $BASE_MODEL"
echo "  data       = chains_template_cvrp20.jsonl (filter cvrp / size 20)"
echo "  sanity 量  = 前 200 条 / 1 epoch  (~30-45 min)"
echo "  epochs=1  batch=2 grad_accum=4  lr=1e-4  zero=0  LoRA r=64/128"
echo "  (单卡测试不走 ZeRO; full sbatch 4 卡走 ZeRO-3)"
echo "============================================================"

python UniCOP-Distill/stage2_reasoning/train_sft_stage2.py \
    --model "$BASE_MODEL" \
    --data UniCOP-Distill/data/chains_template_cvrp20.jsonl \
    --filter_problems cvrp \
    --filter_sizes 20 \
    --max_samples 200 \
    --epochs 1 \
    --batch_size 2 --grad_accum 4 \
    --lr 1e-4 \
    --max_length 8192 \
    --lora_rank 64 --lora_alpha 128 \
    --gradient_checkpointing \
    --output_dir /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_test \
    --logging_steps 5 --save_steps 50
