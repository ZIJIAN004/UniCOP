#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/smoke_qwen3_%j.log

# Smoke test for Qwen3-4B-Thinking-2507 migration
# - Offline 11 sections (tokenizer / chat_template / labels mask / 思维链是否进 loss / ...)
# - Online section 12 (加载真实模型跑 1 step forward+backward + 分段 loss 对照)
# 用法: sbatch UniCOP-Distill/submit_smoke_qwen3.sh

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
echo "  Smoke test: Qwen3-4B-Thinking migration"
echo "  BASE_MODEL = $BASE_MODEL"
echo "  GEN_*      = T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K"
echo "  VLLM_REASONING_FLAGS = $VLLM_REASONING_FLAGS"
echo "  STAGE1_KEEP_THINK    = $STAGE1_KEEP_THINK"
echo "============================================================"

python UniCOP-Distill/smoke_test_base_model.py \
    --online \
    --data UniCOP-Distill/data/chains_template_cvrp20.jsonl \
    --online_max_len 8192
