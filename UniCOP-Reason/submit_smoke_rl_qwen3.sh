#!/bin/bash
#SBATCH --qos express
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/smoke_rl_qwen3_%j.log

# RL/GRPO 兼容性 smoke test (Qwen3-4B-Thinking)
# 覆盖 tokenizer / chat_template / config / GRPOConfig / _strip_chat_specials
# / reward_fn / </think> 边界 / end-to-end prompt build 共 10 个 section
#
# 用法: sbatch UniCOP-Reason/submit_smoke_rl_qwen3.sh
# 1 GPU express, 1-2 分钟结束

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
echo "  RL smoke — Qwen3-4B-Thinking 兼容性 (GRPO 训练栈)"
echo "  BASE_MODEL = $BASE_MODEL"
echo "  GEN_*      = T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K"
echo "  VLLM_REASONING_FLAGS = $VLLM_REASONING_FLAGS"
echo "============================================================"

cd UniCOP-Reason
python smoke_test_rl_compat.py \
    --model "$BASE_MODEL" \
    --model_type "$BASE_MODEL_TYPE"
