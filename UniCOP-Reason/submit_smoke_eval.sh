#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/smoke_eval_%j.log

# evaluate.py 兼容性 smoke test (Qwen3-Thinking)
# 8 个层面: tokenizer.decode / chat_template / generate / 长度 / 结构 / token 残留 / parse / 采样
# 用法: sbatch UniCOP-Reason/submit_smoke_eval.sh

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
echo "  evaluate.py smoke — Qwen3-Thinking 兼容性 (HF backend)"
echo "  BASE_MODEL = $BASE_MODEL"
echo "  GEN_*      = T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K"
echo "============================================================"

cd UniCOP-Reason
python smoke_test_eval.py \
    --model "$BASE_MODEL" \
    --problem cvrp \
    --problem_size 20 \
    --max_new_tokens 4096 \
    --temperature "${GEN_TEMPERATURE:-0.6}"
