#!/bin/bash
# HLR β Sweep 启动器: 并行提交 3 个 submit_train_eval_hlr.sh
# (β=1.0 已单独跑过, 不重复)
#
# 用法 (在 zhuoyi 登录节点直接运行):
#   HLR_MODEL=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_instruct_template_cvrp20/final_model/ \
#   HLR_DATA=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/data/profiled_instruct_trained_cvrp20_10k.jsonl \
#       bash Latent-SFT/submit_beta_sweep.sh
#
# 输出: output_hlr_beta{0.5,1.5,2.0}/ 各含 checkpoint-final/compare_eval/compare.json

set -euo pipefail

: "${HLR_MODEL:?必须设 HLR_MODEL}"
: "${HLR_DATA:?必须设 HLR_DATA}"

BETAS="0.5 1.5 2.0"
HLR_ALPHA=1.0
HLR_GAMMA=1.0
HLR_EPOCHS="${HLR_EPOCHS:-1}"
BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"

EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"
EVAL_PROBLEM="${EVAL_PROBLEM:-cvrp}"
EVAL_PROBLEM_SIZE="${EVAL_PROBLEM_SIZE:-20}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "============================================================"
echo "  HLR β Sweep: 并行提交 3 个 job"
echo "============================================================"
echo "  MODEL   = $HLR_MODEL"
echo "  DATA    = $HLR_DATA"
echo "  α=${HLR_ALPHA}  β∈{${BETAS}}  γ=${HLR_GAMMA}"
echo "  eval    = ${EVAL_PROBLEM}-${EVAL_PROBLEM_SIZE} n=${EVAL_NUM_TEST} T=${EVAL_TEMPERATURE}"
echo "  seed    = 9999 (evaluate.py, 测试集固定)"
echo "============================================================"

JOB_IDS=""
for BETA in $BETAS; do
    OUTPUT_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_beta${BETA}"

    EXPORT_STR="ALL"
    EXPORT_STR="${EXPORT_STR},HLR_MODEL=${HLR_MODEL}"
    EXPORT_STR="${EXPORT_STR},HLR_DATA=${HLR_DATA}"
    EXPORT_STR="${EXPORT_STR},HLR_ALPHA=${HLR_ALPHA}"
    EXPORT_STR="${EXPORT_STR},HLR_BETA=${BETA}"
    EXPORT_STR="${EXPORT_STR},HLR_GAMMA=${HLR_GAMMA}"
    EXPORT_STR="${EXPORT_STR},HLR_OUTPUT_DIR=${OUTPUT_DIR}"
    EXPORT_STR="${EXPORT_STR},HLR_EPOCHS=${HLR_EPOCHS}"
    EXPORT_STR="${EXPORT_STR},EVAL_NUM_TEST=${EVAL_NUM_TEST}"
    EXPORT_STR="${EXPORT_STR},EVAL_PROBLEM=${EVAL_PROBLEM}"
    EXPORT_STR="${EXPORT_STR},EVAL_PROBLEM_SIZE=${EVAL_PROBLEM_SIZE}"
    EXPORT_STR="${EXPORT_STR},EVAL_MAX_LEN=${EVAL_MAX_LEN}"
    EXPORT_STR="${EXPORT_STR},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
    EXPORT_STR="${EXPORT_STR},EVAL_TEMPERATURE=${EVAL_TEMPERATURE}"
    EXPORT_STR="${EXPORT_STR},BASE_MODEL_TYPE=${BASE_MODEL_TYPE}"

    JOB_ID=$(sbatch --parsable --export="$EXPORT_STR" "$SCRIPT_DIR/submit_train_eval_hlr.sh")

    JOB_IDS="$JOB_IDS $JOB_ID"
    echo "  ✓ β=${BETA}  →  job ${JOB_ID}  →  ${OUTPUT_DIR}"
done

echo ""
echo "  3 个 job 并行运行, 每个: profile(缓存跳过) → train(4 GPU) → verify → eval(merge→baseline→HLR→compare.json)"
echo "  完成后运行汇总:"
echo "    bash Latent-SFT/show_beta_summary.sh"
echo "============================================================"
