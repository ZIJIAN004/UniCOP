#!/bin/bash
# submit_beta_sweep.sh — HLR β sweep 启动器: 并行训练 → 全部结束后自动 eval
#
#   流程 (参照 sweep_proc_alpha_v6.sh):
#     1. 并行提交 3 个 submit_train_hlr.sh (β=0.5/1.5/2.0), 各 4 GPU
#     2. 提交 1 个 submit_sweep_eval_hlr.sh (4 GPU, 并行 eval),
#        --dependency=afterany 绑定全部 3 个训练 job
#
#   输出: output_hlr_beta{0.5,1.5,2.0}/checkpoint-final/compare_eval/compare.json
#
#   用法 (集群登录节点, git pull 之后):
#       HLR_MODEL=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_instruct_template_cvrp20/final_model/ \
#       HLR_DATA=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/data/profiled_instruct_trained_cvrp20_10k.jsonl \
#           bash Latent-SFT/submit_beta_sweep.sh
#
#   可覆盖:
#       BETAS="0.1 0.5 2.0"  EVAL_NUM_TEST=100  EVAL_PROBLEM=cvrp

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."

: "${HLR_MODEL:?必须设 HLR_MODEL}"
: "${HLR_DATA:?必须设 HLR_DATA}"

LR="${LR:-2e-5}"
LR_LR="${LR_LR:-5e-5}"
HLR_EPOCHS="${HLR_EPOCHS:-1}"
HLR_ALPHA="${HLR_ALPHA:-1.0}"
HLR_GAMMA="${HLR_GAMMA:-1.0}"
BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"

BETAS="${BETAS:-0.5 1.5 2.0}"
EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"
EVAL_PROBLEM="${EVAL_PROBLEM:-cvrp}"
EVAL_PROBLEM_SIZE="${EVAL_PROBLEM_SIZE:-20}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"

echo "############## HLR β Sweep (train → eval 流水线, 参照 sweep_proc_alpha_v6.sh) ##############"
echo "  MODEL=$HLR_MODEL"
echo "  DATA=$HLR_DATA"
echo "  α=${HLR_ALPHA}  β∈{${BETAS}}  γ=${HLR_GAMMA}"
echo "  EPOCHS=${HLR_EPOCHS}  lr=${LR}  lr_lr=${LR_LR}"
echo "  Eval=${EVAL_PROBLEM}-${EVAL_PROBLEM_SIZE}  n=${EVAL_NUM_TEST}  T=${EVAL_TEMPERATURE}"
echo "  3 个训练 job 并行 → afterany 自动 eval (${EVAL_GPUS})"
echo

TRAIN_SCRIPT="Latent-SFT/submit_train_hlr.sh"
EVAL_SCRIPT="Latent-SFT/submit_sweep_eval_hlr.sh"

# ══════════════════════════════════════════════════════════════════
# [1] 并行提交训练 job
# ══════════════════════════════════════════════════════════════════
train_ids=()
dirs=""

for BETA in $BETAS; do
    OUTPUT_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_beta${BETA}"

    # sbatch --export 一行: KEY=VALUE 逗号分隔
    EXPORT_STR="ALL"
    EXPORT_STR="${EXPORT_STR},HLR_MODEL=${HLR_MODEL}"
    EXPORT_STR="${EXPORT_STR},HLR_DATA=${HLR_DATA}"
    EXPORT_STR="${EXPORT_STR},HLR_ALPHA=${HLR_ALPHA}"
    EXPORT_STR="${EXPORT_STR},HLR_BETA=${BETA}"
    EXPORT_STR="${EXPORT_STR},HLR_GAMMA=${HLR_GAMMA}"
    EXPORT_STR="${EXPORT_STR},HLR_OUTPUT_DIR=${OUTPUT_DIR}"
    EXPORT_STR="${EXPORT_STR},HLR_EPOCHS=${HLR_EPOCHS}"
    EXPORT_STR="${EXPORT_STR},BASE_MODEL_TYPE=${BASE_MODEL_TYPE}"

    echo ">>> 提交训练 β=${BETA} → ${OUTPUT_DIR}"
    out=$(sbatch --parsable \
        --job-name="zijia_hlr_b${BETA}" \
        --export="$EXPORT_STR" \
        "$TRAIN_SCRIPT")
    echo "    job ${out}"
    train_ids+=("$out")
    dirs="$dirs $OUTPUT_DIR"
done

# ══════════════════════════════════════════════════════════════════
# [2] 提交 eval job (afterany 依赖全部训练)
# ══════════════════════════════════════════════════════════════════
dep_list=$(IFS=:; echo "${train_ids[*]}")

echo ""
echo ">>> 提交 sweep eval (DIRS=${dirs# }, 等训练 job ${dep_list} 结束)"

EXPORT_STR="ALL"
EXPORT_STR="${EXPORT_STR},DIRS=${dirs# }"
EXPORT_STR="${EXPORT_STR},EVAL_NUM_TEST=${EVAL_NUM_TEST}"
EXPORT_STR="${EXPORT_STR},EVAL_PROBLEM=${EVAL_PROBLEM}"
EXPORT_STR="${EXPORT_STR},EVAL_PROBLEM_SIZE=${EVAL_PROBLEM_SIZE}"
EXPORT_STR="${EXPORT_STR},EVAL_MAX_LEN=${EVAL_MAX_LEN}"
EXPORT_STR="${EXPORT_STR},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
EXPORT_STR="${EXPORT_STR},EVAL_TEMPERATURE=${EVAL_TEMPERATURE}"
EXPORT_STR="${EXPORT_STR},EVAL_GPUS=${EVAL_GPUS}"

out=$(sbatch --parsable \
    --dependency="afterany:${dep_list}" \
    --job-name="zijia_hlr_sweep_eval" \
    --export="$EXPORT_STR" \
    "$EVAL_SCRIPT")
echo "    job ${out}"

echo ""
echo "全部已提交: 3 个训练 job 并行 + 1 个 eval job 链式绑定。查看队列: squeue -u \$USER"
echo "结果将落在: output_hlr_beta{0.5,1.5,2.0}/checkpoint-final/compare_eval/compare.json"
squeue -u "${USER:-$(whoami)}" 2>/dev/null || true
