#!/bin/bash
# submit_beta_sweep.sh — HLR β sweep 启动器 (参照 sweep_proc_alpha_v6.sh)
#
# 用法:
#   HLR_MODEL=/path/to/model/ \
#   HLR_DATA=/path/to/profiled.jsonl \
#       bash Latent-SFT/submit_beta_sweep.sh
#
# 可覆盖: BETAS EVAL_NUM_TEST ...

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

: "${HLR_MODEL:?必须设 HLR_MODEL}"
: "${HLR_DATA:?必须设 HLR_DATA}"

BETAS="${BETAS:-0.5 1.5 2.0}"
HLR_ALPHA="${HLR_ALPHA:-1.0}"
HLR_GAMMA="${HLR_GAMMA:-1.0}"
HLR_EPOCHS="${HLR_EPOCHS:-1}"
BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"

EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"
EVAL_PROBLEM="${EVAL_PROBLEM:-cvrp}"
EVAL_PROBLEM_SIZE="${EVAL_PROBLEM_SIZE:-20}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"

echo "############## HLR β Sweep ##############"
echo "  MODEL=$HLR_MODEL"
echo "  DATA=$HLR_DATA"
echo "  α=${HLR_ALPHA}  β∈{${BETAS}}  γ=${HLR_GAMMA}  epochs=${HLR_EPOCHS}"
echo "  Eval=${EVAL_PROBLEM}-${EVAL_PROBLEM_SIZE} n=${EVAL_NUM_TEST} T=${EVAL_TEMPERATURE}"
echo

# ── [1] 并行提交训练 (参照 sweep_proc_alpha_v6.sh:17-61) ──
train_ids=()
dirs=""

for BETA in $BETAS; do
    OUTPUT_DIR="Latent-SFT/output_hlr_beta${BETA}"

    echo ">>> 提交训练 β=${BETA} → ${OUTPUT_DIR}"

    out=$(sbatch --parsable \
        --job-name="zijia_hlr_b${BETA}" \
        --export="ALL,HLR_ALPHA=${HLR_ALPHA},HLR_BETA=${BETA},HLR_GAMMA=${HLR_GAMMA},HLR_OUTPUT_DIR=${OUTPUT_DIR},HLR_EPOCHS=${HLR_EPOCHS},BASE_MODEL_TYPE=${BASE_MODEL_TYPE}" \
        submit_train_hlr.sh)

    echo "    job ${out}"
    train_ids+=("$out")
    dirs="$dirs $OUTPUT_DIR"
done

# ── [2] 提交 eval (afterany, 参照 sweep_proc_alpha_v6.sh:63-72) ──
dep_list=$(IFS=:; echo "${train_ids[*]}")

echo ""
echo ">>> 提交 sweep eval (DIRS=${dirs# }, afterany ${dep_list})"

out=$(sbatch --parsable \
    --dependency="afterany:${dep_list}" \
    --job-name="zijia_hlr_sweep_eval" \
    --export="ALL,DIRS=${dirs# },EVAL_NUM_TEST=${EVAL_NUM_TEST},EVAL_PROBLEM=${EVAL_PROBLEM},EVAL_PROBLEM_SIZE=${EVAL_PROBLEM_SIZE},EVAL_MAX_LEN=${EVAL_MAX_LEN},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},EVAL_TEMPERATURE=${EVAL_TEMPERATURE},EVAL_GPUS=${EVAL_GPUS}" \
    submit_sweep_eval_hlr.sh)

echo "    job ${out}"

echo ""
echo "全部已提交。查看: squeue -u \$USER"
echo "结果: output_hlr_beta{0.5,1.5,2.0}/checkpoint-final/compare_eval/compare.json"
squeue -u "${USER:-$(whoami)}" 2>/dev/null || true
