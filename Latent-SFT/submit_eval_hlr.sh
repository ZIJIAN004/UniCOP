#!/bin/bash
#SBATCH --qos express
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/eval_hlr_%j.log

# HLR vs Baseline 对比 eval (1 GPU, ~30-60 min for 100 实例)
#
# 流程 (eval_hlr_compare.py 内部):
#   Step 1: merge HLR LoRA → baseline_merged/ (有缓存就跳过)
#   Step 2: baseline (local backend, batch=4)
#   Step 3: HLR (hlr backend, batch=1)
#   Step 4: 读 JSON 对比, 输出 compare.json + 控制台对比表
#
# 用法:
#   HLR_CHECKPOINT=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_qwen3_thinking/checkpoint-final \
#       sbatch Latent-SFT/submit_eval_hlr.sh
#
#   # 可选环境变量:
#   #   HLR_BASE_MODEL    覆盖基座模型 (不传则从 adapter_config.json 读)
#   #   EVAL_PROBLEM      默认 cvrp
#   #   EVAL_PROBLEM_SIZE 默认 20
#   #   EVAL_NUM_TEST     默认 100
#   #   EVAL_MAX_LEN      默认 4096
#   #   EVAL_TEMPERATURE  默认 0.0 (贪心)

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
# 缓解高显存压力下的 allocator cache flush 抖动 (与训练侧 8442 诊断一致, 推理同样受益).
# 纯 CUDA allocator 行为, 不改任何计算语义.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source paths.sh

HLR_CHECKPOINT="${HLR_CHECKPOINT:?必须设 HLR_CHECKPOINT 环境变量}"
EVAL_PROBLEM="${EVAL_PROBLEM:-cvrp}"
EVAL_PROBLEM_SIZE="${EVAL_PROBLEM_SIZE:-20}"
EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"

SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}
trap 'notify "❌ HLR compare eval 失败 (line $LINENO)"' ERR

echo "============================================================"
echo "  HLR vs Baseline 对比 eval"
echo "============================================================"
echo "  HOST              = $HOST_ID"
echo "  HLR_CHECKPOINT    = $HLR_CHECKPOINT"
echo "  HLR_BASE_MODEL    = ${HLR_BASE_MODEL:-(auto from adapter_config.json)}"
echo "  EVAL              = $EVAL_PROBLEM-$EVAL_PROBLEM_SIZE  n=$EVAL_NUM_TEST  "
echo "                       max_len=$EVAL_MAX_LEN  temp=$EVAL_TEMPERATURE"
echo "============================================================"

# GPU 占用诊断
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader \
    | sed 's/^/    /'
_min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -n "$_min_free" ] && [ "$_min_free" -lt 20000 ]; then
    echo "❌ GPU free ${_min_free} MiB < 20 GiB, 别人在占, 自动 exclude resubmit"
    _bad_node="$SLURM_NODELIST"
    _self_path="$(realpath "$0")"
    sbatch --exclude="$_bad_node" --export=ALL "$_self_path"
    exit 0
fi
echo "============================================================"

# 拼接 hlr_base_model 透传 (eval_hlr_compare.py 不传时 auto)
EXTRA_FLAGS=""
if [ -n "${HLR_BASE_MODEL:-}" ]; then
    EXTRA_FLAGS="--hlr_base_model $HLR_BASE_MODEL"
fi

python Latent-SFT/eval_hlr_compare.py \
    --hlr_checkpoint "$HLR_CHECKPOINT" \
    --problem "$EVAL_PROBLEM" \
    --problem_size "$EVAL_PROBLEM_SIZE" \
    --num_test "$EVAL_NUM_TEST" \
    --max_completion_length "$EVAL_MAX_LEN" \
    --temperature "$EVAL_TEMPERATURE" \
    $EXTRA_FLAGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    notify "✅ HLR compare eval 完成" "checkpoint=$HLR_CHECKPOINT"
    echo "✅ 完成. 对比结果见 $HLR_CHECKPOINT/compare_eval/compare.json"
else
    notify "❌ HLR compare eval 失败 (exit $EXIT_CODE)"
    echo "❌ 失败 exit $EXIT_CODE"
fi
exit $EXIT_CODE
