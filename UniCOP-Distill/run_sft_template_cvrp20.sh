#!/bin/bash
# CVRP20 模板思维链 SFT + 评估
#
# 前提: data/chains_template_cvrp20.jsonl 已存在
#
# 使用方法：
#   bash run_sft_template_cvrp20.sh
#   nohup bash run_sft_template_cvrp20.sh > sft_template_cvrp20.log 2>&1 &

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$_SELF_DIR/sft_template_cvrp20_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1

source "$(dirname "$_SELF_DIR")/paths.sh"

# ── 配置 ──────────────────────────────────────────────────────────────────────

MODEL_PATH="$BASE_MODEL"
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=20
CHAINS_FILE="data/chains_template_${PROBLEM}${SIZE}.jsonl"
OUTPUT_DIR="output_sft_template_${PROBLEM}${SIZE}"

# SFT 配置（与 run_sft_cvrp20.sh 完全一致）
SFT_LR=2e-5
SFT_EPOCHS=3
SFT_LORA_RANK=64
SFT_LORA_ALPHA=128
SFT_MAX_LENGTH=4992
SFT_NUM_GPUS=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 工具函数 ──────────────────────────────────────────────────────────────────
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s "https://sctapi.ftqq.com/$SCKEY.send" \
        -d "title=$title" -d "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

get_free_gpus() {
    local threshold=${1:-500}
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$threshold" '$2 < t {print $1}'
}

wait_for_free_gpus() {
    local required=$1
    local max_wait=${2:-1800}
    local waited=0
    while true; do
        local free=($(get_free_gpus))
        if [ ${#free[@]} -ge "$required" ]; then
            echo "${free[@]:0:$required}"
            return 0
        fi
        echo "  空闲 GPU: ${#free[@]}/$required, 等待中... (${waited}s)" >&2
        sleep 30
        waited=$((waited + 30))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "ERROR: 等待 $required 张空闲 GPU 超时 (${max_wait}s)" >&2
            return 1
        fi
    done
}

trap 'notify "Template CVRP20 SFT 失败: line $LINENO"' ERR

cd "$DISTILL_DIR"

# ── 检查数据 ──────────────────────────────────────────────────────────────────
if [ ! -f "$CHAINS_FILE" ]; then
    echo "ERROR: 数据文件 $CHAINS_FILE 不存在"
    exit 1
fi
DATA_COUNT=$(grep -c '^{' "$CHAINS_FILE" 2>/dev/null || echo 0)

echo "============================================================"
echo "  Template CVRP20 SFT + 评估"
echo "  模型:   $MODEL_PATH"
echo "  数据:   $CHAINS_FILE ($DATA_COUNT 条)"
echo "  GPU:    $SFT_NUM_GPUS 张"
echo "  输出:   $OUTPUT_DIR"
echo "  时间:   $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 1: SFT 训练
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: SFT 训练（等待至少 ${SFT_NUM_GPUS} 张空闲 GPU）..."

SFT_GPU_LIST=($(wait_for_free_gpus $SFT_NUM_GPUS))
SFT_CUDA_DEVICES=$(IFS=,; echo "${SFT_GPU_LIST[*]}")
echo "  使用 GPU: $SFT_CUDA_DEVICES"

CUDA_VISIBLE_DEVICES=$SFT_CUDA_DEVICES accelerate launch \
    --num_processes $SFT_NUM_GPUS \
    --main_process_port 29600 \
    stage2_reasoning/train_sft_stage2.py \
    --model "$MODEL_PATH" \
    --data "$CHAINS_FILE" \
    --filter_problems $PROBLEM \
    --filter_sizes $SIZE \
    --lora_rank $SFT_LORA_RANK --lora_alpha $SFT_LORA_ALPHA \
    --max_length $SFT_MAX_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --zero_stage 3 \
    --gradient_checkpointing \
    --epochs $SFT_EPOCHS \
    --batch_size 2 \
    --grad_accum 4 \
    --lr $SFT_LR \
    --save_steps 500

notify "Step1 完成: Template SFT 训练"

# ══════════════════════════════════════════════════════════════════
# Step 2: 合并 LoRA adapter
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 合并 LoRA adapter..."

python stage1_solution/merge_adapter.py \
    --adapter_path "$OUTPUT_DIR/final_model"

notify "Step2 完成: adapter 合并"

# ══════════════════════════════════════════════════════════════════
# Step 3: 评估
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: 评估..."

REASON_DIR="$(cd "$DISTILL_DIR/../UniCOP-Reason" && pwd)"

cd "$REASON_DIR"
python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/$OUTPUT_DIR/final_model" \
    --problem $PROBLEM \
    --problem_size $SIZE \
    --model_type reasoning \
    --max_completion_length 4096 \
    --num_test 100 \
    --num_samples 1 \
    --batch_size 4 \
    --save_dir "$DISTILL_DIR/eval_results"
cd "$DISTILL_DIR"

notify "Template CVRP20 全部完成" "模型: $OUTPUT_DIR, 数据: $DATA_COUNT 条, 评估结果在 eval_results/"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  数据:   $CHAINS_FILE ($DATA_COUNT 条)"
echo "  模型:   $OUTPUT_DIR/final_model"
echo "  评估:   eval_results/"
echo "============================================================"
