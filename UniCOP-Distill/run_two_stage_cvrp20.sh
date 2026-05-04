#!/bin/bash
# 方案 2: 两阶段课程学习 — CVRP20
#
# Stage 1: R1-Distill + LoRA, 只学直接输出解（无 think chain）
# Stage 2: Stage 1 ckpt + LoRA, 学过滤后的高质量 think chain
#
# 使用方法：
#   nohup bash run_two_stage_cvrp20.sh > two_stage_cvrp20.log 2>&1 &

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$_SELF_DIR/two_stage_cvrp20_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1
source "$(dirname "$_SELF_DIR")/paths.sh"

# ── 配置 ──────────────────────────────────────────────────────────────────────

MODEL_PATH="$BASE_MODEL"
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=20
SOLUTIONS_FILE="data/solutions_${PROBLEM}${SIZE}.jsonl"
CHAINS_FILE="data/chains_self_${PROBLEM}${SIZE}.jsonl"
CHAINS_FILTERED="data/chains_self_${PROBLEM}${SIZE}_filtered.jsonl"
FILTERED_IDS="data/chains_self_${PROBLEM}${SIZE}_filtered_ids.txt"

S1_OUTPUT_DIR="output_two_stage_s1_${PROBLEM}${SIZE}"
S2_OUTPUT_DIR="output_two_stage_s2_${PROBLEM}${SIZE}"

# Stage 1: 与 Stage 2 同 rank，充分学习约束满足
S1_LR=1e-4
S1_EPOCHS=3
S1_LORA_RANK=64
S1_LORA_ALPHA=128
S1_MAX_LENGTH=4096  # 纯解，序列短

# Stage 2: 和之前一致
S2_LR=2e-5
S2_EPOCHS=3
S2_LORA_RANK=64
S2_LORA_ALPHA=128
S2_MAX_LENGTH=8192

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

trap 'notify "两阶段 CVRP20 失败: line $LINENO"' ERR

cd "$DISTILL_DIR"

echo "============================================================"
echo "  两阶段课程学习 — CVRP20"
echo "  Stage 1: 直接输出解 (LoRA r=$S1_LORA_RANK, ep=$S1_EPOCHS)"
echo "  Stage 2: 过滤后 think chain (LoRA r=$S2_LORA_RANK, ep=$S2_EPOCHS)"
echo "  模型:    $MODEL_PATH"
echo "  时间:    $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 0: 过滤 chains 数据（CPU 上跑，不占 GPU）
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 0: 过滤 think chain 数据..."

if [ ! -f "$CHAINS_FILE" ]; then
    echo "ERROR: $CHAINS_FILE 不存在"
    exit 1
fi

if [ ! -f "$SOLUTIONS_FILE" ]; then
    echo "ERROR: $SOLUTIONS_FILE 不存在（Stage 1 需要此文件）"
    exit 1
fi

python filter_chains.py \
    --input "$CHAINS_FILE" \
    --output "$CHAINS_FILTERED" \
    --ids_output "$FILTERED_IDS" \
    --min_coverage 0.8

FILTERED_COUNT=$(grep -c '^{' "$CHAINS_FILTERED" 2>/dev/null || echo 0)
echo "  过滤后: $FILTERED_COUNT 条"
notify "Step0 完成: 过滤后 $FILTERED_COUNT 条"

# ══════════════════════════════════════════════════════════════════
# Step 1: Stage 1 — 直接输出解（与 Stage 2 使用不同实例）
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: Stage 1 SFT（直接输出解，排除 Stage 2 数据）..."
echo "  数据:   $SOLUTIONS_FILE (排除 $FILTERED_IDS 中的实例)"
echo "  LoRA:   rank=$S1_LORA_RANK"
echo "  Epochs: $S1_EPOCHS"

S1_GPU_LIST=($(wait_for_free_gpus $SFT_NUM_GPUS))
S1_CUDA=$(IFS=,; echo "${S1_GPU_LIST[*]}")
echo "  使用 GPU: $S1_CUDA"

CUDA_VISIBLE_DEVICES=$S1_CUDA accelerate launch \
    --num_processes $SFT_NUM_GPUS \
    --main_process_port 29600 \
    stage1_solution/train_sft_stage1.py \
    --model "$MODEL_PATH" \
    --data "$SOLUTIONS_FILE" \
    --filter_problems $PROBLEM \
    --filter_sizes $SIZE \
    --exclude_ids "$FILTERED_IDS" \
    --lora_rank $S1_LORA_RANK --lora_alpha $S1_LORA_ALPHA \
    --max_length $S1_MAX_LENGTH \
    --output_dir "$S1_OUTPUT_DIR" \
    --zero_stage 3 \
    --gradient_checkpointing \
    --epochs $S1_EPOCHS \
    --batch_size 2 \
    --grad_accum 4 \
    --lr $S1_LR \
    --save_steps 500

notify "Step1 完成: Stage 1 SFT"

# ══════════════════════════════════════════════════════════════════
# Step 2: 合并 Stage 1 LoRA
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 合并 Stage 1 LoRA..."

python stage1_solution/merge_adapter.py \
    --adapter_path "$S1_OUTPUT_DIR/final_model"

notify "Step2 完成: Stage 1 adapter 合并"

# ══════════════════════════════════════════════════════════════════
# Step 3: Stage 2 — 过滤后 think chain
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: Stage 2 SFT（过滤后 think chain）..."
echo "  基座:   $S1_OUTPUT_DIR/final_model (Stage 1 合并后)"
echo "  数据:   $CHAINS_FILTERED ($FILTERED_COUNT 条)"
echo "  LoRA:   rank=$S2_LORA_RANK"
echo "  Epochs: $S2_EPOCHS"

S2_GPU_LIST=($(wait_for_free_gpus $SFT_NUM_GPUS))
S2_CUDA=$(IFS=,; echo "${S2_GPU_LIST[*]}")
echo "  使用 GPU: $S2_CUDA"

CUDA_VISIBLE_DEVICES=$S2_CUDA accelerate launch \
    --num_processes $SFT_NUM_GPUS \
    --main_process_port 29600 \
    stage2_reasoning/train_sft_stage2.py \
    --model "$S1_OUTPUT_DIR/final_model" \
    --data "$CHAINS_FILTERED" \
    --filter_problems $PROBLEM \
    --filter_sizes $SIZE \
    --lora_rank $S2_LORA_RANK --lora_alpha $S2_LORA_ALPHA \
    --max_length $S2_MAX_LENGTH \
    --output_dir "$S2_OUTPUT_DIR" \
    --zero_stage 3 \
    --gradient_checkpointing \
    --epochs $S2_EPOCHS \
    --batch_size 1 \
    --grad_accum 8 \
    --lr $S2_LR \
    --save_steps 500

notify "Step3 完成: Stage 2 SFT"

# ══════════════════════════════════════════════════════════════════
# Step 4: 合并 Stage 2 LoRA
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 4: 合并 Stage 2 LoRA..."

python stage1_solution/merge_adapter.py \
    --adapter_path "$S2_OUTPUT_DIR/final_model"

notify "Step4 完成: Stage 2 adapter 合并"

# ══════════════════════════════════════════════════════════════════
# Step 5: 评估
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 5: 评估..."

REASON_DIR="$(cd "$DISTILL_DIR/../UniCOP-Reason" && pwd)"

cd "$REASON_DIR"
python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/$S2_OUTPUT_DIR/final_model" \
    --problem $PROBLEM \
    --problem_size $SIZE \
    --model_type reasoning \
    --max_completion_length 4096 \
    --num_test 100 \
    --num_samples 1 \
    --batch_size 4 \
    --save_dir "$DISTILL_DIR/eval_results"
cd "$DISTILL_DIR"

notify "两阶段 CVRP20 全部完成" "S1: $S1_OUTPUT_DIR, S2: $S2_OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  Stage 1: $S1_OUTPUT_DIR/final_model"
echo "  Stage 2: $S2_OUTPUT_DIR/final_model"
echo "  评估:    eval_results/"
echo "============================================================"
