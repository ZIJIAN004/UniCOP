#!/bin/bash
# Hybrid CVRP20 SFT: 在 template SFT 产物上继续训练 hybrid 数据
#
# 流程:
#   Step 0:   检查基座模型——如果是 LoRA adapter 则先合并为完整权重
#   Step 0.5: 断点续训检查——有 checkpoint 则合并其 LoRA adapter，计算剩余 epoch
#   Step 1:   SFT 训练（hybrid 数据，首次 3 epoch / 续训剩余 epoch）
#   Step 2:   合并 LoRA adapter
#   Step 3:   评估
#
# 使用方法：
#   sbatch submit_sft_hybrid_cvrp20.sh
#   bash run_sft_hybrid_cvrp20.sh

set -euo pipefail

_SELF_DIR="$(pwd)"
LOG_FILE="$_SELF_DIR/sft_hybrid_cvrp20_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1

source "$(dirname "$_SELF_DIR")/paths.sh"

# ── 配置 ──────────────────────────────────────────────────────────────────────

SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=20

# 基座模型: template SFT 产物
INPUT_MODEL_DIR="output_sft_template_${PROBLEM}${SIZE}/final_model"

# 训练数据：原始 chains + annotated（关键步骤带 LLM 简短理由）一起喂入
CHAINS_FILE="data/chains_hybrid_${PROBLEM}${SIZE}.jsonl"
CHAINS_ANNOTATED="data/chains_hybrid_${PROBLEM}${SIZE}_annotated.jsonl"

# 输出目录
OUTPUT_DIR="output_sft_hybrid_${PROBLEM}${SIZE}"

# SFT 配置（与 run_sft_template_cvrp20.sh 一致）
SFT_LR=2e-5
SFT_EPOCHS=3
SFT_LORA_RANK=64
SFT_LORA_ALPHA=128
SFT_MAX_LENGTH=4864
SFT_NUM_GPUS=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 工具函数 ──────────────────────────────────────────────────────────────────
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
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

trap 'notify "❌ Hybrid CVRP20 SFT 失败: line $LINENO"' ERR

cd "$DISTILL_DIR"

# ── 检查数据 ──────────────────────────────────────────────────────────────────
for f in "$CHAINS_FILE" "$CHAINS_ANNOTATED"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: 数据文件 $f 不存在"
        exit 1
    fi
done
DATA_COUNT=$(grep -c '^{' "$CHAINS_FILE" 2>/dev/null || echo 0)
DATA_COUNT_ANN=$(grep -c '^{' "$CHAINS_ANNOTATED" 2>/dev/null || echo 0)
DATA_TOTAL=$((DATA_COUNT + DATA_COUNT_ANN))

echo "============================================================"
echo "  Hybrid CVRP20 SFT + 评估"
echo "  基座模型:    $INPUT_MODEL_DIR"
echo "  原始 chains: $CHAINS_FILE ($DATA_COUNT 条)"
echo "  annotated:   $CHAINS_ANNOTATED ($DATA_COUNT_ANN 条)"
echo "  合计:        $DATA_TOTAL 条"
echo "  GPU:         $SFT_NUM_GPUS 张"
echo "  输出:        $OUTPUT_DIR"
echo "  时间:        $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 0: 检查基座模型——LoRA adapter 则先合并
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 0: 检查基座模型类型..."

MODEL_PATH="$INPUT_MODEL_DIR"

if [ -f "$INPUT_MODEL_DIR/adapter_config.json" ]; then
    echo "  检测到 adapter_config.json → LoRA adapter，需先合并"
    python stage1_solution/merge_adapter.py \
        --adapter_path "$INPUT_MODEL_DIR"
    echo "  ✓ 合并完成，$INPUT_MODEL_DIR 已变为完整权重"
    notify "Step0 完成: 基座 adapter 已合并"
elif [ -f "$INPUT_MODEL_DIR/config.json" ]; then
    echo "  检测到 config.json（无 adapter_config.json）→ 已是完整权重，跳过合并"
else
    echo "ERROR: $INPUT_MODEL_DIR 下既无 adapter_config.json 也无 config.json"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════
# Step 0.5: 断点续训检查
#   checkpoint 保存的是 LoRA adapter，续训前需合并回基座得到完整权重，
#   再以此为新 --model 继续 LoRA 训练剩余 epoch。
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 0.5: 检查断点..."

REMAINING_EPOCHS=$SFT_EPOCHS
SKIP_TRAINING=false

if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)

    if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/trainer_state.json" ]; then
        echo "  检测到断点: $LATEST_CKPT"

        RUN_EPOCHS=$SFT_EPOCHS
        if [ -f "$OUTPUT_DIR/resumed_epochs" ]; then
            RUN_EPOCHS=$(cat "$OUTPUT_DIR/resumed_epochs")
        fi

        COMPLETED_IN_RUN=$(python3 -c "
import json, math
with open('$LATEST_CKPT/trainer_state.json') as f:
    state = json.load(f)
print(int(math.floor(state['epoch'])))
")
        REMAINING_EPOCHS=$((RUN_EPOCHS - COMPLETED_IN_RUN))
        echo "  本轮 $RUN_EPOCHS epochs, 已完成 $COMPLETED_IN_RUN, 剩余 $REMAINING_EPOCHS"

        if [ "$REMAINING_EPOCHS" -le 0 ]; then
            echo "  训练已完成，跳过 Step 1"
            SKIP_TRAINING=true
        else
            echo "  合并断点 LoRA adapter 为完整权重..."
            RESUMED_MODEL="$OUTPUT_DIR/resumed_model"
            python stage1_solution/merge_adapter.py \
                --adapter_path "$LATEST_CKPT" \
                --output_path "$RESUMED_MODEL"
            MODEL_PATH="$RESUMED_MODEL"
            echo "$REMAINING_EPOCHS" > "$OUTPUT_DIR/resumed_epochs"
            rm -rf "$OUTPUT_DIR"/checkpoint-*
            echo "  ✓ 从 $MODEL_PATH 续训 $REMAINING_EPOCHS epochs"
            notify "断点恢复: 续训 $REMAINING_EPOCHS epochs"
        fi

    elif [ -f "$OUTPUT_DIR/resumed_model/config.json" ]; then
        if [ -f "$OUTPUT_DIR/resumed_epochs" ]; then
            REMAINING_EPOCHS=$(cat "$OUTPUT_DIR/resumed_epochs")
        fi
        MODEL_PATH="$OUTPUT_DIR/resumed_model"
        echo "  使用上次已合并的断点模型，续训 $REMAINING_EPOCHS epochs"

    else
        echo "  未发现可用断点，从头训练"
    fi
else
    echo "  输出目录不存在，从头训练"
fi

# ══════════════════════════════════════════════════════════════════
# Step 1: SFT 训练
# ══════════════════════════════════════════════════════════════════
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo ">>> Step 1: SFT 训练 ($REMAINING_EPOCHS epochs)..."

    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

    accelerate launch \
        --num_processes $SFT_NUM_GPUS \
        --main_process_port 29601 \
        stage2_reasoning/train_sft_stage2.py \
        --model "$MODEL_PATH" \
        --data "$CHAINS_FILE" "$CHAINS_ANNOTATED" \
        --filter_problems $PROBLEM \
        --filter_sizes $SIZE \
        --lora_rank $SFT_LORA_RANK --lora_alpha $SFT_LORA_ALPHA \
        --max_length $SFT_MAX_LENGTH \
        --output_dir "$OUTPUT_DIR" \
        --zero_stage 3 \
        --gradient_checkpointing \
        --epochs $REMAINING_EPOCHS \
        --batch_size 1 \
        --grad_accum 8 \
        --lr $SFT_LR \
        --save_steps 500

    notify "Step1 完成: Hybrid SFT 训练"
else
    echo ""
    echo ">>> Step 1: 跳过（训练已完成）"
fi

# ══════════════════════════════════════════════════════════════════
# Step 2: 合并 LoRA adapter
#   注意：必须在清理 resumed_model 之前完成，
#   因为 final_model/adapter_config.json 的 base_model_name_or_path
#   指向 resumed_model（断点续训分支）。
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 合并 LoRA adapter..."

if [ -f "$OUTPUT_DIR/final_model/adapter_config.json" ]; then
    python stage1_solution/merge_adapter.py \
        --adapter_path "$OUTPUT_DIR/final_model"
    notify "Step2 完成: adapter 合并"
else
    echo "  final_model 已是完整权重，跳过合并"
fi

# Step 2 完成后再清理断点续训的中间产物
rm -f "$OUTPUT_DIR/resumed_epochs"
rm -rf "$OUTPUT_DIR/resumed_model"

# ══════════════════════════════════════════════════════════════════
# Step 3: 评估
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: 评估..."

EVAL_SAVE_DIR="$DISTILL_DIR/eval_results_hybrid"
mkdir -p "$EVAL_SAVE_DIR"

REASON_DIR="$(cd "$DISTILL_DIR/../UniCOP-Reason" && pwd)"

cd "$REASON_DIR"
python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/$OUTPUT_DIR/final_model" \
    --problem $PROBLEM \
    --problem_size $SIZE \
    --model_type reasoning \
    --max_completion_length 10000 \
    --num_test 100 \
    --num_samples 1 \
    --batch_size 4 \
    --prompt_mode think \
    --save_dir "$EVAL_SAVE_DIR"
cd "$DISTILL_DIR"

notify "✅ Hybrid CVRP20 全部完成" "模型: $OUTPUT_DIR, 数据: $DATA_TOTAL 条 (原 $DATA_COUNT + annotated $DATA_COUNT_ANN), 评估结果在 eval_results_hybrid/"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  数据:     $CHAINS_FILE ($DATA_COUNT) + $CHAINS_ANNOTATED ($DATA_COUNT_ANN) = $DATA_TOTAL 条"
echo "  模型:     $OUTPUT_DIR/final_model"
echo "  评估:     $EVAL_SAVE_DIR/"
echo "============================================================"
