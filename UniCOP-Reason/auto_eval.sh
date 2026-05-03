#!/bin/bash
# auto_eval.sh — 等待 4 张 GPU 完全空闲后依次评估 3 个模型，OOM 自动减半 batch_size 重试

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/paths.sh"

WORK_DIR="$REASON_DIR"
SAVE_DIR="$WORK_DIR/eval_results"
LOG_DIR="$WORK_DIR/eval_logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

MODELS=(
    "$WORK_DIR/model/Qwen3-4B/"
    "$WORK_DIR/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B/"
    "$WORK_DIR/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/"
)

INIT_BATCH_SIZE=8
TOTAL_GPUS=8
REQUIRED_GPUS=4
FREE_GPUS=""

# ── 检查是否有 4 张卡完全空闲，记录空闲卡号 ──────────────────────────
check_gpus_free() {
    FREE_GPUS=""
    local free=0
    for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
        local procs
        procs=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
        if [ "$procs" -eq 0 ]; then
            if [ -z "$FREE_GPUS" ]; then
                FREE_GPUS="$gpu_id"
            else
                FREE_GPUS="$FREE_GPUS,$gpu_id"
            fi
            free=$((free + 1))
        fi
        # 找够 4 张就够了
        if [ "$free" -ge "$REQUIRED_GPUS" ]; then
            return 0
        fi
    done
    return 1
}

# ── 等待 GPU 空闲 ────────────────────────────────────────────────────
wait_for_gpus() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待 ${REQUIRED_GPUS} 张 GPU 空闲..."
    while ! check_gpus_free; do
        sleep 30
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 已空闲，开始评估"
}

# ── 运行单个模型评估，OOM 时减半 batch_size 重试 ─────────────────────
run_eval() {
    local model_path="$1"
    local prompt_mode="$2"
    local model_name
    model_name=$(basename "$model_path")
    local bs=$INIT_BATCH_SIZE
    local log_file="$LOG_DIR/${model_name}_${prompt_mode}.log"

    while [ "$bs" -ge 1 ]; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 评估模型: $model_name | mode=$prompt_mode | batch_size=$bs"

        echo "===== mode=$prompt_mode  batch_size=$bs  $(date '+%Y-%m-%d %H:%M:%S')  GPUs=$FREE_GPUS =====" >> "$log_file"
        CUDA_VISIBLE_DEVICES="$FREE_GPUS" python "$WORK_DIR/evaluate.py" \
            --model_path "$model_path" \
            --problem tsp cvrp vrptw tsptw \
            --problem_size 20 50 100 \
            --num_test 10 \
            --max_completion_length 10000 \
            --batch_size "$bs" \
            --prompt_mode "$prompt_mode" \
            --save_dir "$SAVE_DIR" \
            >> "$log_file" 2>&1

        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model_name ($prompt_mode) 评估完成 (batch_size=$bs)"
            return 0
        fi

        # 检查是否 OOM
        if grep -qi "out of memory\|CUDA error\|OOM\|cuDNN error" "$log_file"; then
            bs=$((bs / 2))
            if [ "$bs" -ge 1 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model_name ($prompt_mode) OOM! 缩小 batch_size 为 $bs 重试..."
                sleep 5
                wait_for_gpus
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model_name ($prompt_mode) 非 OOM 错误 (exit=$exit_code)，跳过"
            echo "  日志: $log_file"
            return 1
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model_name ($prompt_mode) batch_size 已降至 0 仍 OOM，放弃"
    return 1
}

# ── 主流程 ───────────────────────────────────────────────────────────
cd "$WORK_DIR" || exit 1

PROMPT_MODES=("think" "structured")

for model in "${MODELS[@]}"; do
    for mode in "${PROMPT_MODES[@]}"; do
        wait_for_gpus
        run_eval "$model" "$mode"
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部模型评估结束"
