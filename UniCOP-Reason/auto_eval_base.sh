#!/bin/bash
# auto_eval_base.sh — 对基座模型（未经 SFT）进行 zero-shot 评估
# 调度策略：
#   - 4 张卡空闲 + 同时有 1.5B 和大模型任务：1 卡跑 1.5B + 3 卡跑大模型（并行）
#   - 3 张卡空闲：优先跑 4B，其次 7B
#   - 2 张卡空闲且仅剩 1.5B 任务：两张卡各跑一个 1.5B 任务（并行）
#   - 大模型优先级：4B > 7B

EVAL_DIR="/home/ntu/lzj/UniCOP/UniCOP-Reason"
LOG_DIR="$EVAL_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL_GPUS=4

# 模型列表
MODEL_1_5B="/home/ntu/lzj/UniCOP/UniCOP-Reason/model/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_4B="/home/ntu/lzj/UniCOP/UniCOP-Reason/model/Qwen3-4B-Thinking-2507"
MODEL_7B="/home/ntu/lzj/UniCOP/UniCOP-Reason/model/DeepSeek-R1-Distill-Qwen-7B"

# 全局日志
exec > >(tee -a "$LOG_DIR/auto_eval_base_$(date '+%Y%m%d_%H%M%S').log") 2>&1

# ── 任务队列 ─────────────────────────────────────────────────────────
# 大模型任务（需 3 张卡），4B 优先
LARGE_TASKS=()
[ -d "$MODEL_4B" ]  && LARGE_TASKS+=("$MODEL_4B|qwen3_4b_base|$EVAL_DIR/eval_results_qwen3_4b_base_think|10000|think")
[ -d "$MODEL_4B" ]  && LARGE_TASKS+=("$MODEL_4B|qwen3_4b_base|$EVAL_DIR/eval_results_qwen3_4b_base_structured|10000|structured")
[ -d "$MODEL_7B" ]  && LARGE_TASKS+=("$MODEL_7B|r1_7b_base|$EVAL_DIR/eval_results_r1_7b_base_think|10000|think")
[ -d "$MODEL_7B" ]  && LARGE_TASKS+=("$MODEL_7B|r1_7b_base|$EVAL_DIR/eval_results_r1_7b_base_structured|10000|structured")

# 小模型任务（需 1 张卡）
SMALL_TASKS=()
[ -d "$MODEL_1_5B" ] && SMALL_TASKS+=("$MODEL_1_5B|r1_1.5b_base|$EVAL_DIR/eval_results_r1_1.5b_base_think|10000|think")
[ -d "$MODEL_1_5B" ] && SMALL_TASKS+=("$MODEL_1_5B|r1_1.5b_base|$EVAL_DIR/eval_results_r1_1.5b_base_structured|10000|structured")

LARGE_IDX=0
SMALL_IDX=0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 大模型任务: ${#LARGE_TASKS[@]} 个，小模型任务: ${#SMALL_TASKS[@]} 个"

# ── 获取空闲 GPU 列表（返回空格分隔的卡号） ─────────────────────────
get_free_gpus() {
    local gpus=()
    for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
        local procs
        procs=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v -i "xorg\|gnome\|kde\|wayland\|Xwayland" | grep -c '[0-9]')
        if [ "$procs" -eq 0 ]; then
            gpus+=("$gpu_id")
        fi
    done
    echo "${gpus[@]}"
}

# ── 单次评估（含 OOM 降级） ──────────────────────────────────────────
run_eval() {
    local model_path="$1"
    local label="$2"
    local save_dir="$3"
    local max_len="$4"
    local prompt_mode="$5"
    local gpu_ids="$6"
    local bs=4

    while [ "$bs" -ge 1 ]; do
        local log_file="$LOG_DIR/eval_${label}_${prompt_mode}_bs${bs}.log"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluate | model=$label | mode=$prompt_mode | max_len=$max_len | bs=$bs | GPU=$gpu_ids"

        CUDA_HOME=/home/ntu/anaconda3/envs/unicop CUDA_VISIBLE_DEVICES="$gpu_ids" \
            python "$EVAL_DIR/evaluate.py" \
            --model_path "$model_path" \
            --problem tsp cvrp tsptw vrptw \
            --problem_size 20 50 100 \
            --num_test 10 \
            --max_completion_length "$max_len" \
            --batch_size "$bs" \
            --prompt_mode "$prompt_mode" \
            --save_dir "$save_dir" \
            2>&1 | tee "$log_file"

        local exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluate 完成 ($label, $prompt_mode, bs=$bs)"
            return 0
        fi

        if grep -qi "out of memory\|CUDA error\|OOM" "$log_file"; then
            bs=$((bs / 2))
            if [ "$bs" -ge 1 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] OOM! 缩小 batch_size 为 $bs 重试..."
                sleep 5
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 非 OOM 错误 (exit=$exit_code)，跳过"
            echo "  日志: $log_file"
            return 1
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] batch_size 降至 0 仍 OOM，放弃"
    return 1
}

# ── 从任务描述中解析并执行 ───────────────────────────────────────────
dispatch_task() {
    local task_str="$1"
    local gpu_ids="$2"
    IFS='|' read -r model_path label save_dir max_len prompt_mode <<< "$task_str"
    run_eval "$model_path" "$label" "$save_dir" "$max_len" "$prompt_mode" "$gpu_ids"
}

# ══════════════════════════════════════════════════════════════════════
# 主调度循环
# ══════════════════════════════════════════════════════════════════════
cd "$EVAL_DIR" || exit 1

while [ "$LARGE_IDX" -lt "${#LARGE_TASKS[@]}" ] || [ "$SMALL_IDX" -lt "${#SMALL_TASKS[@]}" ]; do

    FREE_LIST=($(get_free_gpus))
    FREE_COUNT=${#FREE_LIST[@]}

    HAS_LARGE=$(( LARGE_IDX < ${#LARGE_TASKS[@]} ))
    HAS_SMALL=$(( SMALL_IDX < ${#SMALL_TASKS[@]} ))

    # ── 情况 1：4+ 张卡空闲，同时有大模型和小模型任务 → 并行 ────────
    if [ "$FREE_COUNT" -ge 4 ] && [ "$HAS_LARGE" -eq 1 ] && [ "$HAS_SMALL" -eq 1 ]; then
        SMALL_GPU="${FREE_LIST[0]}"
        LARGE_GPU="${FREE_LIST[1]},${FREE_LIST[2]},${FREE_LIST[3]}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 调度：并行（1.5B on GPU $SMALL_GPU + 大模型 on GPU $LARGE_GPU）"

        dispatch_task "${SMALL_TASKS[$SMALL_IDX]}" "$SMALL_GPU" &
        PID_SMALL=$!
        dispatch_task "${LARGE_TASKS[$LARGE_IDX]}" "$LARGE_GPU" &
        PID_LARGE=$!

        SMALL_IDX=$((SMALL_IDX + 1))
        LARGE_IDX=$((LARGE_IDX + 1))

        wait $PID_SMALL $PID_LARGE
        continue
    fi

    # ── 情况 2：3+ 张卡空闲，有大模型任务 → 跑大模型 ────────────────
    if [ "$FREE_COUNT" -ge 3 ] && [ "$HAS_LARGE" -eq 1 ]; then
        LARGE_GPU="${FREE_LIST[0]},${FREE_LIST[1]},${FREE_LIST[2]}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 调度：大模型 on GPU $LARGE_GPU"

        dispatch_task "${LARGE_TASKS[$LARGE_IDX]}" "$LARGE_GPU"
        LARGE_IDX=$((LARGE_IDX + 1))
        continue
    fi

    # ── 情况 3：2+ 张卡空闲，有 2 个小模型任务 → 并行两个 1.5B ──────
    if [ "$FREE_COUNT" -ge 2 ] && [ "$HAS_SMALL" -eq 1 ] && [ $((SMALL_IDX + 1)) -lt "${#SMALL_TASKS[@]}" ]; then
        GPU_A="${FREE_LIST[0]}"
        GPU_B="${FREE_LIST[1]}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 调度：两个 1.5B 并行（GPU $GPU_A + GPU $GPU_B）"

        dispatch_task "${SMALL_TASKS[$SMALL_IDX]}" "$GPU_A" &
        PID_A=$!
        dispatch_task "${SMALL_TASKS[$((SMALL_IDX + 1))]}" "$GPU_B" &
        PID_B=$!

        SMALL_IDX=$((SMALL_IDX + 2))

        wait $PID_A $PID_B
        continue
    fi

    # ── 情况 4：1+ 张卡空闲，有小模型任务 → 跑 1 个 1.5B ───────────
    if [ "$FREE_COUNT" -ge 1 ] && [ "$HAS_SMALL" -eq 1 ]; then
        SMALL_GPU="${FREE_LIST[0]}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 调度：1.5B on GPU $SMALL_GPU"

        dispatch_task "${SMALL_TASKS[$SMALL_IDX]}" "$SMALL_GPU"
        SMALL_IDX=$((SMALL_IDX + 1))
        continue
    fi

    # ── 无足够空闲卡，等待 ───────────────────────────────────────────
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 空闲卡不足 ($FREE_COUNT)，等待 30s..."
    sleep 30

done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 全部评估完成 =========="
