#!/bin/bash
# auto_eval_rep_penalty.sh — 对比 no_repeat_ngram_size 对 SFT 模型的影响
# 评估矩阵：SFT 模型 × n ∈ {4, 5, 6, 7} = 4 组评估
#
# 设计要点：
#   - 只跑 SFT 模型
#   - 使用 HF 原生 no_repeat_ngram_size：全局硬禁（含 Route 输出段）
#     Route 里几乎不会连续出现 4+ 个相同 token，所以不需要门控与 exempt 列表
#   - repetition_penalty 固定 1.0，彻底关闭单 token 惩罚
#   - max_completion_length 保持 10000，与历史截断口径一致
#   - 全 8 卡直接跑，OOM 才降级到 4 空闲卡

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/paths.sh"

EVAL_DIR="$REASON_DIR"
LOG_DIR="$EVAL_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL_GPUS=8
ALL_GPUS="0,1,2,3,4,5,6,7"
OOM_FALLBACK_GPUS=4
FREE_GPUS=""

# 模型路径
MODEL_SFT="$DISTILL_DIR/output/final_model"

MAX_COMPLETION_LENGTH=10000

# 全局日志
exec > >(tee -a "$LOG_DIR/auto_eval_ngram_$(date '+%Y%m%d_%H%M%S').log") 2>&1

# ── OOM 降级时才用：检查是否有足够空闲 GPU ──────────────────────────
check_gpus_free() {
    FREE_GPUS=""
    local free=0
    for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
        local procs
        procs=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v -i "xorg\|gnome\|kde\|wayland\|Xwayland" | grep -c '[0-9]')
        if [ "$procs" -eq 0 ]; then
            if [ -z "$FREE_GPUS" ]; then
                FREE_GPUS="$gpu_id"
            else
                FREE_GPUS="$FREE_GPUS,$gpu_id"
            fi
            free=$((free + 1))
        fi
        if [ "$free" -ge "$OOM_FALLBACK_GPUS" ]; then
            return 0
        fi
    done
    return 1
}

wait_for_fallback_gpus() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OOM 降级：等待 ${OOM_FALLBACK_GPUS} 张 GPU 空闲..."
    while ! check_gpus_free; do
        sleep 30
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $FREE_GPUS 空闲，降级重试"
}

# ── Evaluate：先全 8 卡 bs=4，OOM 时降级到 4 卡并继续缩小 bs ─────────
# 注：使用 expandable_segments 减缓碎片化（2.72 GB reserved-unallocated 的症状）
run_eval() {
    local model_path="$1"
    local label="$2"
    local save_dir="$3"
    local ngram_n="$4"

    # 阶段 1：直接用全 8 卡，bs=4（保守起点，避开 bs=8 的 OOM 风险）
    local bs=4
    local gpus="$ALL_GPUS"
    local log_file="$LOG_DIR/eval_${label}_ngram${ngram_n}_full8.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluate | model=$label | no_repeat_ngram=$ngram_n | bs=$bs | GPU=$gpus (full 8)"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_HOME="$CUDA_HOME" \
    CUDA_VISIBLE_DEVICES="$gpus" python -u "$EVAL_DIR/evaluate.py" \
        --model_path "$model_path" \
        --problem tsp cvrp tsptw vrptw \
        --problem_size 20 50 100 \
        --num_test 10 \
        --max_completion_length "$MAX_COMPLETION_LENGTH" \
        --batch_size "$bs" \
        --prompt_mode think \
        --repetition_penalty 1.0 \
        --no_repeat_ngram_size "$ngram_n" \
        --save_dir "$save_dir" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluate 完成 ($label, ngram=$ngram_n, full8, bs=$bs)"
        return 0
    fi

    if ! grep -qi "out of memory\|CUDA error\|OOM" "$log_file"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 非 OOM 错误 (exit=$exit_code)，跳过"
        echo "  日志: $log_file"
        return 1
    fi

    # 阶段 2：OOM 降级——等 4 张空闲卡，bs 依次减半
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全 8 卡 OOM！进入降级重试流程"
    bs=$((bs / 2))

    while [ "$bs" -ge 1 ]; do
        wait_for_fallback_gpus
        log_file="$LOG_DIR/eval_${label}_ngram${ngram_n}_fb${OOM_FALLBACK_GPUS}_bs${bs}.log"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Retry | model=$label | no_repeat_ngram=$ngram_n | bs=$bs | GPU=$FREE_GPUS"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_HOME="$CUDA_HOME" \
        CUDA_VISIBLE_DEVICES="$FREE_GPUS" python -u "$EVAL_DIR/evaluate.py" \
            --model_path "$model_path" \
            --problem tsp cvrp tsptw vrptw \
            --problem_size 20 50 100 \
            --num_test 10 \
            --max_completion_length "$MAX_COMPLETION_LENGTH" \
            --batch_size "$bs" \
            --prompt_mode think \
            --repetition_penalty 1.0 \
            --no_repeat_ngram_size "$ngram_n" \
            --save_dir "$save_dir" \
            2>&1 | tee "$log_file"

        exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluate 完成 ($label, ngram=$ngram_n, fb, bs=$bs)"
            return 0
        fi

        if grep -qi "out of memory\|CUDA error\|OOM" "$log_file"; then
            bs=$((bs / 2))
            if [ "$bs" -ge 1 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 降级仍 OOM，继续缩 bs 为 $bs"
                sleep 5
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 非 OOM 错误 (exit=$exit_code)，放弃"
            return 1
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] batch_size 降至 0 仍 OOM，放弃"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 主流程：SFT × no_repeat_ngram_size ∈ {4, 5, 6, 7}
# ══════════════════════════════════════════════════════════════════════
cd "$EVAL_DIR" || exit 1

if [ ! -d "$MODEL_SFT" ]; then
    echo "模型不存在: $MODEL_SFT"
    exit 1
fi

# ── 1. n=4 （最严）─────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 1/4: SFT, no_repeat_ngram=4 =========="
run_eval "$MODEL_SFT" "r1_7b_sft" "$EVAL_DIR/eval_results_sft_ngram_n4" 4

# ── 2. n=5 ───────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 2/4: SFT, no_repeat_ngram=5 =========="
run_eval "$MODEL_SFT" "r1_7b_sft" "$EVAL_DIR/eval_results_sft_ngram_n5" 5

# ── 3. n=6 ───────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 3/4: SFT, no_repeat_ngram=6 =========="
run_eval "$MODEL_SFT" "r1_7b_sft" "$EVAL_DIR/eval_results_sft_ngram_n6" 6

# ── 4. n=7 （最松）────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 4/4: SFT, no_repeat_ngram=7 =========="
run_eval "$MODEL_SFT" "r1_7b_sft" "$EVAL_DIR/eval_results_sft_ngram_n7" 7

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 全部评估完成 =========="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 结果目录:"
echo "  - $EVAL_DIR/eval_results_sft_ngram_n4"
echo "  - $EVAL_DIR/eval_results_sft_ngram_n5"
echo "  - $EVAL_DIR/eval_results_sft_ngram_n6"
echo "  - $EVAL_DIR/eval_results_sft_ngram_n7"
