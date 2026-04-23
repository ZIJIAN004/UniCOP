#!/bin/bash
# auto_all.sh — 全局自动化: 数据生成 → SFT → merge → evaluate
#
# 流程:
#   阶段 0: generate_chains.py 数据生成 (concurrency=8)
#   阶段 1: SFT 训练 (4 卡, ZeRO-3 + LoRA + gc)
#   阶段 2: 合并 LoRA → merged_model
#   阶段 3: evaluate (4 卡, TSP/CVRP/VRPTW/TSPTW × 20/50/100)
#
# 任何阶段报错 → 立即退出 + Server 酱通知
#
# 启动: bash auto_all.sh  (建议 nohup 或 tmux)

set -uo pipefail

# ══════════════════════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════════════════════
MONO_DIR="/Data04/yangzhihan/lzj/UniCOP"
DISTILL_DIR="$MONO_DIR/UniCOP-Distill"
REASON_DIR="$MONO_DIR/UniCOP-Reason"
TOOLS_DIR="$MONO_DIR/tools"

BASE_MODEL="/Data04/yangzhihan/lzj/UniCOP-Reason.bak_/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SFT_DATA="$DISTILL_DIR/data/chains_v3_clean.jsonl"

# 数据生成参数
GEN_CREDENTIALS="$DISTILL_DIR/advance-subject-493905-h9-020e2dc30ae7.json"
GEN_PROJECT="advance-subject-493905-h9"
LKH_BIN="/Data04/yangzhihan/lzj/LKH-3.0.9/LKH"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SFT_OUT="$DISTILL_DIR/output_sft_auto_${TIMESTAMP}"
SFT_MERGED="$SFT_OUT/merged_model"

LOG_DIR="$MONO_DIR/logs_auto_all"
EVAL_RESULT_DIR="$MONO_DIR/eval_results_auto_all"
mkdir -p "$LOG_DIR" "$EVAL_RESULT_DIR"

# ══════════════════════════════════════════════════════════════════════
# GPU 调度参数
# ══════════════════════════════════════════════════════════════════════
TOTAL_GPUS=8
NEED_GPUS=4
EVAL_GPUS_SFT=4
GPU_FREE_MEM_THRESHOLD_MB=500
FREE_GPUS=""

SFT_INIT_BATCH_SIZE=4
EVAL_NUM_TEST=10
EVAL_MAX_COMPLETION=10000

# ══════════════════════════════════════════════════════════════════════
# 手机推送 (Server 酱 → 微信)
# ══════════════════════════════════════════════════════════════════════
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"

notify() {
    local title="$1"
    local desp="${2:-}"
    if [ -n "$SCKEY" ]; then
        curl -s --max-time 10 "https://sctapi.ftqq.com/${SCKEY}.send" \
            --data-urlencode "title=${title}" \
            --data-urlencode "desp=${desp}" > /dev/null 2>&1 || true
    fi
}

# ══════════════════════════════════════════════════════════════════════
# 全局日志 + 退出清理
# ══════════════════════════════════════════════════════════════════════
exec > >(tee -a "$LOG_DIR/auto_all_$TIMESTAMP.log") 2>&1

ALL_DONE=0
CURRENT_STAGE="初始化"
on_exit() {
    local exit_code=$?
    if [ "$ALL_DONE" != "1" ] && [ "$exit_code" != "0" ]; then
        notify "❌ UniCOP auto_all 异常退出 ($CURRENT_STAGE)" \
"退出码: $exit_code
阶段: $CURRENT_STAGE
时间: $(date '+%Y-%m-%d %H:%M:%S')
最近日志:
$(ls -t $LOG_DIR/*.log 2>/dev/null | head -1 | xargs tail -n 20 2>/dev/null || echo '(无日志)')"
    fi
}
trap on_exit EXIT INT TERM

# ══════════════════════════════════════════════════════════════════════
# GPU 工具函数
# ══════════════════════════════════════════════════════════════════════
check_gpus_free() {
    local need=$1
    FREE_GPUS=""
    local free=0
    for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
        local procs
        procs=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
                | grep -v -i "xorg\|gnome\|kde\|wayland\|Xwayland" \
                | grep -c '[0-9]')
        local mem_used
        mem_used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                   | tr -d '[:space:]')
        if [ "$procs" -eq 0 ] && [ -n "$mem_used" ] && [ "$mem_used" -lt "$GPU_FREE_MEM_THRESHOLD_MB" ]; then
            if [ -z "$FREE_GPUS" ]; then
                FREE_GPUS="$gpu_id"
            else
                FREE_GPUS="$FREE_GPUS,$gpu_id"
            fi
            free=$((free + 1))
        fi
        if [ "$free" -ge "$need" ]; then
            return 0
        fi
    done
    return 1
}

wait_for_gpus() {
    local need=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待 ${need} 张空闲 GPU (每 15s 轮询)..."
    while ! check_gpus_free "$need"; do
        sleep 15
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $FREE_GPUS 已空闲"
}

# ══════════════════════════════════════════════════════════════════════
# 阶段 0: 数据生成 (generate_chains.py)
# ══════════════════════════════════════════════════════════════════════
run_generate() {
    local log_file="$LOG_DIR/generate_${TIMESTAMP}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 阶段 0: 数据生成"
    echo "  credentials: $GEN_CREDENTIALS"
    echo "  project:     $GEN_PROJECT"
    echo "  output:      $SFT_DATA"
    echo "  concurrency: 8"
    echo "  log:         $log_file"

    cd "$DISTILL_DIR" || return 1
    python -u generate_chains.py \
        --credentials "$GEN_CREDENTIALS" \
        --project "$GEN_PROJECT" \
        --lkh_bin "$LKH_BIN" \
        --output "$SFT_DATA" \
        --num_samples 200 \
        --concurrency 8 \
        --sleep 2 \
        --max_output_tokens 4096 \
        2>&1 | tee "$log_file"

    local ec=${PIPESTATUS[0]}
    if [ $ec -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 数据生成失败 (exit=$ec)"
        return 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 数据生成完成"
    cd "$MONO_DIR" || return 1
    return 0
}

# ══════════════════════════════════════════════════════════════════════
# 阶段 1: SFT 训练 (4 卡, ZeRO-3 + LoRA + gc)
# ══════════════════════════════════════════════════════════════════════
run_sft() {
    local gpus="$1"
    local log_file="$LOG_DIR/sft_${TIMESTAMP}.log"
    local num_proc
    num_proc=$(echo "$gpus" | tr ',' '\n' | wc -l)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 阶段 1: SFT 训练 | GPUs=$gpus (${num_proc} 进程)"
    echo "  base:   $BASE_MODEL"
    echo "  data:   $SFT_DATA"
    echo "  output: $SFT_OUT"
    echo "  log:    $log_file"

    PYTHONPATH="$DISTILL_DIR:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    CUDA_HOME=/Data04/yangzhihan/envs/analog_env \
    CUDA_VISIBLE_DEVICES="$gpus" \
        python -m accelerate.commands.launch --num_processes "$num_proc" \
        "$DISTILL_DIR/train_sft.py" \
        --model "$BASE_MODEL" \
        --data "$SFT_DATA" \
        --lora_rank 64 --lora_alpha 128 \
        --epochs 3 --lr 1e-4 \
        --batch_size 1 --grad_accum 8 \
        --max_length 8192 \
        --val_ratio 0 \
        --zero_stage 3 --gradient_checkpointing \
        --output_dir "$SFT_OUT" \
        2>&1 | tee "$log_file"

    local ec=${PIPESTATUS[0]}
    if [ $ec -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SFT 训练完成"
        return 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ SFT 失败 (exit=$ec)"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 阶段 2: 合并 SFT LoRA adapter → merged_model
# ══════════════════════════════════════════════════════════════════════
run_sft_merge() {
    local adapter="$SFT_OUT/final_model"
    local log_file="$LOG_DIR/sft_merge_${TIMESTAMP}.log"

    if [ ! -d "$adapter" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ SFT adapter 不存在: $adapter"
        return 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 阶段 2: 合并 SFT LoRA → $SFT_MERGED"
    python "$TOOLS_DIR/merge_lora.py" \
        --adapter "$adapter" \
        --base "$BASE_MODEL" \
        --output "$SFT_MERGED" \
        --device cpu \
        2>&1 | tee "$log_file"

    local ec=${PIPESTATUS[0]}
    if [ $ec -eq 0 ] && [ -f "$SFT_MERGED/config.json" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SFT merge 完成"
        return 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ SFT merge 失败 (exit=$ec)"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 阶段 3: evaluate (OOM 时 batch_size 减半重试)
# ══════════════════════════════════════════════════════════════════════
run_eval() {
    local model_path="$1"
    local gpus="$2"
    local init_bs="$3"
    local problems="$4"
    local sizes="$5"
    local log_file="$LOG_DIR/eval_sft_${TIMESTAMP}.log"

    local bs=$init_bs
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 阶段 3: evaluate | GPUs=$gpus | model=$model_path"
    echo "  problems=$problems  sizes=$sizes  num_test=$EVAL_NUM_TEST"

    while [ "$bs" -ge 1 ]; do
        echo "===== batch_size=$bs  $(date '+%Y-%m-%d %H:%M:%S')  GPUs=$gpus =====" >> "$log_file"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
            CUDA_VISIBLE_DEVICES="$gpus" \
            python "$REASON_DIR/evaluate.py" \
            --model_path "$model_path" \
            --problem $problems \
            --problem_size $sizes \
            --num_test "$EVAL_NUM_TEST" \
            --max_completion_length "$EVAL_MAX_COMPLETION" \
            --batch_size "$bs" \
            --prompt_mode think \
            --save_dir "$EVAL_RESULT_DIR" \
            >> "$log_file" 2>&1
        local ec=$?

        if [ $ec -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ evaluate 完成 (bs=$bs)"
            return 0
        fi

        if grep -qiE "out of memory|OOM|CUDA out of memory|OutOfMemoryError" "$log_file"; then
            bs=$((bs / 2))
            if [ "$bs" -ge 1 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ evaluate OOM, batch_size 减半 → $bs, 等空闲卡后重试"
                sleep 5
                wait_for_gpus "$(echo "$gpus" | tr ',' '\n' | wc -l)"
                gpus="$FREE_GPUS"
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ evaluate 非 OOM 错误 (exit=$ec)"
            return 1
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ evaluate bs 降至 0 仍 OOM"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 前置检查
# ══════════════════════════════════════════════════════════════════════
preflight() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 前置检查 =========="
    local fail=0
    for p in "$BASE_MODEL" "$DISTILL_DIR/train_sft.py" "$DISTILL_DIR/generate_chains.py" \
             "$REASON_DIR/evaluate.py" "$TOOLS_DIR/merge_lora.py" \
             "$GEN_CREDENTIALS" "$LKH_BIN"; do
        if [ ! -e "$p" ]; then
            echo "  ✗ 路径不存在: $p"
            fail=1
        else
            echo "  ✓ $p"
        fi
    done
    if [ "$fail" = "1" ]; then
        echo "[FAIL] 前置检查不通过"
        exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 前置检查通过 =========="
}

# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════
cd "$MONO_DIR" || exit 1
preflight

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 开始全流程 =========="
echo "  数据:       $SFT_DATA"
echo "  SFT out:    $SFT_OUT"
echo "  SFT merged: $SFT_MERGED"
echo "  Logs:       $LOG_DIR"
echo ""

notify "🚀 UniCOP auto_all 启动" \
"流程: 数据生成 → SFT → merge → evaluate
启动: $(date '+%Y-%m-%d %H:%M:%S')
SFT data: $SFT_DATA
SFT out: $SFT_OUT"

# ── 阶段 0: 数据生成 ─────────────────────────────────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [0/3] 数据生成 ═══"
CURRENT_STAGE="阶段0-数据生成"
if ! run_generate; then
    notify "❌ UniCOP 数据生成失败" "详见 $LOG_DIR/generate_${TIMESTAMP}.log"
    exit 1
fi
notify "✓ 数据生成完成" "开始等待 GPU 进入 SFT"

# ── 阶段 1: SFT 训练 ─────────────────────────────────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [1/3] SFT 训练 ═══"
CURRENT_STAGE="阶段1-SFT训练"
wait_for_gpus $NEED_GPUS
if ! run_sft "$FREE_GPUS"; then
    notify "❌ UniCOP SFT 训练失败" "详见 $LOG_DIR/sft_${TIMESTAMP}.log"
    exit 1
fi
notify "✓ SFT 训练完成" "开始 merge"

# ── 阶段 2: SFT LoRA merge ───────────────────────────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [2/3] SFT LoRA merge ═══"
CURRENT_STAGE="阶段2-SFT-merge"
if ! run_sft_merge; then
    notify "❌ UniCOP SFT merge 失败" "详见 $LOG_DIR/sft_merge_${TIMESTAMP}.log"
    exit 1
fi

# ── 阶段 3: evaluate ─────────────────────────────────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [3/3] SFT evaluate (全套 4×3) ═══"
CURRENT_STAGE="阶段3-evaluate"
wait_for_gpus $EVAL_GPUS_SFT
if ! run_eval "$SFT_MERGED" "$FREE_GPUS" "$SFT_INIT_BATCH_SIZE" \
    "tsp cvrp vrptw tsptw" "20 50 100"; then
    notify "❌ UniCOP evaluate 失败" "详见 $LOG_DIR/eval_sft_${TIMESTAMP}.log"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════
# 完成
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 全部流程结束 =========="
echo "  数据:       $SFT_DATA"
echo "  SFT 产物:   $SFT_OUT"
echo "  SFT 合并:   $SFT_MERGED"
echo "  eval 结果:  $EVAL_RESULT_DIR"
echo "  日志目录:   $LOG_DIR"

notify "✅ UniCOP auto_all 全部完成" \
"结束: $(date '+%Y-%m-%d %H:%M:%S')
SFT merged: $SFT_MERGED
eval: $EVAL_RESULT_DIR"

ALL_DONE=1
