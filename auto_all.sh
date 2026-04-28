#!/bin/bash
# auto_all.sh — 全局自动化: merge → eval(SFT) → GRPO RL → eval(RL)
#
# 流程 (SFT 已完成,使用已有产物):
#   Step 1: 合并 LoRA → merged_model
#   Step 2: evaluate SFT merged 模型
#   Step 3: GRPO RL (tsptw + cvrp, n=20)
#   Step 4: evaluate RL 模型
#
# 任何阶段报错 → 立即退出 + Server 酱通知
#
# 启动: bash auto_all.sh  (建议 nohup 或 tmux)

set -uo pipefail

# ══════════════════════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════════════════════
MONO_DIR="/home/ntu/lzj/UniCOP"
DISTILL_DIR="$MONO_DIR/UniCOP-Distill"
REASON_DIR="$MONO_DIR/UniCOP-Reason"
TOOLS_DIR="$MONO_DIR/tools"

BASE_MODEL="/home/ntu/lzj/Model/model/DeepSeek-R1-Distill-Qwen-7B"
SFT_DATA="$DISTILL_DIR/data/chains_v3_clean.jsonl"

POMO_CKPT_DIR="/home/ntu/lzj/POMO-Baseline/result"
POMO_BASELINE_DIR="/home/ntu/lzj/POMO-Baseline"
PIPD_CKPT_DIR="/home/ntu/lzj/PIP-D baseline/POMO+PIP/pretrained/TSPTW"
PIPD_DIR="/home/ntu/lzj/PIP-D baseline/POMO+PIP"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
# 使用已有的 SFT 产物,跳过 SFT 训练
SFT_OUT="$DISTILL_DIR/output/sft_auto_20260428_154635"
SFT_MERGED="$SFT_OUT/merged_model"
GRPO_OUTPUT_DIR="$REASON_DIR/output_auto_${TIMESTAMP}"

LOG_DIR="$MONO_DIR/logs_auto_all"
EVAL_RESULT_DIR="$MONO_DIR/eval_results_auto_all"
mkdir -p "$LOG_DIR" "$EVAL_RESULT_DIR"

# ══════════════════════════════════════════════════════════════════════
# GPU 调度参数
# ══════════════════════════════════════════════════════════════════════
TOTAL_GPUS=4
SFT_GPUS=4
GRPO_INIT_GPUS=4
EVAL_GPUS=4
GPU_FREE_MEM_THRESHOLD_MB=500
FREE_GPUS=""

# ── SFT 参数 ─────────────────────────────────────────────────────────
SFT_INIT_BATCH_SIZE=4

# ── GRPO 训练矩阵 ────────────────────────────────────────────────────
GRPO_PROBLEMS=("tsptw" "cvrp")
GRPO_SIZES=(20)
GRPO_ZERO_STAGE=3

# ── vLLM server 参数 ─────────────────────────────────────────────────
VLLM_PORT_BASE=8000
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=5120
VLLM_DTYPE=bfloat16
VLLM_PREFIX_CACHE_FLAG="--enable_prefix_caching True"
VLLM_STARTUP_TIMEOUT=300
VLLM_PID=""

# ── TRL CLI 二进制 ────────────────────────────────────────────────────
TRL_BIN="$(dirname "$(which python)")/trl"
if [ ! -x "$TRL_BIN" ]; then
    echo "❌ TRL binary 未找到: $TRL_BIN"
    echo "   请在当前 env 安装: pip install 'trl[vllm]==1.1.0'"
    exit 1
fi

# ── 评估参数 ─────────────────────────────────────────────────────────
EVAL_NUM_TEST=10
EVAL_MAX_COMPLETION=10000
EVAL_INIT_BATCH_SIZE=4
EVAL_PROBLEMS="tsptw cvrp"
EVAL_SIZES="20"

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
    # 清理 vLLM server
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
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

split_gpus() {
    VLLM_GPU="${FREE_GPUS%%,*}"
    TRAIN_GPUS="${FREE_GPUS#*,}"
}

# ══════════════════════════════════════════════════════════════════════
# 阶段 1: SFT 训练
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
    CUDA_HOME=/home/ntu/anaconda3/envs/zjh \
    CUDA_VISIBLE_DEVICES="$gpus" \
        python -m accelerate.commands.launch --num_processes "$num_proc" \
        "$DISTILL_DIR/train_sft.py" \
        --model "$BASE_MODEL" \
        --data "$SFT_DATA" \
        --lora_rank 64 --lora_alpha 128 \
        --epochs 3 --lr 1e-4 \
        --batch_size 1 --grad_accum 8 \
        --max_length 8192 \
        --max_output_length 4096 \
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

    if [ -f "$SFT_MERGED/config.json" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SFT merged 已存在,跳过 merge"
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 阶段 2: 合并 SFT LoRA → $SFT_MERGED"
    CUDA_HOME=/home/ntu/anaconda3/envs/zjh \
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
# 阶段 3: GRPO RL 训练 (vLLM server mode)
# ══════════════════════════════════════════════════════════════════════
start_vllm_server() {
    local vllm_gpu=$1
    local port=$2
    local log_file=$3

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 vLLM server | GPU=$vllm_gpu | port=$port"
    PYTHONPATH="$REASON_DIR:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$vllm_gpu" \
    CUDA_HOME=/home/ntu/anaconda3/envs/zjh \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
        "$TRL_BIN" vllm-serve \
        --model "$SFT_MERGED" \
        --tensor_parallel_size 1 \
        --port "$port" \
        --gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --dtype "$VLLM_DTYPE" \
        $VLLM_PREFIX_CACHE_FLAG \
        --trust_remote_code True \
        > "$log_file" 2>&1 &
    VLLM_PID=$!

    local waited=0
    while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动失败，详见 $log_file"
            return 1
        fi
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ vLLM server 就绪 (pid=$VLLM_PID, 用时 ${waited}s)"
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动超时 (${VLLM_STARTUP_TIMEOUT}s)"
    kill "$VLLM_PID" 2>/dev/null || true
    return 1
}

stop_vllm_server() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 关闭 vLLM server (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}

run_grpo() {
    local problem=$1
    local size=$2
    local total_gpus=$3
    local free_gpus=$4
    local task_idx=$5

    FREE_GPUS="$free_gpus"
    split_gpus
    local train_proc=$((total_gpus - 1))
    local port=$((VLLM_PORT_BASE + task_idx))
    local label="${problem}_n${size}_g${train_proc}"
    local ts=$(date '+%Y%m%d_%H%M%S')
    local vllm_log="$LOG_DIR/vllm_${label}_${ts}.log"
    local train_log="$LOG_DIR/grpo_${label}_${ts}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GRPO | problem=$problem | size=$size"
    echo "  vLLM GPU:  $VLLM_GPU    port=$port    log=$vllm_log"
    echo "  训练 GPUs: $TRAIN_GPUS ($train_proc 进程)    log=$train_log"

    if ! start_vllm_server "$VLLM_GPU" "$port" "$vllm_log"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动失败,终止本任务"
        return 1
    fi

    PYTHONPATH="$REASON_DIR:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_HOME=/home/ntu/anaconda3/envs/zjh \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
        python -m accelerate.commands.launch --num_processes "$train_proc" "$REASON_DIR/train.py" \
        --problem "$problem" \
        --problem_size "$size" \
        --num_train 2000 \
        --model "$SFT_MERGED" \
        --num_gpus "$train_proc" \
        --zero_stage "$GRPO_ZERO_STAGE" \
        --gradient_checkpointing \
        --output_dir "$GRPO_OUTPUT_DIR" \
        --pomo_ckpt_dir "$POMO_CKPT_DIR" \
        --pomo_baseline_dir "$POMO_BASELINE_DIR" \
        --pipd_ckpt_dir "$PIPD_CKPT_DIR" \
        --pipd_dir "$PIPD_DIR" \
        --vllm_server_host "localhost" \
        --vllm_server_port "$port" \
        2>&1 | tee "$train_log"

    local exit_code=${PIPESTATUS[0]}
    stop_vllm_server

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ GRPO 训练完成 ($label)"
        return 0
    fi

    if grep -qiE "out of memory|OOM|CUDA out of memory|OutOfMemoryError" "$train_log"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ GRPO OOM ($label)"
        return 99
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ GRPO 非 OOM 错误 (exit=$exit_code, $label)"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 阶段 4: evaluate (OOM 时 batch_size 减半重试)
# ══════════════════════════════════════════════════════════════════════
run_eval() {
    local model_path="$1"
    local gpus="$2"
    local init_bs="$3"
    local problems="$4"
    local sizes="$5"
    local tag="$6"
    local log_file="$LOG_DIR/eval_${tag}_${TIMESTAMP}.log"

    local bs=$init_bs
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] evaluate ($tag) | GPUs=$gpus | model=$model_path"
    echo "  problems=$problems  sizes=$sizes  num_test=$EVAL_NUM_TEST"

    while [ "$bs" -ge 1 ]; do
        echo "===== batch_size=$bs  $(date '+%Y-%m-%d %H:%M:%S')  GPUs=$gpus =====" >> "$log_file"
        PYTHONPATH="$REASON_DIR:${PYTHONPATH:-}" \
        CUDA_HOME=/home/ntu/anaconda3/envs/zjh \
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
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ evaluate ($tag) 完成 (bs=$bs)"
            return 0
        fi

        if grep -qiE "out of memory|OOM|CUDA out of memory|OutOfMemoryError" "$log_file"; then
            bs=$((bs / 2))
            if [ "$bs" -ge 1 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ evaluate ($tag) OOM, batch_size 减半 → $bs"
                sleep 5
                wait_for_gpus "$(echo "$gpus" | tr ',' '\n' | wc -l)"
                gpus="$FREE_GPUS"
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ evaluate ($tag) 非 OOM 错误 (exit=$ec)"
            return 1
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ evaluate ($tag) bs 降至 0 仍 OOM"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 前置检查
# ══════════════════════════════════════════════════════════════════════
preflight() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 前置检查 =========="
    local fail=0
    for p in "$BASE_MODEL" "$SFT_OUT/final_model" \
             "$REASON_DIR/train.py" "$REASON_DIR/evaluate.py" \
             "$TOOLS_DIR/merge_lora.py" \
             "$POMO_BASELINE_DIR" "$POMO_CKPT_DIR" \
             "$PIPD_DIR" "$PIPD_CKPT_DIR"; do
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

GRPO_TOTAL=$((${#GRPO_PROBLEMS[@]} * ${#GRPO_SIZES[@]}))

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 开始全流程 =========="
echo "  SFT data:    $SFT_DATA"
echo "  SFT out:     $SFT_OUT"
echo "  SFT merged:  $SFT_MERGED"
echo "  GRPO out:    $GRPO_OUTPUT_DIR"
echo "  GRPO 矩阵:   ${GRPO_PROBLEMS[*]} × ${GRPO_SIZES[*]} = $GRPO_TOTAL 个任务"
echo "  Eval 矩阵:   $EVAL_PROBLEMS × $EVAL_SIZES"
echo "  Logs:        $LOG_DIR"
echo ""

notify "🚀 UniCOP auto_all 启动" \
"流程: merge → eval(SFT) → GRPO(tsp20) → eval(tsp20) → GRPO(剩余) → eval(剩余)
启动: $(date '+%Y-%m-%d %H:%M:%S')
SFT out: $SFT_OUT
GRPO: ${GRPO_PROBLEMS[*]} × ${GRPO_SIZES[*]}"

# ══════════════════════════════════════════════════════════════════════
# Step 1: SFT LoRA merge
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [Step 1] SFT LoRA merge ═══"
CURRENT_STAGE="Step1-SFT-merge"
if ! run_sft_merge; then
    notify "❌ UniCOP SFT merge 失败" "详见 $LOG_DIR/sft_merge_${TIMESTAMP}.log"
    exit 1
fi
notify "✓ SFT merge 完成" "开始 GRPO RL"

# ══════════════════════════════════════════════════════════════════════
# Step 2: GRPO RL 训练 (tsptw + cvrp, n=20)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [Step 2] GRPO RL ($GRPO_TOTAL 个任务) ═══"
CURRENT_STAGE="Step2-GRPO"

run_grpo_with_retry() {
    local problem=$1
    local size=$2
    local idx=$3
    wait_for_gpus "$GRPO_INIT_GPUS"
    run_grpo "$problem" "$size" "$GRPO_INIT_GPUS" "$FREE_GPUS" "$idx"
    local ec=$?
    if [ $ec -eq 1 ]; then
        notify "❌ GRPO 非 OOM 错误" "$problem n=$size"
        exit 1
    fi
    if [ $ec -eq 99 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ GRPO OOM, 等空闲卡重试"
        wait_for_gpus "$GRPO_INIT_GPUS"
        run_grpo "$problem" "$size" "$GRPO_INIT_GPUS" "$FREE_GPUS" "$idx"
        ec=$?
        if [ $ec -ne 0 ]; then
            notify "❌ GRPO 重试仍失败" "$problem n=$size (exit=$ec)"
            exit 1
        fi
    fi
}

GRPO_IDX=0
for problem in "${GRPO_PROBLEMS[@]}"; do
    GRPO_IDX=$((GRPO_IDX + 1))
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ── GRPO [${GRPO_IDX}/${GRPO_TOTAL}] $problem n=20 ──"
    run_grpo_with_retry "$problem" 20 "$GRPO_IDX"
done
notify "✓ GRPO 全部完成" "开始评估"

# ══════════════════════════════════════════════════════════════════════
# Step 3: eval SFT + RL 模型
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [Step 3] evaluate SFT merged ═══"
CURRENT_STAGE="Step3-eval-SFT"
wait_for_gpus $EVAL_GPUS
if ! run_eval "$SFT_MERGED" "$FREE_GPUS" "$EVAL_INIT_BATCH_SIZE" \
    "$EVAL_PROBLEMS" "$EVAL_SIZES" "sft"; then
    notify "❌ SFT evaluate 失败" "详见 $LOG_DIR/eval_sft_${TIMESTAMP}.log"
    exit 1
fi
notify "✓ SFT eval 完成" "开始 RL eval"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [Step 3b] evaluate RL 模型 ═══"
CURRENT_STAGE="Step3-eval-RL"

for problem in "${GRPO_PROBLEMS[@]}"; do
    RL_MODEL="$GRPO_OUTPUT_DIR/${problem}_n20/final_model"
    if [ ! -d "$RL_MODEL" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏭ RL 模型不存在,跳过 eval: $RL_MODEL"
        continue
    fi
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ── evaluate RL $problem n=20 ──"
    wait_for_gpus $EVAL_GPUS
    if ! run_eval "$RL_MODEL" "$FREE_GPUS" "$EVAL_INIT_BATCH_SIZE" \
        "$problem" "20" "rl_${problem}_n20"; then
        notify "❌ RL evaluate 失败" "$problem n=20"
        exit 1
    fi
done

# ══════════════════════════════════════════════════════════════════════
# 完成
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 全部流程结束 =========="
echo "  SFT 产物:   $SFT_OUT"
echo "  SFT 合并:   $SFT_MERGED"
echo "  GRPO 产物:  $GRPO_OUTPUT_DIR"
echo "  eval 结果:  $EVAL_RESULT_DIR"
echo "  日志目录:   $LOG_DIR"

notify "✅ UniCOP auto_all 全部完成" \
"结束: $(date '+%Y-%m-%d %H:%M:%S')
SFT merged: $SFT_MERGED
GRPO out: $GRPO_OUTPUT_DIR
eval: $EVAL_RESULT_DIR"

ALL_DONE=1
