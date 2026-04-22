#!/bin/bash
# auto_all.sh — 全局自动化: SFT → merge → SFT eval → RL (TSP n=20) → RL eval
#
# 流程:
#   阶段 1: SFT 训练 (4 卡, ZeRO-3 + LoRA + gc)
#   阶段 2: 合并 LoRA → merged_model
#   阶段 3: SFT evaluate (4 卡, TSP/CVRP/VRPTW/TSPTW × 20/50/100 = 12 组)
#   阶段 4: RL 训练 (4 卡: 1 vLLM server + 3 训练, TSP n=20)
#   阶段 5: RL evaluate (合并 RL LoRA 后, 1 卡)
#
# OOM 策略:
#   阶段 1/4 OOM → 等 4 张空闲卡重试一次 (不升级到 8 卡)
#   阶段 3/5 OOM → 减半 batch_size 重试
#
# 每个阶段启动前都 wait_for_gpus 检查空闲卡。
#
# 启动: bash auto_all.sh  (建议 nohup 或 tmux)

set -u
set -o pipefail

# ══════════════════════════════════════════════════════════════════════
# 路径配置（远程服务器,有疑问见 README）
# ══════════════════════════════════════════════════════════════════════
MONO_DIR="/Data04/yangzhihan/lzj/UniCOP"
DISTILL_DIR="$MONO_DIR/UniCOP-Distill"
REASON_DIR="$MONO_DIR/UniCOP-Reason"
TOOLS_DIR="$MONO_DIR/tools"

BASE_MODEL="/Data04/yangzhihan/lzj/UniCOP-Reason.bak_/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SFT_DATA="$DISTILL_DIR/data/chains_v3_clean.jsonl"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
# 复用已完成的 SFT 产物，跳过阶段 1
SFT_OUT="$DISTILL_DIR/output_sft_auto_20260423_024302"
SFT_MERGED="$SFT_OUT/merged_model"

# RL 输出沿用 auto_train.sh 的结构
RL_OUTPUT_BASE="$REASON_DIR/output"
RL_PROBLEM="tsp"
RL_SIZE=20
RL_ADAPTER="$RL_OUTPUT_BASE/${RL_PROBLEM}_n${RL_SIZE}/final_model"
RL_MERGED="$RL_OUTPUT_BASE/${RL_PROBLEM}_n${RL_SIZE}/merged_model"

LOG_DIR="$MONO_DIR/logs_auto_all"
EVAL_RESULT_DIR="$MONO_DIR/eval_results_auto_all"
mkdir -p "$LOG_DIR" "$EVAL_RESULT_DIR"

# POMO / PIP-D (RL 阶段需要)
POMO_CKPT_DIR="/Data04/yangzhihan/lzj/POMO-Baseline/result"
POMO_BASELINE_DIR="/Data04/yangzhihan/lzj/POMO-Baseline"
PIPD_CKPT_DIR="/Data04/yangzhihan/lzj/PIP-D baseline/POMO+PIP/pretrained/TSPTW"
PIPD_DIR="/Data04/yangzhihan/lzj/PIP-D baseline/POMO+PIP"


# ══════════════════════════════════════════════════════════════════════
# GPU 调度参数
# ══════════════════════════════════════════════════════════════════════
TOTAL_GPUS=8
NEED_GPUS=4          # SFT 和 RL 都用 4 张卡
EVAL_GPUS_SFT=4      # SFT eval 用 4 卡并行, evaluate.py 会 device_map=auto
EVAL_GPUS_RL=1       # RL eval 单卡
GPU_FREE_MEM_THRESHOLD_MB=500
FREE_GPUS=""

SFT_INIT_BATCH_SIZE=4   # SFT eval 初始 batch (evaluate.py batched generate)
RL_INIT_BATCH_SIZE=4    # RL eval 初始 batch
EVAL_NUM_TEST=10        # 每组测试实例数
EVAL_MAX_COMPLETION=10000

# ══════════════════════════════════════════════════════════════════════
# vLLM server (阶段 4 RL 训练用)
# ══════════════════════════════════════════════════════════════════════
VLLM_PORT=8000
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=5120
VLLM_DTYPE=bfloat16
VLLM_ENABLE_PREFIX_CACHING=True
if [ "$VLLM_ENABLE_PREFIX_CACHING" = "True" ]; then
    VLLM_PREFIX_CACHE_FLAG="--enable_prefix_caching"
else
    VLLM_PREFIX_CACHE_FLAG=""
fi
VLLM_STARTUP_TIMEOUT=300
VLLM_PID=""

TRL_BIN="$(dirname "$(which python)")/trl"

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
on_exit() {
    local exit_code=$?
    stop_vllm_server
    if [ "$ALL_DONE" != "1" ] && [ "$exit_code" != "0" ]; then
        notify "❌ UniCOP auto_all 异常退出" \
"退出码: $exit_code
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

# 拆第一张给 vLLM, 剩余给训练 (RL 阶段用)
split_gpus() {
    VLLM_GPU="${FREE_GPUS%%,*}"
    TRAIN_GPUS="${FREE_GPUS#*,}"
}

# ══════════════════════════════════════════════════════════════════════
# vLLM server 启停
# ══════════════════════════════════════════════════════════════════════
start_vllm_server() {
    local vllm_gpu=$1
    local port=$2
    local model_path=$3
    local log_file=$4

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 vLLM server | GPU=$vllm_gpu | port=$port | model=$model_path"

    PYTHONPATH="$REASON_DIR:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$vllm_gpu" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
        "$TRL_BIN" vllm-serve \
        --model "$model_path" \
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
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动失败,详见 $log_file"
            return 1
        fi
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ vLLM server 就绪 (pid=$VLLM_PID, 用时 ${waited}s)"
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动超时"
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

# ══════════════════════════════════════════════════════════════════════
# 阶段 1: SFT 训练 (4 卡, ZeRO-3 + LoRA + gc)
# 返回: 0=成功 / 99=OOM / 1=其他错误
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
    if grep -qiE "out of memory|OOM|CUDA out of memory|OutOfMemoryError" "$log_file"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ SFT OOM"
        return 99
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ SFT 非 OOM 错误 (exit=$ec)"
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
# 阶段 3 & 5: evaluate (OOM 时 batch_size 减半重试)
# ══════════════════════════════════════════════════════════════════════
run_eval() {
    local model_path="$1"
    local gpus="$2"
    local init_bs="$3"
    local tag="$4"          # sft / rl, 仅用于日志命名
    local problems="$5"     # "tsp cvrp vrptw tsptw"
    local sizes="$6"        # "20 50 100"
    local log_file="$LOG_DIR/eval_${tag}_${TIMESTAMP}.log"

    local bs=$init_bs
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] evaluate [$tag] | GPUs=$gpus | model=$model_path"
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
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ evaluate [$tag] 完成 (bs=$bs)"
            return 0
        fi

        if grep -qiE "out of memory|OOM|CUDA out of memory|OutOfMemoryError" "$log_file"; then
            bs=$((bs / 2))
            if [ "$bs" -ge 1 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ evaluate [$tag] OOM, batch_size 减半 → $bs, 等空闲卡后重试"
                sleep 5
                wait_for_gpus "$(echo "$gpus" | tr ',' '\n' | wc -l)"
                gpus="$FREE_GPUS"
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ evaluate [$tag] 非 OOM 错误 (exit=$ec)"
            return 1
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ evaluate [$tag] bs 降至 0 仍 OOM"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 阶段 4: RL 训练 (TSP n=20, vLLM server mode)
# 首卡 vLLM, 其余训练. 返回 0 / 99 / 1
# ══════════════════════════════════════════════════════════════════════
run_rl() {
    local gpus="$1"
    FREE_GPUS="$gpus"
    split_gpus    # 设置 VLLM_GPU / TRAIN_GPUS
    local total_proc
    total_proc=$(echo "$gpus" | tr ',' '\n' | wc -l)
    local train_proc=$((total_proc - 1))

    local vllm_log="$LOG_DIR/vllm_rl_${TIMESTAMP}.log"
    local train_log="$LOG_DIR/rl_${TIMESTAMP}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 阶段 4: RL 训练 TSP n=$RL_SIZE"
    echo "  base (= SFT merged): $SFT_MERGED"
    echo "  vLLM GPU: $VLLM_GPU    port=$VLLM_PORT    log=$vllm_log"
    echo "  训练 GPUs: $TRAIN_GPUS ($train_proc 进程)    log=$train_log"

    if ! start_vllm_server "$VLLM_GPU" "$VLLM_PORT" "$SFT_MERGED" "$vllm_log"; then
        return 1
    fi

    PYTHONPATH="$REASON_DIR:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
        python -m accelerate.commands.launch --num_processes "$train_proc" \
        "$REASON_DIR/train.py" \
        --problem "$RL_PROBLEM" \
        --problem_size "$RL_SIZE" \
        --num_train 2000 \
        --model "$SFT_MERGED" \
        --num_gpus "$train_proc" \
        --zero_stage 3 \
        --gradient_checkpointing \
        --output_dir "$RL_OUTPUT_BASE" \
        --pomo_ckpt_dir "$POMO_CKPT_DIR" \
        --pomo_baseline_dir "$POMO_BASELINE_DIR" \
        --pipd_ckpt_dir "$PIPD_CKPT_DIR" \
        --pipd_dir "$PIPD_DIR" \
        --vllm_server_host "localhost" \
        --vllm_server_port "$VLLM_PORT" \
        2>&1 | tee "$train_log"

    local ec=${PIPESTATUS[0]}
    stop_vllm_server

    if [ $ec -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ RL 训练完成"
        return 0
    fi
    if grep -qiE "out of memory|OOM|CUDA out of memory|OutOfMemoryError" "$train_log"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ RL OOM"
        return 99
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ RL 非 OOM 错误 (exit=$ec)"
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# RL LoRA merge (阶段 5 前置)
# ══════════════════════════════════════════════════════════════════════
run_rl_merge() {
    if [ ! -d "$RL_ADAPTER" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ RL adapter 不存在: $RL_ADAPTER"
        return 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 合并 RL LoRA → $RL_MERGED"
    python "$TOOLS_DIR/merge_lora.py" \
        --adapter "$RL_ADAPTER" \
        --base "$SFT_MERGED" \
        --output "$RL_MERGED" \
        --device cpu \
        2>&1 | tee "$LOG_DIR/rl_merge_${TIMESTAMP}.log"
    local ec=${PIPESTATUS[0]}
    if [ $ec -eq 0 ] && [ -f "$RL_MERGED/config.json" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ RL merge 完成"
        return 0
    fi
    return 1
}

# ══════════════════════════════════════════════════════════════════════
# 前置检查
# ══════════════════════════════════════════════════════════════════════
preflight() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 前置检查 =========="
    local fail=0
    for p in "$BASE_MODEL" "$SFT_DATA" "$DISTILL_DIR/train_sft.py" "$REASON_DIR/train.py" \
             "$REASON_DIR/evaluate.py" "$TOOLS_DIR/merge_lora.py" \
             "$POMO_BASELINE_DIR" "$POMO_CKPT_DIR"; do
        if [ ! -e "$p" ]; then
            echo "  ✗ 路径不存在: $p"
            fail=1
        else
            echo "  ✓ $p"
        fi
    done
    if [ ! -x "$TRL_BIN" ]; then
        echo "  ✗ TRL binary 未找到: $TRL_BIN (pip install 'trl[vllm]==1.2.0')"
        fail=1
    else
        echo "  ✓ TRL: $TRL_BIN"
    fi
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
echo "  SFT out:    $SFT_OUT"
echo "  SFT merged: $SFT_MERGED"
echo "  RL adapter: $RL_ADAPTER"
echo "  RL merged:  $RL_MERGED"
echo "  Logs:       $LOG_DIR"
echo ""

notify "🚀 UniCOP auto_all 启动" \
"流程: SFT → merge → SFT eval → RL(TSP n=$RL_SIZE) → RL eval
启动: $(date '+%Y-%m-%d %H:%M:%S')
SFT out: $SFT_OUT"

# ── 阶段 1: SFT 训练 (已跳过，复用已有产物) ──────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [1/5] SFT 训练 — 跳过，复用 $SFT_OUT ═══"
if [ ! -d "$SFT_OUT/final_model" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ SFT adapter 不存在: $SFT_OUT/final_model"
    exit 1
fi

# ── 阶段 2: SFT LoRA merge (CPU,无需 GPU) ─────────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [2/5] SFT LoRA merge ═══"
if ! run_sft_merge; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ SFT merge 失败,退出"
    notify "❌ UniCOP SFT merge 失败" "详见 $LOG_DIR/sft_merge_${TIMESTAMP}.log"
    exit 1
fi

# ── 阶段 3: SFT evaluate (4 卡, 12 组) ────────────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [3/5] SFT evaluate (全套 4×3) ═══"
wait_for_gpus $EVAL_GPUS_SFT
run_eval "$SFT_MERGED" "$FREE_GPUS" "$SFT_INIT_BATCH_SIZE" "sft" \
    "tsp cvrp vrptw tsptw" "20 50 100"
# eval 失败不阻塞后续 RL
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ SFT eval 失败,继续 RL 阶段"
    notify "⚠️ UniCOP SFT eval 失败" "不阻塞, 继续 RL"
fi

# ── 阶段 4: RL 训练 (TSP n=20) ─────────────────────────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [4/5] RL 训练 TSP n=$RL_SIZE ═══"
wait_for_gpus $NEED_GPUS
run_rl "$FREE_GPUS"
ec=$?
if [ $ec -eq 99 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RL OOM, 等另一批 4 张空闲卡重试"
    wait_for_gpus $NEED_GPUS
    run_rl "$FREE_GPUS"
    ec=$?
fi
if [ $ec -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ RL 最终失败 (ec=$ec),退出"
    notify "❌ UniCOP RL 失败" "exit=$ec, 详见 $LOG_DIR/rl_${TIMESTAMP}.log"
    exit 1
fi

# ── 阶段 5: RL evaluate (先 merge, 再 1 卡 eval) ──────────────────
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ═══ [5/5] RL evaluate ═══"
if ! run_rl_merge; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ RL merge 失败, 跳过 RL eval"
    notify "⚠️ UniCOP RL merge 失败" "跳过 eval"
else
    wait_for_gpus $EVAL_GPUS_RL
    run_eval "$RL_MERGED" "$FREE_GPUS" "$RL_INIT_BATCH_SIZE" "rl" \
        "$RL_PROBLEM" "$RL_SIZE"
fi

# ══════════════════════════════════════════════════════════════════════
# 完成
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 全部流程结束 =========="
echo "  SFT 产物:     $SFT_OUT"
echo "  SFT 合并:     $SFT_MERGED"
echo "  RL 产物:      $RL_ADAPTER"
echo "  RL 合并:      $RL_MERGED"
echo "  eval 结果:    $EVAL_RESULT_DIR"
echo "  日志目录:     $LOG_DIR"

notify "✅ UniCOP auto_all 全部完成" \
"结束: $(date '+%Y-%m-%d %H:%M:%S')
SFT: $SFT_MERGED
RL:  $RL_MERGED
eval: $EVAL_RESULT_DIR"

ALL_DONE=1
