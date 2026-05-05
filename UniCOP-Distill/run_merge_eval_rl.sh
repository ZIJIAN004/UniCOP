#!/bin/bash
# run_merge_eval_rl.sh — Template SFT 产物的后续流水线：合并 → 评估 → RL
#
# 前提：output_sft_template_cvrp20/final_model 下存在 adapter 文件
#
# 三步流程：
#   Step 1: 合并 LoRA adapter → 完整模型
#   Step 2: 评估合并后模型（CVRP n=20, 本地 HF 推理）
#   Step 3: GRPO RL 训练（vLLM server + ZeRO-3）
#
# SBATCH 提交：
#   sbatch run_merge_eval_rl.sh
# 或手动运行：
#   bash run_merge_eval_rl.sh

#SBATCH --qos express
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/merge_eval_rl_%j.log

set -euo pipefail

# ── 环境初始化 ──────────────────────────────────────────────────────────────
export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$(dirname "$_SELF_DIR")/paths.sh"

# DeepSpeed 必须 export CUDA_HOME
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$DISTILL_DIR"

LOG_FILE="$DISTILL_DIR/merge_eval_rl_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# ── 配置 ────────────────────────────────────────────────────────────────────
PROBLEM="cvrp"
SIZE=20
ADAPTER_DIR="output_sft_template_${PROBLEM}${SIZE}/checkpoint-2000"
MERGED_MODEL="$DISTILL_DIR/$ADAPTER_DIR"

# RL 配置
RL_NUM_TRAIN=2000
RL_NUM_GPUS=3            # 训练进程数（不含 vLLM 的 1 张）
RL_ZERO_STAGE=3
RL_OUTPUT_DIR="$REASON_DIR/output_rl_template"
VLLM_PORT=8010
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=5120
VLLM_NGRAM_SIZE=6

# 评估配置
EVAL_NUM_TEST=100
EVAL_MAX_COMPLETION=10000
EVAL_BATCH_SIZE=4
EVAL_SAVE_DIR="$DISTILL_DIR/eval_results_template"

# Server 酱
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

# ── 工具函数 ────────────────────────────────────────────────────────────────
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

VLLM_PID=""
stop_vllm_server() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] 关闭 vLLM server (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}

COMPLETED=0
on_exit() {
    local exit_code=$?
    stop_vllm_server
    if [ "$COMPLETED" != "1" ] && [ "$exit_code" != "0" ]; then
        notify "❌ merge_eval_rl 失败 (line $LINENO, exit=$exit_code)"
    fi
}
trap 'on_exit' EXIT INT TERM

echo "============================================================"
echo "  Template CVRP20: Merge → Eval → RL"
echo "  Adapter:     $ADAPTER_DIR"
echo "  RL output:   $RL_OUTPUT_DIR"
echo "  RL GPUs:     1 vLLM + $RL_NUM_GPUS train (ZeRO-$RL_ZERO_STAGE)"
echo "  时间:        $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 合并 LoRA adapter
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: 合并 LoRA adapter..."

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    # 可能已经合并过（无 adapter_config.json 但有 config.json）
    if [ -f "$ADAPTER_DIR/config.json" ]; then
        echo "  已是完整模型（config.json 存在，无 adapter_config.json），跳过合并"
    else
        echo "ERROR: $ADAPTER_DIR 下既无 adapter_config.json 也无 config.json"
        exit 1
    fi
else
    python stage1_solution/merge_adapter.py \
        --adapter_path "$ADAPTER_DIR"
    echo "  ✓ 合并完成: $ADAPTER_DIR"
fi

notify "Step1 完成: adapter 合并"

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 评估合并后模型
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 评估合并后模型..."
echo "  模型:    $MERGED_MODEL"
echo "  测试数:  $EVAL_NUM_TEST"
echo "  保存:    $EVAL_SAVE_DIR"

mkdir -p "$EVAL_SAVE_DIR"

EVAL_GPU_LIST=($(wait_for_free_gpus 1))
EVAL_GPU="${EVAL_GPU_LIST[0]}"
echo "  使用 GPU: $EVAL_GPU"

cd "$REASON_DIR"

CUDA_VISIBLE_DEVICES=$EVAL_GPU python evaluate.py \
    --backend local \
    --model_path "$MERGED_MODEL" \
    --problem $PROBLEM \
    --problem_size $SIZE \
    --model_type reasoning \
    --max_completion_length $EVAL_MAX_COMPLETION \
    --num_test $EVAL_NUM_TEST \
    --num_samples 1 \
    --batch_size $EVAL_BATCH_SIZE \
    --prompt_mode think \
    --save_dir "$EVAL_SAVE_DIR"

notify "Step2 完成: 评估结束" "结果在 $EVAL_SAVE_DIR"
echo "  ✓ 评估完成"

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: GRPO RL 训练
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: GRPO RL 训练..."
echo "  基座模型:  $MERGED_MODEL"
echo "  问题:      $PROBLEM n=$SIZE"
echo "  训练样本:  $RL_NUM_TRAIN"
echo "  输出:      $RL_OUTPUT_DIR"

cd "$REASON_DIR"

TOTAL_RL_GPUS=$((RL_NUM_GPUS + 1))
echo "  等待 $TOTAL_RL_GPUS 张空闲 GPU (1 vLLM + $RL_NUM_GPUS train)..."
RL_GPU_LIST=($(wait_for_free_gpus $TOTAL_RL_GPUS))

VLLM_GPU="${RL_GPU_LIST[0]}"
TRAIN_GPUS=$(IFS=,; echo "${RL_GPU_LIST[*]:1}")
echo "  vLLM GPU:  $VLLM_GPU (port=$VLLM_PORT)"
echo "  训练 GPUs: $TRAIN_GPUS ($RL_NUM_GPUS 进程)"

# ── 启动 vLLM server ──────────────────────────────────────────────────────
VLLM_LOG="$REASON_DIR/logs/vllm_template_rl_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$REASON_DIR/logs"

echo "[$(date '+%H:%M:%S')] 启动 vLLM server..."

PYTHONPATH="$REASON_DIR:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
CUDA_HOME="$CUDA_HOME" \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
    python "$REASON_DIR/utils/vllm_serve_ngram.py" \
    --no_repeat_ngram_size "$VLLM_NGRAM_SIZE" \
    --model "$MERGED_MODEL" \
    --tensor_parallel_size 1 \
    --port "$VLLM_PORT" \
    --gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
    --max_model_len "$VLLM_MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --enable_prefix_caching \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# 等待 vLLM 就绪
VLLM_TIMEOUT=300
waited=0
while [ "$waited" -lt "$VLLM_TIMEOUT" ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  ✗ vLLM server 启动失败，详见 $VLLM_LOG"
        exit 1
    fi
    if curl -s "http://localhost:${VLLM_PORT}/health/" > /dev/null 2>&1; then
        echo "  ✓ vLLM server 就绪 (pid=$VLLM_PID, 用时 ${waited}s)"
        break
    fi
    sleep 3
    waited=$((waited + 3))
done
if [ "$waited" -ge "$VLLM_TIMEOUT" ]; then
    echo "  ✗ vLLM server 启动超时 (${VLLM_TIMEOUT}s)"
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

# ── 启动 GRPO 训练 ──────────────────────────────────────────────────────────
TRAIN_LOG="$REASON_DIR/logs/train_template_rl_$(date '+%Y%m%d_%H%M%S').log"

echo "[$(date '+%H:%M:%S')] 启动 GRPO 训练..."

PYTHONPATH="$REASON_DIR:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_HOME="$CUDA_HOME" \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    python -m accelerate.commands.launch \
    --num_processes "$RL_NUM_GPUS" \
    "$REASON_DIR/train.py" \
    --problem "$PROBLEM" \
    --problem_size "$SIZE" \
    --num_train "$RL_NUM_TRAIN" \
    --model "$MERGED_MODEL" \
    --num_gpus "$RL_NUM_GPUS" \
    --zero_stage "$RL_ZERO_STAGE" \
    --gradient_checkpointing \
    --output_dir "$RL_OUTPUT_DIR" \
    --pomo_ckpt_dir "$POMO_CKPT_DIR" \
    --pomo_baseline_dir "$POMO_BASELINE_DIR" \
    --pipd_ckpt_dir "$PIPD_CKPT_DIR" \
    --pipd_dir "$PIPD_DIR" \
    --vllm_server_host "localhost" \
    --vllm_server_port "$VLLM_PORT" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

stop_vllm_server

if [ $TRAIN_EXIT -ne 0 ]; then
    notify "❌ GRPO 训练失败 (exit=$TRAIN_EXIT)" "日志: $TRAIN_LOG"
    exit $TRAIN_EXIT
fi

echo "  ✓ GRPO 训练完成"

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 训练后评估（RL 产物）
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 4: 评估 RL 产物..."

RL_MODEL="$RL_OUTPUT_DIR/${PROBLEM}_n${SIZE}/final_model"

if [ ! -d "$RL_MODEL" ]; then
    echo "  ⚠️ RL 模型目录不存在: $RL_MODEL，跳过后评估"
else
    EVAL_GPU_LIST2=($(wait_for_free_gpus 1))
    EVAL_GPU2="${EVAL_GPU_LIST2[0]}"

    CUDA_VISIBLE_DEVICES=$EVAL_GPU2 python "$REASON_DIR/evaluate.py" \
        --backend local \
        --model_path "$RL_MODEL" \
        --problem $PROBLEM \
        --problem_size $SIZE \
        --model_type reasoning \
        --max_completion_length $EVAL_MAX_COMPLETION \
        --num_test $EVAL_NUM_TEST \
        --num_samples 1 \
        --batch_size $EVAL_BATCH_SIZE \
        --prompt_mode think \
        --save_dir "$EVAL_SAVE_DIR"

    echo "  ✓ RL 后评估完成"
fi

# ══════════════════════════════════════════════════════════════════════════════
COMPLETED=1

notify "✅ Template CVRP20 全流程完成" \
"Merge → Eval → RL → Eval
RL 模型: $RL_MODEL
评估目录: $EVAL_SAVE_DIR
完成时间: $(date)"

echo ""
echo "============================================================"
echo "  全流程完成! $(date)"
echo "  合并模型:    $MERGED_MODEL"
echo "  RL 模型:     $RL_MODEL"
echo "  评估结果:    $EVAL_SAVE_DIR/"
echo "  训练日志:    $TRAIN_LOG"
echo "============================================================"
