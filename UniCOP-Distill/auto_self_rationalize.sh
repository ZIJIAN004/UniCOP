#!/bin/bash
# R1-Distill 自举 rationalization → SFT 一键流水线
#
# 流程：
#   1. 检测空闲 GPU，在每张空闲卡上启动一个 vLLM 服务器（带 ngram）
#   2. 读取现有 LKH 解，多路并行生成 rationalization 数据
#   3. 停止所有 vLLM 服务器（释放 GPU）
#   4. 等待至少 4 张空闲 GPU，SFT 训练（R1-Distill + LoRA，ZeRO-3）
#   5. 合并 LoRA adapter
#   6. 评估
#
# 使用方法：
#   bash auto_self_rationalize.sh
#   nohup bash auto_self_rationalize.sh > self_rationalize.log 2>&1 &

set -euo pipefail

# ── 自动 log ─────────────────────────────────────────────────────────────────
DISTILL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$DISTILL_DIR/self_rationalize_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# Python stdout 不缓冲，确保 print 立即写入 log
export PYTHONUNBUFFERED=1

# ── 配置 ──────────────────────────────────────────────────────────────────────

MODEL_PATH="/Data04/yangzhihan/lzj/UniCOP-Reason.bak_/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SOLUTIONS_FILE="data/solutions_tsp20.jsonl"
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="tsp"
SIZE=20
NUM_SAMPLES=0
OUTPUT_DIR="output_sft_self_rationalize_tsp20"
CHAINS_FILE="data/chains_self_${PROBLEM}${SIZE}.jsonl"

# vLLM 配置
VLLM_BASE_PORT=8100
NGRAM_SIZE=6

# SFT 配置
SFT_LR=2e-5
SFT_EPOCHS=3
SFT_LORA_RANK=64
SFT_LORA_ALPHA=128

# LKH 求解器（TSP 解生成）
export LKH_BIN=/Data04/yangzhihan/lzj/LKH-3.0.9/LKH

# CUDA
export CUDA_HOME=/Data04/yangzhihan/envs/lzj_env
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 工具函数 ──────────────────────────────────────────────────────────────────
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s "https://sctapi.ftqq.com/$SCKEY.send" \
        -d "title=$title" -d "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

VLLM_PIDS=()

cleanup_all_vllm() {
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${VLLM_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    VLLM_PIDS=()
    echo "  所有 vLLM 已停止"
    sleep 5
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

trap 'notify "自举 Rationalize 失败: line $LINENO"; cleanup_all_vllm' ERR

wait_for_vllm_port() {
    local port=$1
    local max_wait=300
    local waited=0
    while ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "ERROR: vLLM :${port} 启动超时"
            exit 1
        fi
    done
    echo "    GPU $((port - VLLM_BASE_PORT)) (:${port}) 就绪 (${waited}s)"
}

start_vllm_servers() {
    local model_len=$1
    FREE_GPU_LIST=($(get_free_gpus))
    NUM_GPUS=${#FREE_GPU_LIST[@]}
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "ERROR: 没有空闲 GPU，无法启动 vLLM"
        exit 1
    fi
    echo "  检测到 $NUM_GPUS 张空闲 GPU (${FREE_GPU_LIST[*]})"

    VLLM_LOG_DIR="$DISTILL_DIR/vllm_logs"
    mkdir -p "$VLLM_LOG_DIR"

    for i in $(seq 0 $((NUM_GPUS - 1))); do
        gpu=${FREE_GPU_LIST[$i]}
        port=$((VLLM_BASE_PORT + i))
        CUDA_VISIBLE_DEVICES=$gpu python "$DISTILL_DIR/vllm_serve_ngram.py" \
            --model "$MODEL_PATH" \
            --port $port \
            --no_repeat_ngram_size $NGRAM_SIZE \
            --dtype bfloat16 \
            --max-model-len "$model_len" \
            --gpu-memory-utilization 0.95 \
            --enable-prefix-caching \
            --disable-log-requests \
            --disable-log-stats \
            > "$VLLM_LOG_DIR/gpu${gpu}.log" 2>&1 &
        VLLM_PIDS+=($!)
        echo "  GPU $gpu → :${port} (PID=${VLLM_PIDS[-1]})"
    done

    echo "  等待所有服务器就绪..."
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        wait_for_vllm_port $((VLLM_BASE_PORT + i))
    done
    echo "  全部就绪!"

    VLLM_URLS=""
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        VLLM_URLS="$VLLM_URLS http://localhost:$((VLLM_BASE_PORT + i))/v1"
    done
}

cd "$DISTILL_DIR"

echo "============================================================"
echo "  R1-Distill 自举 Rationalization → SFT"
echo "  模型:       $MODEL_PATH"
echo "  Solutions:  $SOLUTIONS_FILE"
echo "  问题:       ${PROBLEM^^} n=$SIZE"
echo "  生成数量:   全部 (NUM_SAMPLES=0 表示使用全部数据)"
echo "  并行 GPU:   动态检测空闲卡"
echo "  输出:       $OUTPUT_DIR"
echo "  时间:       $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 0a: 生成 solutions 文件（如果不存在）
# ══════════════════════════════════════════════════════════════════
TARGET_SAMPLES=50000

echo ""
echo ">>> Step 0a: 检查 solutions 文件..."
if [ -f "$SOLUTIONS_FILE" ]; then
    EXISTING=$(grep -c '^{' "$SOLUTIONS_FILE" 2>/dev/null || echo 0)
    echo "  已有 $EXISTING 条样本"
    if [ "$EXISTING" -ge "$TARGET_SAMPLES" ]; then
        echo "  样本数已达标，跳过数据生成"
    else
        echo "  样本不足，继续生成 (断点续传)..."
        python stage1_solution/generate_solutions.py \
            --problems $PROBLEM \
            --sizes $SIZE \
            --num_samples $TARGET_SAMPLES \
            --output "$SOLUTIONS_FILE" \
            --workers 32
    fi
else
    echo "  数据文件不存在，开始生成..."
    python stage1_solution/generate_solutions.py \
        --problems $PROBLEM \
        --sizes $SIZE \
        --num_samples $TARGET_SAMPLES \
        --output "$SOLUTIONS_FILE" \
        --workers 32
fi
notify "Step0a 完成: ${PROBLEM^^}${SIZE} 数据生成"

# ══════════════════════════════════════════════════════════════════
# Step 0b: 计算 max prompt length
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 0b: 计算 max prompt length..."
MAX_PROMPT_LEN=$(python rationalize_solutions.py \
    --solutions "$SOLUTIONS_FILE" \
    --problem $PROBLEM --size $SIZE \
    --max_tokens 0 \
    --tokenizer "$MODEL_PATH" \
    --calc_max_model_len)
echo "  max prompt length (aligned 256) = $MAX_PROMPT_LEN"

# ══════════════════════════════════════════════════════════════════
# Step 1: Pilot — 生成 30 条校准 output 长度
# ══════════════════════════════════════════════════════════════════
PILOT_SAMPLES=30
PILOT_MODEL_LEN=$((MAX_PROMPT_LEN + 4096))
PILOT_FILE="data/chains_pilot_${PROBLEM}${SIZE}.jsonl"

echo ""
echo ">>> Step 1: Pilot 生成 $PILOT_SAMPLES 条 (max-model-len=$PILOT_MODEL_LEN)..."
start_vllm_servers $PILOT_MODEL_LEN

python rationalize_solutions.py \
    --solutions "$SOLUTIONS_FILE" \
    --vllm_urls $VLLM_URLS \
    --output "$PILOT_FILE" \
    --problem $PROBLEM \
    --size $SIZE \
    --num_samples $PILOT_SAMPLES \
    --max_tokens 4096 \
    --concurrency $((NUM_GPUS * 4))

echo "  停止 pilot vLLM..."
cleanup_all_vllm

P95_OUTPUT=$(python -c "
import json, math
tokens = []
with open('$PILOT_FILE') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        t = r.get('output_tokens', 0)
        if t: tokens.append(t)
if not tokens:
    print(4096)
else:
    tokens.sort()
    idx = min(int(math.ceil(0.95 * len(tokens))) - 1, len(tokens) - 1)
    print(tokens[idx])
")
echo "  Pilot P95 output tokens = $P95_OUTPUT"

VLLM_MAX_MODEL_LEN=$(( ((MAX_PROMPT_LEN + P95_OUTPUT + 255) / 256) * 256 ))
SFT_MAX_LENGTH=$VLLM_MAX_MODEL_LEN
echo "  校准后 vLLM max-model-len = $VLLM_MAX_MODEL_LEN"
echo "  SFT max-length            = $SFT_MAX_LENGTH"
notify "Step1 完成: P95=$P95_OUTPUT, model_len=$VLLM_MAX_MODEL_LEN"

# ══════════════════════════════════════════════════════════════════
# Step 2: 全量生成 rationalization 数据
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 全量生成 rationalization 数据 (max-model-len=$VLLM_MAX_MODEL_LEN)..."
start_vllm_servers $VLLM_MAX_MODEL_LEN

python rationalize_solutions.py \
    --solutions "$SOLUTIONS_FILE" \
    --vllm_urls $VLLM_URLS \
    --output "$CHAINS_FILE" \
    --problem $PROBLEM \
    --size $SIZE \
    --num_samples $NUM_SAMPLES \
    --max_tokens $P95_OUTPUT \
    --concurrency $((NUM_GPUS * 16))

ACTUAL_COUNT=$(grep -c '^{' "$CHAINS_FILE" 2>/dev/null || echo 0)
echo "  实际生成: $ACTUAL_COUNT 条"
notify "Step2 完成: $ACTUAL_COUNT 条 rationalization"

# ══════════════════════════════════════════════════════════════════
# Step 3: 停止所有 vLLM 服务器
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: 停止 vLLM，释放 GPU..."
cleanup_all_vllm

# ══════════════════════════════════════════════════════════════════
# Step 4: SFT 训练
# ══════════════════════════════════════════════════════════════════
echo ""
SFT_NUM_GPUS=4
echo ">>> Step 4: SFT 训练（等待至少 ${SFT_NUM_GPUS} 张空闲 GPU）..."

SFT_GPU_LIST=($(wait_for_free_gpus $SFT_NUM_GPUS))
SFT_CUDA_DEVICES=$(IFS=,; echo "${SFT_GPU_LIST[*]}")
echo "  使用 GPU: $SFT_CUDA_DEVICES"

CUDA_VISIBLE_DEVICES=$SFT_CUDA_DEVICES accelerate launch --num_processes $SFT_NUM_GPUS --main_process_port 29600 \
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
    --batch_size 1 \
    --grad_accum 8 \
    --lr $SFT_LR \
    --save_steps 500

notify "Step4 完成: SFT 训练"

# ══════════════════════════════════════════════════════════════════
# Step 5: 合并 LoRA adapter
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 5: 合并 LoRA adapter..."

python stage1_solution/merge_adapter.py \
    --adapter_path "$OUTPUT_DIR/final_model"

notify "Step5 完成: adapter 合并"

# ══════════════════════════════════════════════════════════════════
# Step 6: 评估
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 6: 评估..."

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

notify "自举 Rationalize 全部完成" "模型: $OUTPUT_DIR, 数据: $ACTUAL_COUNT 条, 评估结果在 eval_results/"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  数据:   $CHAINS_FILE ($ACTUAL_COUNT 条)"
echo "  模型:   $OUTPUT_DIR/final_model"
echo "  评估:   eval_results/"
echo "============================================================"
