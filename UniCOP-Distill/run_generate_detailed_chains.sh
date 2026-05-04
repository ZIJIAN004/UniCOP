#!/bin/bash
# 全量生成 detailed prompt rationalization 数据 — CVRP20
#
# 流程：
#   1. 计算 max prompt length (detailed prompt 含 few-shot，更长)
#   2. Pilot 30 条校准 output 长度
#   3. 全量生成
#   4. 过滤
#
# 使用方法：
#   nohup bash run_generate_detailed_chains.sh > gen_detailed.log 2>&1 &

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$_SELF_DIR/gen_detailed_$(date '+%Y%m%d_%H%M%S').log"
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
CHAINS_FILE="data/chains_detailed_${PROBLEM}${SIZE}.jsonl"
CHAINS_FILTERED="data/chains_detailed_${PROBLEM}${SIZE}_filtered.jsonl"
FILTERED_IDS="data/chains_detailed_${PROBLEM}${SIZE}_filtered_ids.txt"

# vLLM 配置
VLLM_BASE_PORT=8100
NGRAM_SIZE=6

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
    echo "    (:${port}) 就绪 (${waited}s)"
}

find_free_port() {
    local port=$1
    while ss -tlnp 2>/dev/null | grep -q ":${port} " || \
          curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

start_vllm_servers() {
    local model_len=$1
    FREE_GPU_LIST=($(get_free_gpus))
    NUM_GPUS=${#FREE_GPU_LIST[@]}
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "ERROR: 没有空闲 GPU"
        exit 1
    fi
    echo "  检测到 $NUM_GPUS 张空闲 GPU (${FREE_GPU_LIST[*]})"

    VLLM_LOG_DIR="$DISTILL_DIR/vllm_logs"
    mkdir -p "$VLLM_LOG_DIR"

    VLLM_PORTS=()
    local next_port=$VLLM_BASE_PORT
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        gpu=${FREE_GPU_LIST[$i]}
        next_port=$(find_free_port $next_port)
        VLLM_PORTS+=($next_port)
        CUDA_VISIBLE_DEVICES=$gpu python "$DISTILL_DIR/vllm_serve_ngram.py" \
            --model "$MODEL_PATH" \
            --port $next_port \
            --no_repeat_ngram_size $NGRAM_SIZE \
            --dtype bfloat16 \
            --max-model-len "$model_len" \
            --gpu-memory-utilization 0.95 \
            --enable-prefix-caching \
            --disable-log-requests \
            --disable-log-stats \
            > "$VLLM_LOG_DIR/detailed_gpu${gpu}.log" 2>&1 &
        VLLM_PIDS+=($!)
        echo "  GPU $gpu → :${next_port} (PID=${VLLM_PIDS[-1]})"
        next_port=$((next_port + 1))
    done

    echo "  等待所有服务器就绪..."
    for port in "${VLLM_PORTS[@]}"; do
        wait_for_vllm_port $port
    done
    echo "  全部就绪!"

    VLLM_URLS=""
    for port in "${VLLM_PORTS[@]}"; do
        VLLM_URLS="$VLLM_URLS http://localhost:${port}/v1"
    done
}

trap 'notify "Detailed chains 生成失败: line $LINENO"; cleanup_all_vllm' ERR

cd "$DISTILL_DIR"

echo "============================================================"
echo "  全量生成 Detailed Rationalization — CVRP${SIZE}"
echo "  模型:       $MODEL_PATH"
echo "  Solutions:  $SOLUTIONS_FILE"
echo "  prompt:     detailed (逐步构建 + few-shot)"
echo "  时间:       $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 0: 检查 solutions 文件
# ══════════════════════════════════════════════════════════════════
if [ ! -f "$SOLUTIONS_FILE" ]; then
    echo "ERROR: $SOLUTIONS_FILE 不存在，请先运行 generate_solutions.py"
    exit 1
fi
SOLUTIONS_COUNT=$(grep -c '^{' "$SOLUTIONS_FILE" 2>/dev/null || echo 0)
echo "  Solutions: $SOLUTIONS_COUNT 条"

# ══════════════════════════════════════════════════════════════════
# Step 1: 计算 max prompt length
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: 计算 max prompt length (detailed prompt)..."
MAX_PROMPT_LEN=$(python rationalize_solutions.py \
    --solutions "$SOLUTIONS_FILE" \
    --problem $PROBLEM --size $SIZE \
    --max_tokens 0 \
    --tokenizer "$MODEL_PATH" \
    --prompt_style detailed \
    --calc_max_model_len)
echo "  max prompt length = $MAX_PROMPT_LEN"

# ══════════════════════════════════════════════════════════════════
# Step 2: Pilot — 生成 30 条校准 output 长度
# ══════════════════════════════════════════════════════════════════
PILOT_SAMPLES=30
PILOT_MODEL_LEN=$((MAX_PROMPT_LEN + 4096))
PILOT_FILE="data/chains_pilot_${PROBLEM}${SIZE}.jsonl"

PILOT_COUNT=$(grep -c '^{' "$PILOT_FILE" 2>/dev/null || echo 0)
if [ "$PILOT_COUNT" -ge 10 ]; then
    echo ""
    echo ">>> Step 2: Pilot 已存在 ($PILOT_COUNT 条)，跳过生成，直接计算 P95..."
else
    echo ""
    echo ">>> Step 2: Pilot 生成 $PILOT_SAMPLES 条 (max-model-len=$PILOT_MODEL_LEN)..."
    start_vllm_servers $PILOT_MODEL_LEN

    python rationalize_solutions.py \
        --solutions "$SOLUTIONS_FILE" \
        --vllm_urls $VLLM_URLS \
        --output "$PILOT_FILE" \
        --problem $PROBLEM --size $SIZE \
        --num_samples $PILOT_SAMPLES \
        --max_tokens 4096 \
        --prompt_style detailed \
        --concurrency $((NUM_GPUS * 4))

    echo "  停止 pilot vLLM..."
    cleanup_all_vllm
fi

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
echo "  校准后 vLLM max-model-len = $VLLM_MAX_MODEL_LEN"
notify "Pilot 完成: P95=$P95_OUTPUT, model_len=$VLLM_MAX_MODEL_LEN"

# ══════════════════════════════════════════════════════════════════
# Step 3: 全量生成
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: 全量生成 (max-model-len=$VLLM_MAX_MODEL_LEN)..."
start_vllm_servers $VLLM_MAX_MODEL_LEN

python rationalize_solutions.py \
    --solutions "$SOLUTIONS_FILE" \
    --vllm_urls $VLLM_URLS \
    --output "$CHAINS_FILE" \
    --problem $PROBLEM --size $SIZE \
    --num_samples 0 \
    --max_tokens $P95_OUTPUT \
    --prompt_style detailed \
    --concurrency $((NUM_GPUS * 32))

ACTUAL_COUNT=$(grep -c '^{' "$CHAINS_FILE" 2>/dev/null || echo 0)
echo "  实际生成: $ACTUAL_COUNT 条"

echo ""
echo ">>> 停止 vLLM..."
cleanup_all_vllm

notify "全量生成完成: $ACTUAL_COUNT 条"

# ══════════════════════════════════════════════════════════════════
# Step 4: 过滤
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 4: 过滤..."

python filter_chains.py \
    --input "$CHAINS_FILE" \
    --output "$CHAINS_FILTERED" \
    --ids_output "$FILTERED_IDS" \
    --min_coverage 0.8

FILTERED_COUNT=$(grep -c '^{' "$CHAINS_FILTERED" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  总生成: $ACTUAL_COUNT 条"
echo "  过滤后: $FILTERED_COUNT 条"
echo "  输出:   $CHAINS_FILE"
echo "  过滤:   $CHAINS_FILTERED"
echo "  ID:     $FILTERED_IDS"
echo "============================================================"

notify "Detailed chains 全部完成" "总: $ACTUAL_COUNT, 过滤后: $FILTERED_COUNT"
