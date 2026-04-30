#!/bin/bash
# R1-Distill 自举 rationalization → SFT 一键流水线
#
# 流程：
#   1. 在 4 张卡上各启动一个 vLLM 服务器（带 ngram）
#   2. 读取现有 LKH 解，4 路并行生成 rationalization 数据
#   3. 停止所有 vLLM 服务器（释放 GPU）
#   4. SFT 训练（R1-Distill + LoRA，4 卡 ZeRO-3）
#   5. 合并 LoRA adapter
#
# 使用方法：
#   bash auto_self_rationalize.sh
#   nohup bash auto_self_rationalize.sh > self_rationalize.log 2>&1 &

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────────
DISTILL_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="/home/ntu/lzj/Model/model/DeepSeek-R1-Distill-Qwen-7B"
SOLUTIONS_FILE="data/solutions_cvrp20.jsonl"
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=20
NUM_SAMPLES=0
OUTPUT_DIR="output_sft_self_rationalize_cvrp20"
CHAINS_FILE="data/chains_self_cvrp${SIZE}.jsonl"

# vLLM 配置（4 卡并行）
NUM_GPUS=4
VLLM_BASE_PORT=8000
NGRAM_SIZE=6
# max-model-len = prompt + completion 的上限，设大一点保险；输出由 max_tokens=4096 限制
VLLM_MAX_MODEL_LEN=8192

# SFT 配置
SFT_LR=2e-5
SFT_EPOCHS=3
SFT_LORA_RANK=64
SFT_LORA_ALPHA=128
SFT_MAX_LENGTH=4096
SFT_MAX_TOKENS=4096

# CUDA
export CUDA_HOME=/home/ntu/anaconda3/envs/unicop
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

cd "$DISTILL_DIR"

echo "============================================================"
echo "  R1-Distill 自举 Rationalization → SFT"
echo "  模型:       $MODEL_PATH"
echo "  Solutions:  $SOLUTIONS_FILE"
echo "  问题:       ${PROBLEM^^} n=$SIZE"
echo "  生成数量:   全部 (NUM_SAMPLES=0 表示使用全部数据)"
echo "  并行 GPU:   $NUM_GPUS"
echo "  输出:       $OUTPUT_DIR"
echo "  时间:       $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 1: 启动 4 个 vLLM 服务器
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: 启动 $NUM_GPUS 个 vLLM 服务器..."

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((VLLM_BASE_PORT + gpu))
    CUDA_VISIBLE_DEVICES=$gpu python "$DISTILL_DIR/vllm_serve_ngram.py" \
        --model "$MODEL_PATH" \
        --port $port \
        --no_repeat_ngram_size $NGRAM_SIZE \
        --dtype bfloat16 \
        --max-model-len $VLLM_MAX_MODEL_LEN &
    VLLM_PIDS+=($!)
    echo "  GPU $gpu → :${port} (PID=${VLLM_PIDS[-1]})"
done

echo "  等待所有服务器就绪..."
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    wait_for_vllm_port $((VLLM_BASE_PORT + gpu))
done
echo "  全部就绪!"

# ══════════════════════════════════════════════════════════════════
# Step 2: 并行生成 rationalization 数据
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 生成 rationalization 数据 ($NUM_GPUS 路并行)..."

VLLM_URLS=""
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    VLLM_URLS="$VLLM_URLS http://localhost:$((VLLM_BASE_PORT + gpu))/v1"
done

python rationalize_solutions.py \
    --solutions "$SOLUTIONS_FILE" \
    --vllm_urls $VLLM_URLS \
    --output "$CHAINS_FILE" \
    --problem $PROBLEM \
    --size $SIZE \
    --num_samples $NUM_SAMPLES \
    --max_tokens $SFT_MAX_TOKENS \
    --concurrency $((NUM_GPUS * 8))

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
echo ">>> Step 4: SFT 训练..."

accelerate launch --num_processes $NUM_GPUS --main_process_port 29600 \
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

notify "自举 Rationalize 全部完成" "模型: $OUTPUT_DIR, 数据: $ACTUAL_COUNT 条"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  数据:   $CHAINS_FILE ($ACTUAL_COUNT 条)"
echo "  模型:   $OUTPUT_DIR/final_model"
echo "============================================================"
