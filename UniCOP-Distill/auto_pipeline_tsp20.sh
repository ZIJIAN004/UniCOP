#!/bin/bash
# TSP n=20 全自动流水线：数据生成 → Stage1 SFT → Eval → Stage2 SFT → Eval
#
# 使用方法:
#   bash auto_pipeline_tsp20.sh
#   nohup bash auto_pipeline_tsp20.sh > pipeline.log 2>&1 &

set -euo pipefail

# ── 自动日志 ─────────────────────────────────────────────────────────────────
DISTILL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DISTILL_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_tsp20_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# ── 配置 ──────────────────────────────────────────────────────────────────────
REASON_DIR="$(cd "$DISTILL_DIR/../UniCOP-Reason" && pwd)"
NUM_GPUS=4
GPU_MEM_THRESHOLD=2000    # MB，低于此值视为空闲
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

# CUDA（DeepSpeed 编译需要）
export CUDA_HOME=/home/ntu/anaconda3/envs/unicop
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# LKH 求解器
export LKH_BIN=/home/ntu/LKH/LKH

# ── 工具函数 ──────────────────────────────────────────────────────────────────
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s "https://sctapi.ftqq.com/$SCKEY.send" \
        -d "title=$title" -d "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

trap 'notify "Pipeline 失败: line $LINENO" "$(tail -5 pipeline.log 2>/dev/null || echo unknown)"' ERR

wait_for_gpus() {
    local needed=$1
    echo "[$(date '+%H:%M:%S')] 等待 $needed 张 GPU 空闲 (显存 < ${GPU_MEM_THRESHOLD}MB)..."
    while true; do
        local free
        free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk -v t="$GPU_MEM_THRESHOLD" '$1 < t {n++} END {print n+0}')
        if [ "$free" -ge "$needed" ]; then
            echo "[$(date '+%H:%M:%S')] $free 张 GPU 空闲，继续"
            return
        fi
        echo "  空闲: $free/$needed, 等待 60s..."
        sleep 60
    done
}

cd "$DISTILL_DIR"

echo "============================================================"
echo "  TSP n=20 全自动流水线"
echo "  Distill: $DISTILL_DIR"
echo "  Reason:  $REASON_DIR"
echo "  GPU:     $NUM_GPUS 张"
echo "  时间:    $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 1: 生成 50K TSP n=20 solver 解
# ══════════════════════════════════════════════════════════════════
SOLUTIONS_FILE="data/solutions_tsp20.jsonl"
TARGET_SAMPLES=50000

echo ""
echo ">>> Step 1: 生成 TSP n=20 solutions (${TARGET_SAMPLES} 条)..."
if [ -f "$SOLUTIONS_FILE" ]; then
    EXISTING=$(grep -c '^{' "$SOLUTIONS_FILE" 2>/dev/null || echo 0)
    echo "  已有 $EXISTING 条样本"
    if [ "$EXISTING" -ge "$TARGET_SAMPLES" ]; then
        echo "  样本数已达标，跳过数据生成"
    else
        echo "  样本不足，继续生成 (断点续传)..."
        python stage1_solution/generate_solutions.py \
            --problems tsp \
            --sizes 20 \
            --num_samples $TARGET_SAMPLES \
            --output "$SOLUTIONS_FILE" \
            --workers 32
    fi
else
    echo "  数据文件不存在，开始生成..."
    python stage1_solution/generate_solutions.py \
        --problems tsp \
        --sizes 20 \
        --num_samples $TARGET_SAMPLES \
        --output "$SOLUTIONS_FILE" \
        --workers 32
fi
notify "Step1 完成: TSP20 数据生成"

# ══════════════════════════════════════════════════════════════════
# Step 2: Stage 1 SFT (Qwen2.5-Instruct → 学可行解)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: Stage 1 SFT..."
wait_for_gpus $NUM_GPUS
accelerate launch --num_processes $NUM_GPUS --main_process_port 29600 \
    stage1_solution/train_sft_stage1.py \
    --data data/solutions_tsp20.jsonl \
    --output_dir ./output_sft_stage1_tsp20 \
    --lora_rank 64 --lora_alpha 128 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-4 \
    --save_steps 500
notify "Step2 完成: Stage1 SFT"

# ══════════════════════════════════════════════════════════════════
# Step 3: 评估 Stage 1
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: 评估 Stage 1..."
cd "$REASON_DIR"
python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/output_sft_stage1_tsp20/final_model" \
    --problem tsp \
    --problem_size 20 \
    --model_type instruct \
    --max_completion_length 512 \
    --num_test 100 \
    --num_samples 1 \
    --batch_size 8 \
    --save_dir "$DISTILL_DIR/eval_results"
cd "$DISTILL_DIR"
notify "Step3 完成: Stage1 评估"

# ══════════════════════════════════════════════════════════════════
# Step 4: Stage 2 SFT (学 <think> 推理链)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 4: Stage 2 SFT..."

CHAINS_FILE="data/chains_v3_clean.jsonl"
if [ ! -f "$CHAINS_FILE" ]; then
    echo "ERROR: $CHAINS_FILE 不存在"
    echo "请将 chains_v3_clean.jsonl 复制到 $DISTILL_DIR/data/ 目录"
    notify "Pipeline 中止: 缺少 Stage2 chains 数据"
    exit 1
fi

TSP20_COUNT=$(python -c "
import json
count = 0
with open('$CHAINS_FILE', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get('problem_type') == 'tsp' and r.get('n') == 20:
            count += 1
print(count)
")
echo "  chains 文件中 TSP n=20 样本数: $TSP20_COUNT"
if [ "$TSP20_COUNT" -lt 10 ]; then
    echo "ERROR: TSP n=20 chains 样本不足 ($TSP20_COUNT 条)，无法训练 Stage 2"
    notify "Pipeline 中止: TSP20 chains 仅 $TSP20_COUNT 条"
    exit 1
fi

wait_for_gpus $NUM_GPUS
accelerate launch --num_processes $NUM_GPUS --main_process_port 29600 \
    stage2_reasoning/train_sft_stage2.py \
    --model ./output_sft_stage1_tsp20/final_model \
    --data "$CHAINS_FILE" \
    --filter_problems tsp \
    --filter_sizes 20 \
    --lora_rank 64 --lora_alpha 128 \
    --max_length 4096 \
    --output_dir ./output_sft_stage2_tsp20 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-4 \
    --save_steps 100
notify "Step4 完成: Stage2 SFT"

# ══════════════════════════════════════════════════════════════════
# Step 5: 评估 Stage 2
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 5: 评估 Stage 2..."
cd "$REASON_DIR"
python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/output_sft_stage2_tsp20/final_model" \
    --problem tsp \
    --problem_size 20 \
    --model_type reasoning \
    --max_completion_length 4096 \
    --num_test 100 \
    --num_samples 1 \
    --batch_size 4 \
    --save_dir "$DISTILL_DIR/eval_results"
cd "$DISTILL_DIR"
notify "Pipeline 全部完成: Stage1+Stage2 TSP20"

echo ""
echo "============================================================"
echo "  Pipeline 完成! $(date)"
echo "  Stage 1 模型: output_sft_stage1_tsp20/final_model"
echo "  Stage 2 模型: output_sft_stage2_tsp20/final_model"
echo "  评估结果:     eval_results/"
echo "============================================================"
