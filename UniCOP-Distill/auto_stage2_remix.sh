#!/bin/bash
# Stage 2 Replay-Mix 重训脚本：混合数据生成 → SFT 训练 → LoRA 合并
#
# 使用方法:
#   bash auto_stage2_remix.sh
#   nohup bash auto_stage2_remix.sh > stage2_remix.log 2>&1 &

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────────
DISTILL_DIR="$(cd "$(dirname "$0")" && pwd)"
REASON_DIR="$(cd "$DISTILL_DIR/../UniCOP-Reason" && pwd)"
NUM_GPUS=4
GPU_MEM_THRESHOLD=2000
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=20
STAGE1_OUT="output_sft_stage1_cvrp20"
STAGE2_OUT="output_sft_stage2_cvrp20_v2"

SOLUTIONS_FILE="data/solutions_cvrp${SIZE}.jsonl"
CHAINS_FILE="data/chains_v3_clean.jsonl"
MIXED_FILE="data/chains_v3_mixed.jsonl"
NUM_REPLAY=1000

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

trap 'notify "Stage2 Remix 失败: line $LINENO" "$(tail -5 stage2_remix.log 2>/dev/null || echo unknown)"' ERR

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
echo "  Stage 2 Replay-Mix 重训"
echo "  Stage 1 模型: $STAGE1_OUT/final_model"
echo "  输出目录:     $STAGE2_OUT"
echo "  回放数量:     $NUM_REPLAY"
echo "  时间:         $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 1: 生成混合数据集
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: 生成混合数据集..."

if [ ! -f "$SOLUTIONS_FILE" ]; then
    echo "ERROR: Stage 1 数据不存在: $SOLUTIONS_FILE"
    notify "Stage2 Remix 中止: 缺少 $SOLUTIONS_FILE"
    exit 1
fi
if [ ! -f "$CHAINS_FILE" ]; then
    echo "ERROR: Gemini chains 不存在: $CHAINS_FILE"
    notify "Stage2 Remix 中止: 缺少 $CHAINS_FILE"
    exit 1
fi
if [ ! -d "$STAGE1_OUT/final_model" ]; then
    echo "ERROR: Stage 1 模型不存在: $STAGE1_OUT/final_model"
    notify "Stage2 Remix 中止: 缺少 Stage 1 模型"
    exit 1
fi

python mix_replay_data.py \
    --solutions "$SOLUTIONS_FILE" \
    --chains "$CHAINS_FILE" \
    --output "$MIXED_FILE" \
    --problem $PROBLEM \
    --size $SIZE \
    --num_replay $NUM_REPLAY

echo "  混合数据集已生成: $MIXED_FILE"

# ══════════════════════════════════════════════════════════════════
# Step 2: Stage 2 SFT 训练
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: Stage 2 SFT (replay-mix)..."

wait_for_gpus $NUM_GPUS

accelerate launch --num_processes $NUM_GPUS --main_process_port 29600 \
    stage2_reasoning/train_sft_stage2.py \
    --model "$STAGE1_OUT/final_model" \
    --data "$MIXED_FILE" \
    --filter_problems $PROBLEM \
    --filter_sizes $SIZE \
    --lora_rank 64 --lora_alpha 128 \
    --max_length 4096 \
    --output_dir "$STAGE2_OUT" \
    --zero_stage 3 \
    --gradient_checkpointing \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-5 \
    --save_steps 100

notify "Stage2 Remix: SFT 训练完成"

# ══════════════════════════════════════════════════════════════════
# Step 3: 合并 LoRA adapter
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: 合并 LoRA adapter..."

python stage1_solution/merge_adapter.py \
    --adapter_path "$STAGE2_OUT/final_model"

notify "Stage2 Remix: adapter 合并完成"

# ══════════════════════════════════════════════════════════════════
# Step 4: 评估
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 4: 评估 Stage 2 (remix)..."

cd "$REASON_DIR"
python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/$STAGE2_OUT/final_model" \
    --problem $PROBLEM \
    --problem_size $SIZE \
    --model_type reasoning \
    --max_completion_length 4096 \
    --num_test 100 \
    --num_samples 1 \
    --batch_size 4 \
    --save_dir "$DISTILL_DIR/eval_results"
cd "$DISTILL_DIR"

notify "Stage2 Remix 全部完成" "模型: $STAGE2_OUT, 评估结果在 eval_results/"

echo ""
echo "============================================================"
echo "  Stage 2 Remix 完成! $(date)"
echo "  模型: $STAGE2_OUT/final_model"
echo "  评估结果: eval_results/"
echo "============================================================"
