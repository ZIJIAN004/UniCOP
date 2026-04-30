#!/bin/bash
# CVRP n=20 全自动流水线：数据生成 → Stage1 SFT → Eval → Stage2 SFT → Eval
#
# 支持断点续跑：自动检测每个阶段的产出，跳过已完成的步骤
#   - 已有合并后完整模型 → 跳过训练 + 合并
#   - 只有 adapter 文件   → 跳过训练，只做合并
#   - 什么都没有         → 训练 + 合并
#
# 使用方法:
#   bash auto_pipeline_cvrp20.sh
#   nohup bash auto_pipeline_cvrp20.sh > pipeline.log 2>&1 &

set -euo pipefail

# ── 自动日志 ─────────────────────────────────────────────────────────────────
DISTILL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DISTILL_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_cvrp20_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# ── 配置 ──────────────────────────────────────────────────────────────────────
REASON_DIR="$(cd "$DISTILL_DIR/../UniCOP-Reason" && pwd)"
NUM_GPUS=4
GPU_MEM_THRESHOLD=2000    # MB，低于此值视为空闲
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=20
STAGE1_OUT="output_sft_stage1_cvrp20"
STAGE2_OUT="output_sft_stage2_cvrp20"

# CUDA（DeepSpeed 编译需要）
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

trap 'notify "Pipeline 失败: line $LINENO" "$(tail -5 "$LOG_FILE" 2>/dev/null || echo unknown)"' ERR

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

# 检测模型目录状态: merged / adapter / none
check_model_state() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "none"
        return
    fi
    if ls "$dir"/model*.safetensors 1>/dev/null 2>&1 || \
       ls "$dir"/pytorch_model*.bin 1>/dev/null 2>&1; then
        echo "merged"
        return
    fi
    if [ -f "$dir/adapter_config.json" ]; then
        echo "adapter"
        return
    fi
    echo "none"
}

# 检测评估结果是否已存在（匹配 model_path 后缀 + problem + size + model_type）
check_eval_exists() {
    local eval_dir="$1"
    local model_suffix="$2"
    local problem="$3"
    local size="$4"
    local model_type="$5"
    python -c "
import json, glob, sys
for f in sorted(glob.glob('${eval_dir}/*.json'), reverse=True):
    try:
        with open(f) as fp:
            d = json.load(fp)
        hp = d.get('hyperparams', {})
        if hp.get('model_path','').endswith('${model_suffix}') \
           and '${problem}' in hp.get('problems', []) \
           and ${size} in hp.get('problem_sizes', []) \
           and hp.get('model_type') == '${model_type}':
            print(f)
            sys.exit(0)
    except: pass
sys.exit(1)
" 2>/dev/null
}

cd "$DISTILL_DIR"

echo "============================================================"
echo "  CVRP n=$SIZE 全自动流水线"
echo "  Distill: $DISTILL_DIR"
echo "  Reason:  $REASON_DIR"
echo "  GPU:     $NUM_GPUS 张"
echo "  时间:    $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 1: 生成 50K CVRP n=20 solver 解
# ══════════════════════════════════════════════════════════════════
SOLUTIONS_FILE="data/solutions_cvrp${SIZE}.jsonl"
TARGET_SAMPLES=50000

echo ""
echo ">>> Step 1: 生成 CVRP n=$SIZE solutions (${TARGET_SAMPLES} 条)..."
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
notify "Step1 完成: CVRP${SIZE} 数据生成"

# ══════════════════════════════════════════════════════════════════
# Step 2: Stage 1 SFT (Qwen2.5-Instruct → 学可行解)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: Stage 1 SFT..."

STAGE1_STATE=$(check_model_state "$STAGE1_OUT/final_model")
echo "  Stage 1 模型状态: $STAGE1_STATE"

case "$STAGE1_STATE" in
    merged)
        echo "  已有合并后的完整模型，跳过训练和合并"
        ;;
    adapter)
        echo "  已有 adapter，跳过训练，执行合并..."
        python stage1_solution/merge_adapter.py \
            --adapter_path "$STAGE1_OUT/final_model"
        notify "Step2 完成: Stage1 adapter 合并 (跳过训练)"
        ;;
    none)
        wait_for_gpus $NUM_GPUS
        accelerate launch --num_processes $NUM_GPUS --main_process_port 29600 \
            stage1_solution/train_sft_stage1.py \
            --data "$SOLUTIONS_FILE" \
            --output_dir "$STAGE1_OUT" \
            --lora_rank 64 --lora_alpha 128 \
            --zero_stage 3 \
            --gradient_checkpointing \
            --epochs 3 \
            --batch_size 4 \
            --grad_accum 2 \
            --lr 1e-4 \
            --save_steps 500
        notify "Step2 完成: Stage1 SFT 训练"

        echo "  合并 Stage 1 LoRA adapter..."
        python stage1_solution/merge_adapter.py \
            --adapter_path "$STAGE1_OUT/final_model"
        notify "Step2 完成: Stage1 adapter 合并"
        ;;
esac

# ══════════════════════════════════════════════════════════════════
# Step 3: 评估 Stage 1
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: 评估 Stage 1..."

EVAL1_FILE=$(check_eval_exists "$DISTILL_DIR/eval_results" "$STAGE1_OUT/final_model" "$PROBLEM" "$SIZE" "instruct") || true
if [ -n "$EVAL1_FILE" ]; then
    echo "  已有评估结果: $EVAL1_FILE，跳过"
else
    cd "$REASON_DIR"
    python evaluate.py \
        --backend local \
        --model_path "$DISTILL_DIR/$STAGE1_OUT/final_model" \
        --problem $PROBLEM \
        --problem_size $SIZE \
        --model_type instruct \
        --max_completion_length 512 \
        --num_test 100 \
        --num_samples 1 \
        --batch_size 8 \
        --save_dir "$DISTILL_DIR/eval_results"
    cd "$DISTILL_DIR"
    notify "Step3 完成: Stage1 评估"
fi

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

PROBLEM_COUNT=$(python -c "
import json
count = 0
with open('$CHAINS_FILE', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get('problem_type') == '$PROBLEM' and r.get('n') == $SIZE:
            count += 1
print(count)
")
echo "  chains 文件中 ${PROBLEM^^} n=$SIZE 样本数: $PROBLEM_COUNT"
if [ "$PROBLEM_COUNT" -lt 10 ]; then
    echo "ERROR: ${PROBLEM^^} n=$SIZE chains 样本不足 ($PROBLEM_COUNT 条)，无法训练 Stage 2"
    notify "Pipeline 中止: ${PROBLEM^^}${SIZE} chains 仅 $PROBLEM_COUNT 条"
    exit 1
fi

STAGE2_STATE=$(check_model_state "$STAGE2_OUT/final_model")
echo "  Stage 2 模型状态: $STAGE2_STATE"

case "$STAGE2_STATE" in
    merged)
        echo "  已有合并后的完整模型，跳过训练和合并"
        ;;
    adapter)
        echo "  已有 adapter，跳过训练，执行合并..."
        python stage1_solution/merge_adapter.py \
            --adapter_path "$STAGE2_OUT/final_model"
        notify "Step4 完成: Stage2 adapter 合并 (跳过训练)"
        ;;
    none)
        wait_for_gpus $NUM_GPUS
        accelerate launch --num_processes $NUM_GPUS --main_process_port 29600 \
            stage2_reasoning/train_sft_stage2.py \
            --model "$STAGE1_OUT/final_model" \
            --data "$CHAINS_FILE" \
            --filter_problems $PROBLEM \
            --filter_sizes $SIZE \
            --lora_rank 64 --lora_alpha 128 \
            --max_length 4096 \
            --output_dir "$STAGE2_OUT" \
            --zero_stage 3 \
            --gradient_checkpointing \
            --epochs 3 \
            --batch_size 1 \
            --grad_accum 8 \
            --lr 1e-4 \
            --save_steps 100
        notify "Step4 完成: Stage2 SFT 训练"

        echo "  合并 Stage 2 LoRA adapter..."
        python stage1_solution/merge_adapter.py \
            --adapter_path "$STAGE2_OUT/final_model"
        notify "Step4 完成: Stage2 adapter 合并"
        ;;
esac

# ══════════════════════════════════════════════════════════════════
# Step 5: 评估 Stage 2
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 5: 评估 Stage 2..."

EVAL2_FILE=$(check_eval_exists "$DISTILL_DIR/eval_results" "$STAGE2_OUT/final_model" "$PROBLEM" "$SIZE" "reasoning") || true
if [ -n "$EVAL2_FILE" ]; then
    echo "  已有评估结果: $EVAL2_FILE，跳过"
else
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
fi
notify "Pipeline 全部完成: Stage1+Stage2 ${PROBLEM^^}${SIZE}"

echo ""
echo "============================================================"
echo "  Pipeline 完成! $(date)"
echo "  Stage 1 模型: $STAGE1_OUT/final_model"
echo "  Stage 2 模型: $STAGE2_OUT/final_model"
echo "  评估结果:     eval_results/"
echo "============================================================"
