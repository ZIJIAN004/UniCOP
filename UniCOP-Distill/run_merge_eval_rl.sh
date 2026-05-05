#!/bin/bash
# run_merge_eval_rl.sh — Template SFT 产物：合并 → 评估
#
# 前提：output_sft_template_cvrp20/checkpoint-2000 下存在 adapter 文件
#
# 流程：
#   Step 1: 合并 LoRA adapter → 完整模型
#   Step 2: 评估合并后模型（CVRP n=20, 本地 HF 推理）
#
# SBATCH 提交：
#   sbatch submit_merge_eval_rl.sh
# 或手动运行：
#   bash run_merge_eval_rl.sh

#SBATCH --qos express
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/merge_eval_%j.log

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

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$DISTILL_DIR"

LOG_FILE="$DISTILL_DIR/merge_eval_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# ── 配置 ────────────────────────────────────────────────────────────────────
PROBLEM="cvrp"
SIZE=20
ADAPTER_DIR="output_sft_template_${PROBLEM}${SIZE}/checkpoint-2000"
MERGED_MODEL="$DISTILL_DIR/$ADAPTER_DIR"

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

trap 'notify "❌ merge_eval 失败 (line $LINENO)"' ERR

echo "============================================================"
echo "  Template CVRP20: Merge → Eval"
echo "  Adapter:   $ADAPTER_DIR"
echo "  时间:      $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 合并 LoRA adapter
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: 合并 LoRA adapter..."

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
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

cd "$REASON_DIR"

python evaluate.py \
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

notify "✅ Template CVRP20 Merge+Eval 完成" "结果在 $EVAL_SAVE_DIR"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  合并模型:  $MERGED_MODEL"
echo "  评估结果:  $EVAL_SAVE_DIR/"
echo "============================================================"
