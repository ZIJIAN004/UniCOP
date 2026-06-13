#!/bin/bash
# run_sft_qwen3_instruct_cvrp20.sh
# ──────────────────────────────────────────────────────────────────────────────
# 我们方法的 SFT 搬到 Qwen3-4B-Instruct-2507 (非 thinking) 基座, 一体化:
#   Step 1: Stage2 SFT (template 推理链, 4 GPU ZeRO-3)
#   Step 2: merge LoRA adapter → 完整权重
#   Step 3: bo1 eval (best-of-1, num_samples=1), 与 thinking 跑同一 eval 集
#
# 用途: 把 thinking 上跑过的 "我们方法" 复制到 instruct 基座, 以便和 FOARL
#       (Qwen2.5-7B-Instruct + SFT + RL) 在同一 instruct 范式下受控对比。
#
# 主机: A*STAR-Zhihan (直连 SSH, 无 SLURM) → 必须挂 tmux, 关终端不断:
#   tmux new -s sft_instruct
#   bash run_sft_qwen3_instruct_cvrp20.sh
#   (Ctrl-b d 脱离; tmux attach -t sft_instruct 回看)
# 或: nohup bash run_sft_qwen3_instruct_cvrp20.sh > /dev/null 2>&1 &
#
# eval 集合一致性: evaluate.py 用硬编码 seed=9999 生成测试实例 (evaluate.py:865),
#   测试集只由 (seed=9999, problem, problem_size, num_test) 决定, 与模型/温度无关。
#   故下面 --problem cvrp --problem_size 20 --num_test 100 跑出的 100 个实例,
#   与 thinking 基座下同参数跑出的完全是同一批 → bo1 对比公平。
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 选定 instruct 基座, 再 source paths.sh (paths.sh 读 BASE_MODEL_TYPE) ──
export BASE_MODEL_TYPE=qwen3_instruct
source "$(dirname "$_SELF_DIR")/paths.sh"

LOG_FILE="$DISTILL_DIR/sft_qwen3_instruct_cvrp20_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# ── 配置 ──────────────────────────────────────────────────────────────────────
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=20
CHAINS_FILE="data/chains_template_${PROBLEM}${SIZE}.jsonl"
OUTPUT_DIR="output_sft_qwen3_instruct_template_${PROBLEM}${SIZE}"
EVAL_SAVE_DIR="$DISTILL_DIR/eval_results_qwen3_instruct_bo1"

# SFT 配置 (与 submit_sft_qwen3_full.sh 的 qwen3 方法配置一致, 保证 instruct/thinking 同超参)
SFT_LR=1e-4
SFT_EPOCHS=3
SFT_LORA_RANK=64
SFT_LORA_ALPHA=128
SFT_MAX_LENGTH=8192
SFT_NUM_GPUS=4

# bo1 eval 配置
EVAL_NUM_TEST=100
EVAL_NUM_SAMPLES=1                 # bo1 = best-of-1
EVAL_TEMPERATURE=0                 # bo1 用贪心解码 (deterministic, 完全可复现)
EVAL_MAX_COMPLETION=4096
EVAL_BATCH_SIZE=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 工具函数 ──────────────────────────────────────────────────────────────────
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

trap 'notify "❌ Qwen3-Instruct CVRP20 SFT 失败: line $LINENO"' ERR

cd "$DISTILL_DIR"

# ── 前置检查: 基座模型与数据 ──────────────────────────────────────────────────
if [ ! -d "$BASE_MODEL" ] || [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "ERROR: instruct 基座模型不存在: $BASE_MODEL"
    echo "  请先下载 Qwen3-4B-Instruct-2507 到该路径, 例如:"
    echo "    huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir \"$BASE_MODEL\""
    echo "  (路径在 paths.sh 的 BASE_MODEL_QWEN3_INSTRUCT 配置, 按主机切换)"
    notify "❌ instruct 基座缺失" "$BASE_MODEL 不存在, 请先下载 Qwen3-4B-Instruct-2507"
    exit 1
fi
if [ ! -f "$CHAINS_FILE" ]; then
    echo "ERROR: 训练数据 $CHAINS_FILE 不存在"
    exit 1
fi
DATA_COUNT=$(grep -c '^{' "$CHAINS_FILE" 2>/dev/null || echo 0)

echo "============================================================"
echo "  我们方法 SFT @ Qwen3-4B-Instruct-2507 (非 thinking)"
echo "  基座:     $BASE_MODEL"
echo "  数据:     $CHAINS_FILE ($DATA_COUNT 条)"
echo "  输出:     $OUTPUT_DIR"
echo "  采样:     T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K"
echo "  时间:     $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 0: think mask preflight —— 确认 <think> 格式能被正确识别和训练
#   (collator 找不到 response_template 会静默全 mask 白训, 必须先验证)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 0: think mask preflight (CPU, 不占 GPU)..."
python check_sft_think_masking.py \
    --model "$BASE_MODEL" \
    --data "$CHAINS_FILE" \
    --filter_problems $PROBLEM --filter_sizes $SIZE \
    --n_samples 3
echo "  ✓ preflight 通过, think 格式 mask 正确"

# ══════════════════════════════════════════════════════════════════
# Step 1: Stage2 SFT 训练
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: SFT 训练 (等待至少 ${SFT_NUM_GPUS} 张空闲 GPU)..."

SFT_GPU_LIST=($(wait_for_free_gpus $SFT_NUM_GPUS))
SFT_CUDA_DEVICES=$(IFS=,; echo "${SFT_GPU_LIST[*]}")
echo "  使用 GPU: $SFT_CUDA_DEVICES"

CUDA_VISIBLE_DEVICES=$SFT_CUDA_DEVICES accelerate launch \
    --num_processes $SFT_NUM_GPUS \
    --main_process_port 29600 \
    stage2_reasoning/train_sft_stage2.py \
    --model "$BASE_MODEL" \
    --data "$CHAINS_FILE" \
    --filter_problems $PROBLEM \
    --filter_sizes $SIZE \
    --lora_rank $SFT_LORA_RANK --lora_alpha $SFT_LORA_ALPHA \
    --max_length $SFT_MAX_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --zero_stage 3 \
    --gradient_checkpointing \
    --resume_from_checkpoint auto \
    --epochs $SFT_EPOCHS \
    --batch_size 1 \
    --grad_accum 8 \
    --lr $SFT_LR \
    --save_steps 200

if [ ! -f "$OUTPUT_DIR/final_model/adapter_config.json" ] && \
   [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: 训练未产出 final_model (既无 adapter_config.json 也无 config.json)"
    exit 1
fi

notify "Step1 完成: Qwen3-Instruct Stage2 SFT 训练"

# ══════════════════════════════════════════════════════════════════
# Step 2: 合并 LoRA adapter
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 合并 LoRA adapter..."

if [ -f "$OUTPUT_DIR/final_model/adapter_config.json" ]; then
    python stage1_solution/merge_adapter.py \
        --adapter_path "$OUTPUT_DIR/final_model"
    echo "  ✓ 合并完成: $OUTPUT_DIR/final_model"
else
    echo "  final_model 已是完整权重 (无 adapter_config.json), 跳过合并"
fi

if [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: merge 后未生成 config.json, 中止 eval"
    exit 1
fi

notify "Step2 完成: adapter 合并"

# ══════════════════════════════════════════════════════════════════
# Step 3: bo1 评估 (best-of-1) —— 与 thinking 跑同一 eval 集 (seed=9999)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: bo1 评估 (num_samples=1, 贪心 temperature=0)..."
echo "  模型:     $OUTPUT_DIR/final_model"
echo "  测试数:   $EVAL_NUM_TEST (seed=9999 固定, 与 thinking 同一批实例)"
echo "  解码:     greedy (temperature=$EVAL_TEMPERATURE), deterministic 可复现"
echo "  保存:     $EVAL_SAVE_DIR"

mkdir -p "$EVAL_SAVE_DIR"

# 等 1 张空闲 GPU 做 eval (merge 后基座已释放显存)
EVAL_GPU=($(wait_for_free_gpus 1))
echo "  eval 使用 GPU: ${EVAL_GPU[0]}"

cd "$REASON_DIR"
CUDA_VISIBLE_DEVICES=${EVAL_GPU[0]} python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/$OUTPUT_DIR/final_model" \
    --problem $PROBLEM \
    --problem_size $SIZE \
    --model_type reasoning \
    --prompt_mode think \
    --max_completion_length $EVAL_MAX_COMPLETION \
    --num_test $EVAL_NUM_TEST \
    --num_samples $EVAL_NUM_SAMPLES \
    --batch_size $EVAL_BATCH_SIZE \
    --temperature $EVAL_TEMPERATURE \
    --save_dir "$EVAL_SAVE_DIR"
cd "$DISTILL_DIR"

notify "✅ Qwen3-Instruct CVRP20 SFT+bo1 全部完成" \
    "模型: $OUTPUT_DIR/final_model, eval 结果在 $EVAL_SAVE_DIR/"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  数据:     $CHAINS_FILE ($DATA_COUNT 条)"
echo "  合并模型: $OUTPUT_DIR/final_model"
echo "  bo1 评估: $EVAL_SAVE_DIR/"
echo ""
echo "  Quick view:"
echo "    jq '.results[0] | {parse:.format_match_rate, feas:.global_feasibility_rate, dist:.avg_best_dist}' $EVAL_SAVE_DIR/*.json"
echo "============================================================"
