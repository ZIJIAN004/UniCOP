#!/bin/bash
# run_sft_qwen3_instruct_cvrp100.sh
# ──────────────────────────────────────────────────────────────────────────────
# UniCOP 思维臂 SFT @ Qwen3-4B-Instruct-2507 (非 thinking), CVRP100, 一体化:
#   Step 0: think mask preflight (CPU)
#   Step 1: Stage2 SFT (stride=5 块决策思维链, 4 GPU ZeRO-2)
#   Step 2: merge LoRA adapter → 完整权重
#   Step 3: bo1 eval (best-of-1 贪心), 与同基座 FOARL 臂跑同一 eval 集 (seed=9999)
#
# 与 cvrp20 版 (run_sft_qwen3_instruct_cvrp20.sh) 的差异 = 仅规模相关:
#   SIZE 20→100; 数据换 chains_template_cvrp100; 长度按真 Qwen3 tokenizer 实测放大:
#     SFT_MAX_LENGTH      8192 → 13568  (cvrp100 total p99 实测)
#     --max_output_length 4096(默认) → 9856  (cvrp100 completion 实测; cvrp20 没显式传, cvrp100 必须传, 否则 100% 样本被丢)
#     EVAL_MAX_COMPLETION 4096 → 9856  (eval 要生成 ~9.5k token 链, 4096 会截断 → parse 失败)
#
# 主机: A*STAR-Zhihan (直连 SSH, 无 SLURM) → 必须挂 tmux:
#   tmux new -s sft100_uni
#   bash run_sft_qwen3_instruct_cvrp100.sh
#   (Ctrl-b d 脱离; tmux attach -t sft100_uni 回看)
#
# ⚠️ 显存: ZeRO-2 每卡存完整 4B 参数(~8GB), 较 ZeRO-3 多 ~6GB。
#     若 4 卡 OOM: NUM_GPUS=6 bash run_sft_qwen3_instruct_cvrp100.sh (加卡降单卡显存)。
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1
_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

export BASE_MODEL_TYPE=qwen3_instruct
source "$(dirname "$_SELF_DIR")/paths.sh"

LOG_FILE="$DISTILL_DIR/sft_qwen3_instruct_cvrp100_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# ── 配置 ──────────────────────────────────────────────────────────────────────
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=100
CHAINS_FILE="data/chains_template_${PROBLEM}${SIZE}.jsonl"
OUTPUT_DIR="output_sft_qwen3_instruct_template_${PROBLEM}${SIZE}"
EVAL_SAVE_DIR="$DISTILL_DIR/eval_results_qwen3_instruct_bo1"

# SFT 超参 (LoRA/lr/epoch 与 cvrp20 一致; 仅长度按 cvrp100 实测放大)
SFT_LR=1e-4
SFT_EPOCHS=3
SFT_LORA_RANK=64
SFT_LORA_ALPHA=128
SFT_MAX_LENGTH=13568
SFT_MAX_OUTPUT=9856
SFT_NUM_GPUS="${NUM_GPUS:-4}"

# bo1 eval 配置
EVAL_NUM_TEST=100
EVAL_NUM_SAMPLES=1
EVAL_TEMPERATURE=0
EVAL_MAX_COMPLETION=9856
EVAL_BATCH_SIZE=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 工具函数 ──────────────────────────────────────────────────────────────────
notify() {
    local title="${1:0:100}"; local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}
get_free_gpus() {
    local threshold=${1:-500}
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$threshold" '$2 < t {print $1}'
}
wait_for_free_gpus() {
    local required=$1; local max_wait=${2:-1800}; local waited=0
    while true; do
        local free=($(get_free_gpus))
        if [ ${#free[@]} -ge "$required" ]; then echo "${free[@]:0:$required}"; return 0; fi
        echo "  空闲 GPU: ${#free[@]}/$required, 等待中... (${waited}s)" >&2
        sleep 30; waited=$((waited + 30))
        if [ "$waited" -ge "$max_wait" ]; then echo "ERROR: 等待 $required 张 GPU 超时" >&2; return 1; fi
    done
}
trap 'notify "❌ Qwen3-Instruct CVRP100 SFT 失败: line $LINENO"' ERR
cd "$DISTILL_DIR"

# ── 前置检查 ────────────────────────────────────────────────────────────────
if [ ! -d "$BASE_MODEL" ] || [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "ERROR: instruct 基座不存在: $BASE_MODEL (paths.sh BASE_MODEL_QWEN3_INSTRUCT)"
    notify "❌ instruct 基座缺失" "$BASE_MODEL 不存在"
    exit 1
fi
if [ ! -f "$CHAINS_FILE" ]; then
    echo "ERROR: 训练数据 $CHAINS_FILE 不存在 (先跑 FOARL/build_train_data_cvrp100.sh)"
    exit 1
fi
DATA_COUNT=$(grep -c '^{' "$CHAINS_FILE" 2>/dev/null || echo 0)

echo "============================================================"
echo "  UniCOP 思维臂 SFT @ Qwen3-4B-Instruct-2507  CVRP100 (stride=5)"
echo "  基座:     $BASE_MODEL"
echo "  数据:     $CHAINS_FILE ($DATA_COUNT 条)"
echo "  输出:     $OUTPUT_DIR"
echo "  长度:     max_length=$SFT_MAX_LENGTH  max_output=$SFT_MAX_OUTPUT"
echo "  时间:     $(date)"
echo "============================================================"

# ── Step 0: think mask preflight ──────────────────────────────────────────────
echo ""; echo ">>> Step 0: think mask preflight (CPU)..."
python check_sft_think_masking.py \
    --model "$BASE_MODEL" --data "$CHAINS_FILE" \
    --filter_problems $PROBLEM --filter_sizes $SIZE --n_samples 3
echo "  ✓ preflight 通过"

# ── Step 1: Stage2 SFT ────────────────────────────────────────────────────────
echo ""; echo ">>> Step 1: SFT 训练 (等待 ${SFT_NUM_GPUS} 张空闲 GPU)..."
SFT_GPU_LIST=($(wait_for_free_gpus $SFT_NUM_GPUS))
SFT_CUDA_DEVICES=$(IFS=,; echo "${SFT_GPU_LIST[*]}")
echo "  使用 GPU: $SFT_CUDA_DEVICES"

CUDA_VISIBLE_DEVICES=$SFT_CUDA_DEVICES accelerate launch \
    --num_processes $SFT_NUM_GPUS --main_process_port 29600 \
    stage2_reasoning/train_sft_stage2.py \
    --model "$BASE_MODEL" \
    --data "$CHAINS_FILE" \
    --filter_problems $PROBLEM --filter_sizes $SIZE \
    --lora_rank $SFT_LORA_RANK --lora_alpha $SFT_LORA_ALPHA \
    --max_length $SFT_MAX_LENGTH \
    --max_output_length $SFT_MAX_OUTPUT \
    --output_dir "$OUTPUT_DIR" \
    --zero_stage 2 --gradient_checkpointing \
    --resume_from_checkpoint auto \
    --epochs $SFT_EPOCHS --batch_size 2 --grad_accum 4 \
    --lr $SFT_LR --save_steps 200

if [ ! -f "$OUTPUT_DIR/final_model/adapter_config.json" ] && \
   [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: 训练未产出 final_model"; exit 1
fi
notify "Step1 完成: UniCOP CVRP100 SFT 训练"

# ── Step 2: merge LoRA ────────────────────────────────────────────────────────
echo ""; echo ">>> Step 2: 合并 LoRA adapter..."
if [ -f "$OUTPUT_DIR/final_model/adapter_config.json" ]; then
    python stage1_solution/merge_adapter.py --adapter_path "$OUTPUT_DIR/final_model"
    echo "  ✓ 合并完成: $OUTPUT_DIR/final_model"
else
    echo "  final_model 已是完整权重, 跳过合并"
fi
if [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: merge 后无 config.json, 中止 eval"; exit 1
fi
# 权重非空校验 (踩坑#23: ZeRO-3+LoRA 可能存出空壳 adapter/权重, 见 CLAUDE.md 代码自审)
_W=$(find "$OUTPUT_DIR/final_model" -maxdepth 1 -name '*.safetensors' -size +0c 2>/dev/null | head -1)
[ -n "$_W" ] || { echo "ERROR: 合并权重为空 (无非空 safetensors)"; exit 1; }
notify "Step2 完成: adapter 合并"

# ── Step 3: bo1 评估 ──────────────────────────────────────────────────────────
echo ""; echo ">>> Step 3: bo1 评估 (贪心 temperature=0)..."
mkdir -p "$EVAL_SAVE_DIR"
EVAL_GPU=($(wait_for_free_gpus 1))
echo "  eval 使用 GPU: ${EVAL_GPU[0]}"

cd "$REASON_DIR"
CUDA_VISIBLE_DEVICES=${EVAL_GPU[0]} python evaluate.py \
    --backend local \
    --model_path "$DISTILL_DIR/$OUTPUT_DIR/final_model" \
    --problem $PROBLEM --problem_size $SIZE \
    --model_type reasoning --prompt_mode think --stride 5 \
    --max_completion_length $EVAL_MAX_COMPLETION \
    --num_test $EVAL_NUM_TEST --num_samples $EVAL_NUM_SAMPLES \
    --batch_size $EVAL_BATCH_SIZE --temperature $EVAL_TEMPERATURE \
    --save_dir "$EVAL_SAVE_DIR"
cd "$DISTILL_DIR"

notify "✅ UniCOP CVRP100 SFT+bo1 全部完成" "模型: $OUTPUT_DIR/final_model"
echo ""; echo "============================================================"
echo "  完成! $(date)"
echo "  合并模型: $OUTPUT_DIR/final_model"
echo "  bo1 评估: $EVAL_SAVE_DIR/"
echo "    jq '.results[0]|{parse:.format_match_rate,feas:.global_feasibility_rate,dist:.avg_best_dist}' $EVAL_SAVE_DIR/*.json"
echo "============================================================"
