#!/bin/bash
# run_sft_foarl_cvrp100.sh
# ──────────────────────────────────────────────────────────────────────────────
# FOARL 无推理臂 SFT @ Qwen3-4B-Instruct-2507, CVRP100, zhihan 一体化:
#   Step 1: Stage-1 SFT (FOARL 原版格式, 无 think, 4 GPU ZeRO-2)
#   Step 2: merge LoRA adapter → 完整权重
#   Step 3: bo1 eval (best-of-1 贪心, vLLM TP=1, prompt_mode foarl)
#
# 与 UniCOP 思维臂受控对比: 同基座 / 同 LoRA(r64α128) / 同 lr2e-5·3ep·bs1ga8 / 同实例集,
#   唯一差异 = "有无 think"。长度按真 Qwen3 tokenizer 实测 (FOARL 输出短, 仅 Routes+Objective):
#     max_length=6400  max_output_length=512   (cvrp20 是 4864/1024)
#
# 主机: A*STAR-Zhihan (直连 SSH, 无 SLURM) → 必须挂 tmux:
#   tmux new -s sft100_foarl
#   bash run_sft_foarl_cvrp100.sh
#   (Ctrl-b d 脱离; tmux attach -t sft100_foarl 回看)
#
# ⚠️ 显存: ZeRO-2 每卡存完整 4B 参数(~8GB), 较 ZeRO-3 多 ~6GB。FOARL 序列仅 6400 tokens,
#    显存压力远小于 UniCOP 思维臂(13568), 基本无 OOM 风险。
# ⚠️ FOARL eval 用 vLLM TP=1: vLLM 0.7.3 无 Qwen3 原生实现→回退 Transformers backend,
#    TP>1 多 worker 生成期崩溃 (见 submit_eval_bo1_foarl_compare.sh)。TP=1 规避, 对贪心结果零影响。
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1
_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

export BASE_MODEL_TYPE=qwen3_instruct
source "$(dirname "$_SELF_DIR")/paths.sh"
FOARL_DIR="$UNICOP_ROOT/FOARL"

LOG_FILE="$FOARL_DIR/sft_foarl_cvrp100_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

# ── 配置 ──────────────────────────────────────────────────────────────────────
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

PROBLEM="cvrp"
SIZE=100
DATA="data/foarl_${PROBLEM}${SIZE}.jsonl"
SRC="../UniCOP-Distill/data/chains_template_${PROBLEM}${SIZE}.jsonl"   # 缺数据时从 chains 转 (与思维臂同源)
OUTPUT_DIR="output_sft_foarl_${PROBLEM}${SIZE}"
EVAL_SAVE_DIR="$FOARL_DIR/eval_results_foarl_bo1"

# SFT 超参 (与 UniCOP 思维臂对齐; 仅长度按 cvrp100 FOARL 实测)
SFT_LR=2e-5
SFT_EPOCHS=3
SFT_LORA_RANK=64
SFT_LORA_ALPHA=128
SFT_MAX_LENGTH=6400
SFT_MAX_OUTPUT=512
SFT_NUM_GPUS="${NUM_GPUS:-4}"
K_NN=2

# bo1 eval 配置
EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"   # 快排 100; 正式冻结对比用 1000
EVAL_MAX_COMPLETION=512
EVAL_GPU_MEM=0.8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

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
trap 'notify "❌ FOARL CVRP100 SFT 失败: line $LINENO"' ERR
cd "$FOARL_DIR"

# ── 前置检查 ────────────────────────────────────────────────────────────────
if [ ! -d "$BASE_MODEL" ] || [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "ERROR: instruct 基座不存在: $BASE_MODEL"; notify "❌ instruct 基座缺失" "$BASE_MODEL"; exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "[data] $DATA 不存在, 从 $SRC 转换 (k_nn=$K_NN)..."
    python build_foarl_cvrp_data.py --src "$SRC" --out "$DATA" --k_nn $K_NN
fi
[ -f "$DATA" ] || { echo "ERROR: 数据仍不存在: $DATA (检查 SRC=$SRC, 先跑 build_train_data_cvrp100.sh)"; exit 1; }
DATA_COUNT=$(grep -c '^{' "$DATA" 2>/dev/null || echo 0)

echo "============================================================"
echo "  FOARL 无推理臂 SFT @ Qwen3-4B-Instruct-2507  CVRP100"
echo "  基座:     $BASE_MODEL"
echo "  数据:     $DATA ($DATA_COUNT 条)"
echo "  输出:     $OUTPUT_DIR"
echo "  长度:     max_length=$SFT_MAX_LENGTH  max_output=$SFT_MAX_OUTPUT"
echo "  时间:     $(date)"
echo "============================================================"

# ── Step 1: Stage-1 SFT ───────────────────────────────────────────────────────
echo ""; echo ">>> Step 1: FOARL SFT 训练 (等待 ${SFT_NUM_GPUS} 张空闲 GPU)..."
SFT_GPU_LIST=($(wait_for_free_gpus $SFT_NUM_GPUS))
SFT_CUDA_DEVICES=$(IFS=,; echo "${SFT_GPU_LIST[*]}")
echo "  使用 GPU: $SFT_CUDA_DEVICES"

CUDA_VISIBLE_DEVICES=$SFT_CUDA_DEVICES accelerate launch \
    --num_processes $SFT_NUM_GPUS --main_process_port 29601 \
    train_sft_foarl.py \
    --model "$BASE_MODEL" \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank $SFT_LORA_RANK --lora_alpha $SFT_LORA_ALPHA \
    --lr $SFT_LR --epochs $SFT_EPOCHS \
    --batch_size 2 --grad_accum 4 --warmup_ratio 0.05 \
    --max_length $SFT_MAX_LENGTH --max_output_length $SFT_MAX_OUTPUT \
    --zero_stage 2 --gradient_checkpointing \
    --resume_from_checkpoint auto \
    --save_steps 200 --logging_steps 10

if [ ! -f "$OUTPUT_DIR/final_model/adapter_config.json" ] && \
   [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: 训练未产出 final_model"; exit 1
fi
notify "Step1 完成: FOARL CVRP100 SFT 训练"

# ── Step 2: merge LoRA (复用 Distill 的 merge_adapter.py, 就地合并进 final_model) ──
echo ""; echo ">>> Step 2: 合并 LoRA adapter..."
if [ -f "$OUTPUT_DIR/final_model/adapter_config.json" ]; then
    ( cd "$DISTILL_DIR" && python stage1_solution/merge_adapter.py \
        --adapter_path "$FOARL_DIR/$OUTPUT_DIR/final_model" )
    echo "  ✓ 合并完成: $OUTPUT_DIR/final_model"
else
    echo "  final_model 已是完整权重, 跳过合并"
fi
if [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: merge 后无 config.json, 中止 eval"; exit 1
fi
# 权重非空校验 (防 ZeRO-3+LoRA 空壳)
_W=$(find "$OUTPUT_DIR/final_model" -maxdepth 1 -name '*.safetensors' -size +0c 2>/dev/null | head -1)
[ -n "$_W" ] || { echo "ERROR: 合并权重为空 (无非空 safetensors)"; exit 1; }
notify "Step2 完成: FOARL adapter 合并"

# ── Step 3: bo1 评估 (vLLM TP=1, prompt_mode foarl) ───────────────────────────
echo ""; echo ">>> Step 3: bo1 评估 (贪心, vLLM TP=1)..."
mkdir -p "$EVAL_SAVE_DIR"
EVAL_GPU=($(wait_for_free_gpus 1))
echo "  eval 使用 GPU: ${EVAL_GPU[0]}  (num_test=$EVAL_NUM_TEST, seed=9999 固定)"

cd "$MASK_DIR"
CUDA_VISIBLE_DEVICES=${EVAL_GPU[0]} python evaluate.py \
    --backend vllm --model_path "$FOARL_DIR/$OUTPUT_DIR/final_model" --tp_size 1 \
    --vllm_gpu_mem_util $EVAL_GPU_MEM \
    --problem $PROBLEM --problem_size $SIZE --num_test $EVAL_NUM_TEST \
    --prompt_mode foarl --model_type instruct \
    --num_samples 1 --max_completion_length $EVAL_MAX_COMPLETION \
    --save_dir "$EVAL_SAVE_DIR" --run_tag "foarl_SFT_BO1_cvrp100"
cd "$FOARL_DIR"

notify "✅ FOARL CVRP100 SFT+bo1 全部完成" "模型: $OUTPUT_DIR/final_model"
echo ""; echo "============================================================"
echo "  完成! $(date)"
echo "  合并模型: $OUTPUT_DIR/final_model"
echo "  bo1 评估: $EVAL_SAVE_DIR/foarl_SFT_BO1_cvrp100.json"
echo "    jq '.results[0]|{parse:.format_match_rate,feas:.global_feasibility_rate,dist:.avg_best_dist}' $EVAL_SAVE_DIR/foarl_SFT_BO1_cvrp100.json"
echo "============================================================"
