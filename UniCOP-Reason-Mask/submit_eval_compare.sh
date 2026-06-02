#!/bin/bash
# submit_eval_compare.sh — instruct(FOARL prompt) vs thinking(think prompt) 能力对比 (zhuoyi SLURM)
#   同一批 seed=9999 的 NUM_TEST 个 CVRP20 实例, 各用各 SFT prompt:
#     instruct 臂: Qwen3-4B-Instruct-2507 + --prompt_mode foarl, 直接解(无 think_budget)
#     thinking 臂: Qwen3-4B-Thinking-2507 + --prompt_mode think + budget forcing(--think_budget 10000,
#                  治循环截断, 不惩罚重复)
#   两臂并行(instruct=GPU0-3, thinking=GPU4-7), 每臂内 BO1→BO8 顺序, 各 4 shard 数据并行(TP=1),
#   跑完自动 merge_shards 合并。结果在 eval_compare/<tag>/MERGED.json。
#   注: wave 是思维步 PRM 剪枝, 仅 thinking 臂加 --wave; instruct 直接答案无 wave 意义。
#   提交: sbatch submit_eval_compare.sh   (8 卡需 large QOS)

#SBATCH --qos=large
#SBATCH --gpus=8
#SBATCH --job-name=zijia_eval_cmp
#SBATCH --comment="zijianliu, instruct vs thinking capability eval, do not cancel"
#SBATCH --exclude=canele1
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_cmp_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_cmp_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

INSTRUCT_MODEL="${INSTRUCT_MODEL:-/homes/zhuoyi/zijianliu/models/Qwen3-4B-Instruct-2507}"
THINKING_MODEL="${THINKING_MODEL:-/homes/zhuoyi/zijianliu/models/Qwen3-4B-Thinking-2507}"
NUM_TEST="${NUM_TEST:-1000}"      # 与 optimal 对齐的冻结集; 勿改小
TEMP="${TEMP:-0.6}"               # BO8 采样温度
THINK_BUDGET="${THINK_BUDGET:-10000}"   # thinking 臂 budget forcing 预算 (确保推理充分)

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
set -uo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source "$(dirname "$(pwd)")/paths.sh"     # 注入 POMO_CKPT_DIR / POMO_BASELINE_DIR

SAVE_BASE="$(pwd)/eval_compare"; LOG_DIR="$SAVE_BASE/logs"; mkdir -p "$LOG_DIR"

# POMO ckpt 守卫 (wave 必需)
POMO_CVRP_DIR=$(ls -d "$POMO_CKPT_DIR"/*POMO_CVRP_n20 2>/dev/null | tail -1)
if [ -z "$POMO_CVRP_DIR" ] || { [ ! -f "$POMO_CVRP_DIR/MODEL_BEST.pt" ] && [ ! -f "$POMO_CVRP_DIR/MODEL_FINAL.pt" ]; }; then
    echo "❌ 缺 POMO CVRP ckpt: $POMO_CKPT_DIR/*POMO_CVRP_n20/{MODEL_BEST,MODEL_FINAL}.pt (wave 必需)"
    exit 1
fi
WAVE_ARGS=(--wave --bestofn --pomo_ckpt_dir "$POMO_CKPT_DIR" --pomo_baseline_dir "$POMO_BASELINE_DIR")

for m in "$INSTRUCT_MODEL" "$THINKING_MODEL"; do
    [ -d "$m" ] || { echo "❌ 模型不存在: $m"; exit 1; }
done

# run_sharded <tag> <model> <gpus_csv> <model_type> <prompt_mode> <maxlen> <think_budget> <stage_args...>
run_sharded() {
    local tag=$1 model=$2 gpus=$3 mtype=$4 pmode=$5 maxlen=$6 tbudget=$7; shift 7
    local sd="$SAVE_BASE/$tag"; mkdir -p "$sd"
    IFS=',' read -ra G <<< "$gpus"
    local nsh=${#G[@]}
    local tb_flag=""
    [ "$tbudget" -gt 0 ] && tb_flag="--think_budget $tbudget --answer_budget 1024"
    echo "[$(date '+%T')] >>> $tag : $nsh-shard 数据并行 (GPU $gpus)"
    local pids=()
    local s
    for s in "${!G[@]}"; do
        CUDA_VISIBLE_DEVICES="${G[$s]}" python evaluate.py \
            --backend vllm --model_path "$model" --tp_size 1 \
            --num_shards "$nsh" --shard_id "$s" \
            --problem cvrp --problem_size 20 --num_test "$NUM_TEST" \
            --prompt_mode "$pmode" --model_type "$mtype" \
            --max_completion_length "$maxlen" --vllm_gpu_mem_util 0.8 \
            --save_dir "$sd" $tb_flag "$@" \
            > "$LOG_DIR/${tag}_shard${s}.log" 2>&1 &
        pids+=($!)
    done
    local ok=0
    for p in "${pids[@]}"; do wait "$p" || ok=1; done
    if [ "$ok" -ne 0 ]; then echo "[$(date '+%T')] ⚠️ $tag 有 shard 非零退出, 详见 $LOG_DIR/${tag}_shard*.log"; fi
    python merge_shards.py --glob "$sd/*_shard*of${nsh}.json" --out "$sd/MERGED.json" \
        > "$LOG_DIR/${tag}_merge.log" 2>&1 \
        && echo "[$(date '+%T')] <<< $tag 合并完成 → $sd/MERGED.json" \
        || echo "[$(date '+%T')] ⚠️ $tag 合并失败, 详见 $LOG_DIR/${tag}_merge.log"
}

echo "############## instruct vs thinking 能力对比 ##############  $(date '+%F %T')"
echo "  NUM_TEST=$NUM_TEST TEMP=$TEMP THINK_BUDGET=$THINK_BUDGET"

# 两臂并行: instruct(GPU0-3, foarl, 无budget) | thinking(GPU4-7, think, budget forcing)
(
    run_sharded instruct_bo1     "$INSTRUCT_MODEL" 0,1,2,3 instruct foarl 2048 0 --num_samples 1
    run_sharded instruct_bo8     "$INSTRUCT_MODEL" 0,1,2,3 instruct foarl 2048 0 --num_samples 8 --temperature "$TEMP" --bestofn
) &
PI=$!
(
    run_sharded thinking_bo1     "$THINKING_MODEL" 4,5,6,7 reasoning think "$THINK_BUDGET" "$THINK_BUDGET" --num_samples 1
    run_sharded thinking_bo8wave "$THINKING_MODEL" 4,5,6,7 reasoning think "$THINK_BUDGET" "$THINK_BUDGET" --num_samples 8 --temperature "$TEMP" "${WAVE_ARGS[@]}"
) &
PT=$!
wait "$PI"; wait "$PT"

echo "============================================================"
echo "  ✅ 全部完成  $(date '+%F %T')"
echo "  结果: $SAVE_BASE/{instruct_bo1,instruct_bo8,thinking_bo1,thinking_bo8wave}/MERGED.json"
echo "  日志: $LOG_DIR/"
echo "============================================================"
