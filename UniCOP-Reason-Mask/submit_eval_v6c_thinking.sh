#!/bin/bash
# submit_eval_v6c_thinking.sh — v6_complete thinking 臂 CVRP20 eval (zhuoyi SLURM, 4 卡 large QOS)
#   两个 4 卡 job 之一 (另一个: submit_eval_v6c_instruct.sh)。各自独立 sbatch, 可同时排。
#   模型: v6_complete merged (Qwen3-4B-Instruct 基座训成 think 范式), --prompt_mode think +
#     budget forcing(--think_budget 6400, 治循环截断, 不惩罚重复)
#   注: vLLM 0.7.3 无 Qwen3 原生实现 → 回退 Transformers backend → 必须 --enforce_eager
#       (CUDA graph capture 阶段会 OOM, 见 job 17:35 OOM), 且 max_seq_len 不宜过长。
#     /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/v6_complete/merged_model
#   同一批 seed=9999 的 NUM_TEST 个 CVRP20 实例 (evaluate.py 硬编码 9999, 与 instruct job 逐实例一致),
#   BO1→BO8(+WAVE) 顺序, 4 shard 数据并行(GPU0-3, TP=1), 跑完 merge_shards 合并。
#   结果: eval_v6c_vs_foarl/{thinking_bo1,thinking_bo8wave}/MERGED.json
#   注: wave 是思维步 PRM 剪枝, 本 job 需要 POMO CVRP n20 ckpt (paths.sh 注入)。
#   提交: sbatch submit_eval_v6c_thinking.sh

#SBATCH --qos=large
#SBATCH --gpus=4
#SBATCH --job-name=zijia_eval_v6c_think
#SBATCH --comment="zijianliu, v6_complete thinking eval, do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_v6c_think_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_v6c_think_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

THINKING_MODEL="${THINKING_MODEL:-/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/v6_complete/merged_model}"
NUM_TEST="${NUM_TEST:-1000}"      # 与 optimal 对齐的冻结集; 勿改小
SAMPLE_TEMP="${SAMPLE_TEMP:-0.6}"  # BO8 采样温度 (勿用 TEMP: 系统环境变量名, 会拿到旧值/tmp路径)
THINK_BUDGET="${THINK_BUDGET:-6400}"    # budget forcing 预算 (6400 足够, 不需 10000; max_seq_len=6400+1024+1536=8960, 省 KV cache)

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
set -uo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source "$(dirname "$(pwd)")/paths.sh"     # 注入 POMO_CKPT_DIR / POMO_BASELINE_DIR

SAVE_BASE="$(pwd)/eval_v6c_vs_foarl"; LOG_DIR="$SAVE_BASE/logs"; mkdir -p "$LOG_DIR"

# POMO ckpt 守卫 (wave 必需)
POMO_CVRP_DIR=$(ls -d "$POMO_CKPT_DIR"/*POMO_CVRP_n20 2>/dev/null | tail -1)
if [ -z "$POMO_CVRP_DIR" ] || { [ ! -f "$POMO_CVRP_DIR/MODEL_BEST.pt" ] && [ ! -f "$POMO_CVRP_DIR/MODEL_FINAL.pt" ]; }; then
    echo "❌ 缺 POMO CVRP ckpt: $POMO_CKPT_DIR/*POMO_CVRP_n20/{MODEL_BEST,MODEL_FINAL}.pt (wave 必需)"
    exit 1
fi
WAVE_ARGS=(--wave --bestofn --pomo_ckpt_dir "$POMO_CKPT_DIR" --pomo_baseline_dir "$POMO_BASELINE_DIR")

[ -d "$THINKING_MODEL" ] || { echo "❌ 模型不存在: $THINKING_MODEL"; exit 1; }

# ── GPU 占用预检 (4 卡): 分到的卡被占 → exclude 本节点重投本 job ──
export SUBMIT_SCRIPT="$(pwd)/submit_eval_v6c_thinking.sh"
export BASE_EXCLUDE=""
source "$(pwd)/preflight_gpu.sh"
preflight_gpu_or_resubmit

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
        # ── shard 级断点续跑: 该 shard 的确定性 JSON 已存在且有效 → 跳过, 不重跑 ──
        local out_json="$sd/${tag}_shard${s}of${nsh}.json"
        if [ -f "$out_json" ] && python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('results') and d['results'][0].get('n_eval',0)>0 else 1)" "$out_json" 2>/dev/null; then
            echo "[$(date '+%T')] ⏭️  $tag shard$s 已完成, 跳过 ($out_json)"
            continue
        fi
        # --run_tag 确定性命名: 重跑覆盖同名, merge 不会把同 shard 多次时间戳文件重复计数
        CUDA_VISIBLE_DEVICES="${G[$s]}" python evaluate.py \
            --backend vllm --model_path "$model" --tp_size 1 \
            --num_shards "$nsh" --shard_id "$s" --run_tag "$tag" \
            --problem cvrp --problem_size 20 --num_test "$NUM_TEST" \
            --prompt_mode "$pmode" --model_type "$mtype" \
            --max_completion_length "$maxlen" --vllm_gpu_mem_util 0.8 --enforce_eager \
            --save_dir "$sd" $tb_flag "$@" \
            > "$LOG_DIR/${tag}_shard${s}.log" 2>&1 &
        pids+=($!)
    done
    local ok=0
    if [ "${#pids[@]}" -gt 0 ]; then
        for p in "${pids[@]}"; do wait "$p" || ok=1; done
    else
        echo "[$(date '+%T')] $tag: 所有 shard 已完成 (全部跳过), 直接合并。"
    fi
    if [ "$ok" -ne 0 ]; then echo "[$(date '+%T')] ⚠️ $tag 有 shard 非零退出, 详见 $LOG_DIR/${tag}_shard*.log"; fi
    python merge_shards.py --glob "$sd/${tag}_shard*of${nsh}.json" --out "$sd/MERGED.json" \
        > "$LOG_DIR/${tag}_merge.log" 2>&1 \
        && echo "[$(date '+%T')] <<< $tag 合并完成 → $sd/MERGED.json" \
        || echo "[$(date '+%T')] ⚠️ $tag 合并失败, 详见 $LOG_DIR/${tag}_merge.log"
}

echo "############## v6_complete thinking 臂 eval ##############  $(date '+%F %T')"
echo "  THINKING_MODEL=$THINKING_MODEL"
echo "  NUM_TEST=$NUM_TEST SAMPLE_TEMP=$SAMPLE_TEMP THINK_BUDGET=$THINK_BUDGET"

run_sharded thinking_bo1     "$THINKING_MODEL" 0,1,2,3 reasoning think "$THINK_BUDGET" "$THINK_BUDGET" --num_samples 1
run_sharded thinking_bo8wave "$THINKING_MODEL" 0,1,2,3 reasoning think "$THINK_BUDGET" "$THINK_BUDGET" --num_samples 8 --temperature "$SAMPLE_TEMP" "${WAVE_ARGS[@]}"

echo "============================================================"
echo "  ✅ thinking 臂完成  $(date '+%F %T')"
echo "  结果: $SAVE_BASE/{thinking_bo1,thinking_bo8wave}/MERGED.json"
echo "  日志: $LOG_DIR/"
echo "============================================================"
