#!/bin/bash
# submit_eval_v6c_instruct.sh — FOARL instruct 臂 CVRP20 eval (zhuoyi SLURM, 4 卡 large QOS)
#   两个 4 卡 job 之一 (另一个: submit_eval_v6c_thinking.sh)。各自独立 sbatch, 可同时排。
#   模型: FOARL GRPO 产物 (Qwen3-4B-Instruct 基座), --prompt_mode foarl 直接解(无 think_budget)
#     /homes/zhuoyi/zijianliu/UniCOP/FOARL/output_grpo_foarl_cvrp20/merged_model
#   同一批 seed=9999 的 NUM_TEST 个 CVRP20 实例 (evaluate.py 硬编码 9999, 与 thinking job 逐实例一致),
#   BO1→BO8 顺序, 4 shard 数据并行(GPU0-3, TP=1), 跑完 merge_shards 合并。
#   结果: eval_v6c_vs_foarl/{instruct_bo1,instruct_bo8}/MERGED.json
#   注: instruct 直接答案无 wave 意义, 本 job 不需要 POMO ckpt。
#   提交: sbatch submit_eval_v6c_instruct.sh

#SBATCH --qos=large
#SBATCH --gpus=4
#SBATCH --job-name=zijia_eval_v6c_inst
#SBATCH --comment="zijianliu, FOARL instruct eval, do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_v6c_inst_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_v6c_inst_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

INSTRUCT_MODEL="${INSTRUCT_MODEL:-/homes/zhuoyi/zijianliu/UniCOP/FOARL/output_grpo_foarl_cvrp20/merged_model}"
NUM_TEST="${NUM_TEST:-1000}"      # 与 optimal 对齐的冻结集; 勿改小
SAMPLE_TEMP="${SAMPLE_TEMP:-0.6}"  # BO8 采样温度 (勿用 TEMP: 系统环境变量名, 会拿到旧值/tmp路径)

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
set -uo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SAVE_BASE="$(pwd)/eval_v6c_vs_foarl"; LOG_DIR="$SAVE_BASE/logs"; mkdir -p "$LOG_DIR"

[ -d "$INSTRUCT_MODEL" ] || { echo "❌ 模型不存在: $INSTRUCT_MODEL"; exit 1; }

# ── GPU 占用预检 (4 卡): 分到的卡被占 → exclude 本节点重投本 job ──
export SUBMIT_SCRIPT="$(pwd)/submit_eval_v6c_instruct.sh"
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

echo "############## FOARL instruct 臂 eval ##############  $(date '+%F %T')"
echo "  INSTRUCT_MODEL=$INSTRUCT_MODEL"
echo "  NUM_TEST=$NUM_TEST SAMPLE_TEMP=$SAMPLE_TEMP"

run_sharded instruct_bo1  "$INSTRUCT_MODEL" 0,1,2,3 instruct foarl 2048 0 --num_samples 1
run_sharded instruct_bo8  "$INSTRUCT_MODEL" 0,1,2,3 instruct foarl 2048 0 --num_samples 8 --temperature "$SAMPLE_TEMP" --bestofn

echo "============================================================"
echo "  ✅ instruct 臂完成  $(date '+%F %T')"
echo "  结果: $SAVE_BASE/{instruct_bo1,instruct_bo8}/MERGED.json"
echo "  日志: $LOG_DIR/"
echo "============================================================"
