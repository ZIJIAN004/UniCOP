#!/bin/bash
# submit_eval_base4shard.sh
# ── zhuoyi 集群: instruct / thinking 基线模型的 BO1 + BO8/wave eval (4 卡数据并行分片) ──
#
#   一个 job 评一个模型 (MODEL_KIND=thinking|instruct), 4 张卡各跑一个分片
#   (evaluate.py --num_shards 4 --shard_id 0..3, 各 TP=1; 所有分片生成同一批 seed=9999
#   实例、跨步取自己那份), 跑完用 merge_shards.py 精确合并 → 与单进程全量跑同口径。
#
#   阶段 (可 DO_BO1/DO_BO8WAVE=0 跳过):
#     BO1     : --num_samples 1 (evaluate.py:855 强制 temp=0 → 纯 greedy)
#     BO8wave : --num_samples 8 + --bestofn (朴素 best-of-k 曲线) + --wave (POMO PRM 波次剪枝)
#
#   thinking 特殊逻辑: budget forcing (evaluate.py 两段式生成):
#     --think_budget 10112 (=79×128, 10000 以上最接近的 128 倍数): think 段到此预算还没出
#     </think> 就强制注入 </think>\n\n 再用 --answer_budget 1024 生成答案段 → 治 thinking
#     循环导致的答案截断。instruct 不思考, 不开 budget forcing。
#
#   结果:
#     eval_results_matrix/THINKING_BO1_shard{0..3}of4.json + THINKING_BO1.json (合并后)
#     eval_results_matrix/INSTRUCT_BO8wave_shard{0..3}of4.json + INSTRUCT_BO8wave.json
#     日志: eval_logs_matrix/<TAG>_shard<i>.log
#   幂等: 有效的分片 JSON 跳过重跑; 合并文件存在且有效跳过合并 → 崩了直接重投
#
#   提交 (登录节点, git pull 之后, 两个模型各一个 job 并行排队):
#     sbatch --export=ALL,MODEL_KIND=thinking submit_eval_base4shard.sh
#     sbatch --export=ALL,MODEL_KIND=instruct submit_eval_base4shard.sh
#   可覆盖: NUM_TEST(1000) TEMP(thinking 0.6 / instruct 0.7) GPU_MEM(0.8)
#          THINK_BUDGET(10112) ANSWER_BUDGET(1024) MAXLEN_INSTRUCT(2048)
#          INSTRUCT_MODEL(默认 <models>/Qwen3-4B-Instruct-2507) DO_BO1(1) DO_BO8WAVE(1)

#SBATCH --qos=large
#SBATCH --gpus=4
#SBATCH --job-name=zijia_eval_base
#SBATCH --comment="zijianliu, base model BO1+BO8wave eval 4-shard, do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_base4shard_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/eval_base4shard_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
export BASE_MODEL_TYPE=qwen3_thinking
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ⚠️ conda activate 前别开 set -u (activate.d 引用未设变量会挂)
source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
set -uo pipefail
source ../paths.sh   # 注入 UNICOP_ROOT/POMO_*/BASE_MODEL_QWEN3 (本机 = astar-zhuoyi)

# ── 参数 ──────────────────────────────────────────────────────────
MODEL_KIND="${MODEL_KIND:-thinking}"      # thinking | instruct
NUM_TEST="${NUM_TEST:-1000}"              # 与 optimal 冻结集 seed=9999 逐一对齐, 勿改小
GPU_MEM="${GPU_MEM:-0.8}"
NSHARD=4; GPUS=(0 1 2 3)
DO_BO1="${DO_BO1:-1}"
DO_BO8WAVE="${DO_BO8WAVE:-1}"
THINK_BUDGET="${THINK_BUDGET:-10112}"     # =79×128; thinking 强制收尾预算
ANSWER_BUDGET="${ANSWER_BUDGET:-1024}"    # 强制 </think> 后答案段预算
MAXLEN_INSTRUCT="${MAXLEN_INSTRUCT:-2048}"  # instruct 不思考, 答案段短 (config 默认 512 太紧)
MODELS_DIR="$(dirname "$UNICOP_ROOT")/models"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-$MODELS_DIR/Qwen3-4B-Instruct-2507}"

case "$MODEL_KIND" in
  thinking)
    MODEL="$BASE_MODEL_QWEN3"             # paths.sh: zhuoyi 上 = <models>/Qwen3-4B-Thinking-2507
    MTYPE="reasoning"
    TEMP="${TEMP:-0.6}"                   # Qwen3-Thinking-2507 官方推荐
    ML="$THINK_BUDGET"
    EXTRA_GEN=(--think_budget "$THINK_BUDGET" --answer_budget "$ANSWER_BUDGET")
    ;;
  instruct)
    MODEL="$INSTRUCT_MODEL"
    MTYPE="instruct"                      # evaluate.py:1093 自动剥 system prompt 里的 <think> 指令
    TEMP="${TEMP:-0.7}"                   # Qwen3-Instruct-2507 官方推荐
    ML="$MAXLEN_INSTRUCT"
    EXTRA_GEN=()                          # instruct 不开 budget forcing
    ;;
  *) echo "[FATAL] MODEL_KIND='$MODEL_KIND' 应为 thinking 或 instruct"; exit 1;;
esac
KIND_TAG="$(echo "$MODEL_KIND" | tr '[:lower:]' '[:upper:]')"   # THINKING / INSTRUCT
SAVE_DIR="$(pwd)/eval_results_matrix"; LOG_DIR="$(pwd)/eval_logs_matrix"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ ! -f "$MODEL/config.json" ]; then
    echo "[FATAL] 模型不存在: $MODEL"
    [ "$MODEL_KIND" = "instruct" ] && echo "  下载: hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir $INSTRUCT_MODEL"
    exit 1
fi
echo "MODEL_KIND=$MODEL_KIND  模型=$MODEL"
echo "NUM_TEST=$NUM_TEST  TEMP=$TEMP  maxlen=$ML  分片=${NSHARD}×TP1  BO1=$DO_BO1 BO8wave=$DO_BO8WAVE"

# POMO ckpt 守卫 (仅 wave 需要; 与 run_eval_matrix.sh 同逻辑)
if [ "$DO_BO8WAVE" = "1" ]; then
  POMO_CVRP_DIR=$(ls -d "$POMO_CKPT_DIR"/*POMO_CVRP_n20 2>/dev/null | tail -1)
  if [ -z "$POMO_CVRP_DIR" ] || { [ ! -f "$POMO_CVRP_DIR/MODEL_BEST.pt" ] && [ ! -f "$POMO_CVRP_DIR/MODEL_FINAL.pt" ]; }; then
    echo "❌ POMO ckpt 缺失: 需要 $POMO_CKPT_DIR/*POMO_CVRP_n20/{MODEL_BEST,MODEL_FINAL}.pt (wave 必需)"
    echo "   zhuoyi 上若没有, 从 Zhihan 拷: /Data04/yangzhihan/lzj/POMO-Baseline/result/20260410_1831__POMO_CVRP_n20/"
    exit 1
  fi
fi
WAVE_ARGS=(--wave --bestofn --pomo_ckpt_dir "$POMO_CKPT_DIR" --pomo_baseline_dir "$POMO_BASELINE_DIR")

SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=${1:0:100}" \
        --data-urlencode "desp=${2:0:500}" > /dev/null 2>&1 || true
}

export SUBMIT_SCRIPT="$(pwd)/submit_eval_base4shard.sh"
export BASE_EXCLUDE=""
source "$(pwd)/preflight_gpu.sh"
preflight_gpu_or_resubmit

json_ok() {  # json_ok <file> → 0 当 JSON 有效且 n_eval>0
    [ -f "$1" ] && python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('results') and d['results'][0].get('n_eval',0)>0 else 1)" "$1" 2>/dev/null
}

run_stage() {  # run_stage <tag> <extra evaluate args...>  — 4 分片并行 + 合并
    local tag="$1"; shift
    local merged_json="$SAVE_DIR/${tag}.json"
    if json_ok "$merged_json"; then
        echo "[$tag] ⏭️ 合并结果已存在, 整个阶段跳过 ($merged_json)"; return 0
    fi
    echo "[$(date '+%F %T')] >>> $tag  (${NSHARD} 分片并行)"
    local pids=() s
    for s in $(seq 0 $((NSHARD-1))); do
        local shard_json="$SAVE_DIR/${tag}_shard${s}of${NSHARD}.json"
        if json_ok "$shard_json"; then
            echo "[$tag] ⏭️ shard $s 已完成, 跳过"; continue
        fi
        CUDA_VISIBLE_DEVICES="${GPUS[$s]}" python evaluate.py \
            --backend vllm --model_path "$MODEL" --tp_size 1 \
            --vllm_gpu_mem_util "$GPU_MEM" \
            --problem cvrp --problem_size 20 \
            --num_test "$NUM_TEST" --prompt_mode think --model_type "$MTYPE" \
            --max_completion_length "$ML" \
            --save_dir "$SAVE_DIR" --run_tag "$tag" \
            --num_shards "$NSHARD" --shard_id "$s" \
            "$@" > "$LOG_DIR/${tag}_shard${s}.log" 2>&1 &
        pids+=($!)
    done
    local p; for p in "${pids[@]}"; do wait "$p" || true; done
    # 校验所有分片 → 合并
    local missing=0
    for s in $(seq 0 $((NSHARD-1))); do
        json_ok "$SAVE_DIR/${tag}_shard${s}of${NSHARD}.json" || { echo "[$tag] ❌ shard $s 无效 (见 $LOG_DIR/${tag}_shard${s}.log)"; missing=1; }
    done
    if [ "$missing" = "1" ]; then echo "[$tag] ❌ 有分片失败, 不合并 (重投本 submit 只补失败分片)"; return 1; fi
    python merge_shards.py --glob "$SAVE_DIR/${tag}_shard*of${NSHARD}.json" --out "$merged_json"
    json_ok "$merged_json" || { echo "[$tag] ❌ 合并失败"; return 1; }
    echo "[$(date '+%F %T')] <<< $tag ✓ 合并完成: $merged_json"
}

FAILED=""
if [ "$DO_BO1" = "1" ]; then
    run_stage "${KIND_TAG}_BO1" --num_samples 1 ${EXTRA_GEN[@]+"${EXTRA_GEN[@]}"} \
        || FAILED="$FAILED ${KIND_TAG}_BO1"
fi
if [ "$DO_BO8WAVE" = "1" ]; then
    run_stage "${KIND_TAG}_BO8wave" --num_samples 8 --temperature "$TEMP" \
        ${EXTRA_GEN[@]+"${EXTRA_GEN[@]}"} "${WAVE_ARGS[@]}" \
        || FAILED="$FAILED ${KIND_TAG}_BO8wave"
fi

echo "============================================================"
if [ -z "$FAILED" ]; then
    echo "  ✅ $KIND_TAG 全部完成: $SAVE_DIR/${KIND_TAG}_BO1.json + ${KIND_TAG}_BO8wave.json"
    notify "✅ $KIND_TAG base eval 完成" "BO1+BO8wave 已合并  $(date '+%F %T')"
else
    echo "  ⚠️ 失败阶段:$FAILED (重投本 submit 幂等补跑)"
    notify "⚠️ $KIND_TAG base eval 部分失败" "$FAILED  $(date '+%F %T')"
    exit 1
fi
