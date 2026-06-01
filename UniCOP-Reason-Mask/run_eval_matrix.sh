#!/bin/bash
# CVRP-20 eval 矩阵: {RL, SFT-template, base} × {BO1, BO8朴素, BO8+wave剪枝}
#   - 每个模型 2 次运行: BO1(greedy) + BO8/wave(合一)
#   - BO8/wave 一次生成 8 条样本, --bestofn 出朴素 best-of-k 曲线, --wave 出波次剪枝
# 串行(全部模型一张卡):
#   nohup bash run_eval_matrix.sh > eval_matrix.out 2>&1 &
# 三卡并行(每模型一张卡):
#   ONLY=RL   GPU=0 nohup bash run_eval_matrix.sh > eval_RL.out   2>&1 &
#   ONLY=SFT  GPU=1 nohup bash run_eval_matrix.sh > eval_SFT.out  2>&1 &
#   ONLY=BASE GPU=2 nohup bash run_eval_matrix.sh > eval_BASE.out 2>&1 &
# 可调环境变量: NUM_TEST(默认100) TEMP(0.6) MAXLEN(6144) GPU(0) ONLY(空=全部|RL|SFT|BASE)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = MASK_DIR
source "$(dirname "$SCRIPT_DIR")/paths.sh"                   # 注入 UNICOP_ROOT/MASK_DIR/DISTILL_DIR/POMO_*
cd "$SCRIPT_DIR"

RL_MODEL="$MASK_DIR/output_v5/cvrp_n20/merged_model"
SFT_MODEL="$DISTILL_DIR/output_sft_qwen3_template_cvrp20/final_model"
BASE_MODEL="$UNICOP_ROOT/model/Qwen3-4B-Thinking-2507"
# POMO_BASELINE_DIR / POMO_CKPT_DIR 由 paths.sh 按主机注入

NUM_TEST=${NUM_TEST:-1000}     # 默认全 1000 (与 optimal 冻结集同 seed=9999 逐一对齐, gap 可控); smoke 可 NUM_TEST=100
TEMP=${TEMP:-0.6}             # Qwen3-thinking 推荐采样温度 (BO8)
MAXLEN=${MAXLEN:-6144}        # vllm max_model_len=8192 硬编码, prompt~1200, 勿超
GPU=${GPU:-0}
ONLY=${ONLY:-}               # 空=跑全部三个模型; 或 RL / SFT / BASE (三卡并行用)
SAVE_DIR="$SCRIPT_DIR/eval_results_matrix"; LOG_DIR="$SCRIPT_DIR/eval_logs_matrix"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

# POMO ckpt 守卫: POMOPRM glob "*POMO_CVRP_n20" 取最新, 优先加载 MODEL_BEST.pt (回退 FINAL)
POMO_CVRP_DIR=$(ls -d "$POMO_CKPT_DIR"/*POMO_CVRP_n20 2>/dev/null | tail -1)
if [ -z "$POMO_CVRP_DIR" ] || { [ ! -f "$POMO_CVRP_DIR/MODEL_BEST.pt" ] && [ ! -f "$POMO_CVRP_DIR/MODEL_FINAL.pt" ]; }; then
  echo "❌ POMO ckpt 问题: 需要 $POMO_CKPT_DIR/*POMO_CVRP_n20/{MODEL_BEST,MODEL_FINAL}.pt (wave 剪枝必需)"
  echo "   匹配到目录: ${POMO_CVRP_DIR:-<无>}"
  [ -n "$POMO_CVRP_DIR" ] && { echo "   目录内容:"; ls -la "$POMO_CVRP_DIR"; }
  exit 1
fi

run() {  # run <tag> <model> <extra args...>
  local tag="$1"; local model="$2"; shift 2
  echo "[$(date '+%F %T')] >>> $tag"
  CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --backend vllm --model_path "$model" --tp_size 1 \
    --problem cvrp --problem_size 20 \
    --num_test "$NUM_TEST" --prompt_mode think --model_type reasoning \
    --max_completion_length "$MAXLEN" --save_dir "$SAVE_DIR" \
    "$@" > "$LOG_DIR/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] <<< $tag  (exit $?)  log: $LOG_DIR/${tag}.log"
}

WAVE=(--wave --bestofn --pomo_ckpt_dir "$POMO_CKPT_DIR" --pomo_baseline_dir "$POMO_BASELINE_DIR")

for pair in "RL:$RL_MODEL" "SFT:$SFT_MODEL" "BASE:$BASE_MODEL"; do
  name="${pair%%:*}"; model="${pair#*:}"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then continue; fi
  run "${name}_BO1"     "$model" --num_samples 1
  run "${name}_BO8wave" "$model" --num_samples 8 --temperature "$TEMP" "${WAVE[@]}"
done
echo "[$(date '+%F %T')] 完成 (ONLY=${ONLY:-ALL}). 结果 JSON: $SAVE_DIR  日志: $LOG_DIR"
