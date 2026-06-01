#!/bin/bash
# CVRP-20 eval 矩阵: {RL, SFT-template, base} × {BO1, BO8朴素, BO8+wave剪枝}
#   - 每个模型 2 次运行: BO1(greedy) + BO8/wave(合一)
#   - BO8/wave 一次生成 8 条样本, --bestofn 出朴素 best-of-k 曲线, --wave 出波次剪枝
# 用法:
#   conda activate <unicop env>
#   nohup bash run_eval_matrix.sh > eval_matrix.out 2>&1 &
# 可调环境变量: NUM_TEST(默认100) TEMP(0.6) MAXLEN(6144) GPU(0)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = MASK_DIR
source "$(dirname "$SCRIPT_DIR")/paths.sh"                   # 注入 UNICOP_ROOT/MASK_DIR/DISTILL_DIR/POMO_*
cd "$SCRIPT_DIR"

RL_MODEL="$MASK_DIR/output_v5/cvrp_n20/merged_model"
SFT_MODEL="$DISTILL_DIR/output_sft_qwen3_template_cvrp20/final_model"
BASE_MODEL="$UNICOP_ROOT/model/Qwen3-4B-Thinking-2507"
# POMO_BASELINE_DIR / POMO_CKPT_DIR 由 paths.sh 按主机注入

NUM_TEST=${NUM_TEST:-100}      # 先 100 看趋势, 终版可 1000
TEMP=${TEMP:-0.6}             # Qwen3-thinking 推荐采样温度 (BO8)
MAXLEN=${MAXLEN:-6144}        # vllm max_model_len=8192 硬编码, prompt~1200, 勿超
GPU=${GPU:-0}
SAVE_DIR="$SCRIPT_DIR/eval_results_matrix"; LOG_DIR="$SCRIPT_DIR/eval_logs_matrix"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

# POMO ckpt 守卫: wave 必需, 缺了直接停, 别白跑
if [ ! -f "$POMO_CKPT_DIR/cvrp_n20/MODEL_BEST.pt" ]; then
  echo "❌ 找不到 POMO ckpt: $POMO_CKPT_DIR/cvrp_n20/MODEL_BEST.pt (wave 剪枝必需)"
  echo "   核对 POMO-Baseline/result 下 CVRP-20 ckpt 路径后再跑."
  exit 1
fi

run() {  # run <tag> <model> <extra args...>
  local tag="$1"; local model="$2"; shift 2
  echo "[$(date '+%F %T')] >>> $tag"
  CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --backend vllm --model_path "$model" \
    --problem cvrp --problem_size 20 \
    --num_test "$NUM_TEST" --prompt_mode think --model_type reasoning \
    --max_completion_length "$MAXLEN" --save_dir "$SAVE_DIR" \
    "$@" > "$LOG_DIR/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] <<< $tag  (exit $?)  log: $LOG_DIR/${tag}.log"
}

WAVE=(--wave --bestofn --pomo_ckpt_dir "$POMO_CKPT_DIR" --pomo_baseline_dir "$POMO_BASELINE_DIR")

for pair in "RL:$RL_MODEL" "SFT:$SFT_MODEL" "BASE:$BASE_MODEL"; do
  name="${pair%%:*}"; model="${pair#*:}"
  run "${name}_BO1"     "$model" --num_samples 1
  run "${name}_BO8wave" "$model" --num_samples 8 --temperature "$TEMP" "${WAVE[@]}"
done
echo "全部完成. 结果 JSON: $SAVE_DIR  日志: $LOG_DIR"
