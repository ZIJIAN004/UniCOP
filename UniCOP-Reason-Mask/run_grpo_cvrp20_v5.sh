#!/bin/bash
# run_grpo_cvrp20_v5.sh — GRPO + POMO PRM · CVRP n=20 · reward_scheme=v5
#   7 卡 (1 vLLM + 6 训练) · v4 + hardgate distance + cov/cons 加权 A_feas
#
# v5 设计 (修 v4 7414 run 信号弱 + 冷启动):
#   - A_feas 加回 parse + cov + cons(hardgate cov_gate_v5=1.0) + format
#     cov+cons 占权重 82% (w_p=0.5, w_cov=2.5, w_cons=2.0, w_f=0.5)
#   - A_outcome 用 raw prob.get_tour_distance (不 repair) on strict
#     fully_feasible 子集 (parse + cov=1 + cons=1 + format=1, 子集 >=2 才启用)
#   - 前期 fully_feasible<2 时 A_outcome=0, 完全靠 A_feas + PRM 机会成本推可行性
#   - PRM 复用 v4: absolute base + tanh(R_step), 违例/重复及之后 step 游离
#   - 配 LR 2e-5 + warmup_ratio 0.01 (5 step) 加快收敛 (v4 7414 run grad_norm=0.05
#     远低于 clip 阈值, LR 信号空间充足)
#
# 输出目录: output_v5 (跟 output_v3/v4/mask 隔离)
# 端口 8004 错开 hardgate (8001) / v4 (8002) / mask (8002) / 6gpu (8000)
#
# SBATCH 提交:
#   sbatch submit_grpo_cvrp20_v5.sh
# 手动:
#   bash run_grpo_cvrp20_v5.sh

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
# BASE_MODEL_TYPE: 选哪条 SFT 产物作为 RL 起点 (不是加载原始基座!):
#   qwen3_thinking (默认) → output_sft_qwen3_template_cvrp20/final_model (Qwen3-4B SFT)
#   r1_distill            → output_sft_hybrid_cvrp20/final_model        (DeepSeek-R1-7B SFT)
# paths.sh 据此设采样参数 GEN_TEMPERATURE/TOP_P/TOP_K, trainer 自动读 env
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source "$(dirname "$_SELF_DIR")/paths.sh"

WORK_DIR="$MASK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/grpo_cvrp20_v5_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# ── 配置 ──────────────────────────────────────────────────────────────
PROBLEM="cvrp"
SIZE=20

case "$BASE_MODEL_TYPE" in
    r1_distill)     MODEL_BASE="$DISTILL_DIR/output_sft_hybrid_cvrp20/final_model" ;;
    qwen3_thinking) MODEL_BASE="$DISTILL_DIR/output_sft_qwen3_template_cvrp20/final_model" ;;
    *) echo "❌ 未知 BASE_MODEL_TYPE='$BASE_MODEL_TYPE'"; exit 1 ;;
esac
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ 基座模型不存在: $MODEL_BASE"
    exit 1
fi
echo "[RL 起点] SFT 产物 (非原始基座): $MODEL_BASE"
echo "[BASE_MODEL_TYPE=$BASE_MODEL_TYPE] qwen3_thinking→Qwen3-4B SFT | r1_distill→R1-7B SFT"

TOTAL_GPUS=7
VLLM_GPU=6
TRAIN_GPUS_CSV="0,1,2,3,4,5"
TRAIN_PROC=6

ZERO_STAGE=3
NUM_TRAIN=4000
OUTPUT_DIR_BASE="$WORK_DIR/output_v5"

VLLM_PORT=8004
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=8192
VLLM_DTYPE=bfloat16
VLLM_STARTUP_TIMEOUT=300

SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

VLLM_LOG="$LOG_DIR/vllm_${PROBLEM}${SIZE}_v5_$(date +%Y%m%d_%H%M%S).log"
VLLM_PID=""

start_vllm_server() {
    echo "[$(date '+%H:%M:%S')] 启动 vLLM server | GPU=$VLLM_GPU | port=$VLLM_PORT"
    PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    CUDA_HOME="$CUDA_HOME" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
        python "$WORK_DIR/utils/vllm_serve_logprobs.py" \
        --model "$MODEL_BASE" \
        --tensor_parallel_size 1 \
        --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --dtype "$VLLM_DTYPE" \
        --enable_prefix_caching True \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    local waited=0
    while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] ✗ vLLM server 启动失败,详见 $VLLM_LOG"
            tail -n 50 "$VLLM_LOG" || true
            return 1
        fi
        if curl -s "http://localhost:${VLLM_PORT}/health/" > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] ✓ vLLM server 就绪 (pid=$VLLM_PID, 用时 ${waited}s)"
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
    done
    echo "[$(date '+%H:%M:%S')] ✗ vLLM server 启动超时 (${VLLM_STARTUP_TIMEOUT}s)"
    kill "$VLLM_PID" 2>/dev/null || true
    return 1
}

stop_vllm_server() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] 关闭 vLLM server (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}

TRAINING_COMPLETED=0
on_exit() {
    local exit_code=$?
    stop_vllm_server
    if [ "$TRAINING_COMPLETED" != "1" ] && [ "$exit_code" != "0" ]; then
        notify "❌ CVRP20 GRPO v5 异常退出" \
"退出码: $exit_code
时间: $(date '+%Y-%m-%d %H:%M:%S')
日志末尾:
$(tail -n 20 "$LOG_FILE" 2>/dev/null || echo '(无日志)')"
    fi
}
trap 'on_exit' EXIT INT TERM

cd "$WORK_DIR"

echo "============================================================"
echo "  GPU 拓扑"
echo "============================================================"
nvidia-smi topo -m 2>&1 || echo "(nvidia-smi topo unavailable)"
echo ""
echo "============================================================"
echo "  GRPO + POMO PRM · CVRP n=$SIZE · 7 卡 · reward_scheme=v5"
echo "  BASE_MODEL_TYPE: $BASE_MODEL_TYPE  (T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K)"
echo "  RL 起点:   $MODEL_BASE (SFT 产物, 非原始基座)"
echo "  GPU:       1 vLLM (GPU $VLLM_GPU) + $TRAIN_PROC 训练 (GPU $TRAIN_GPUS_CSV)"
echo "  ZeRO:      stage $ZERO_STAGE | gradient_checkpointing on"
echo "  Reward:    v5 (v4 + hardgate distance + cov/cons 加权)"
echo "             A_feas = w_p×parse + w_cov×cov + w_cons×cons(gate) + w_f×format"
echo "                      权重 0.5/2.5/2.0/0.5, cov_gate=1.0 hardgate"
echo "             A_outcome = z(-raw_distance) on strict fully_feasible 子集 >=2"
echo "             PRM = absolute base 1.5 + tanh(R_step) (跟 v4 共用)"
echo "  LR:        2e-5 (v4 加倍, 配 warmup 5 step 快收敛)"
echo "  Warmup:    0.01 × 500 step = 5 step"
echo "  输出目录:  $OUTPUT_DIR_BASE"
echo "  整除检查:  per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)),  整除 num_generations (8) ? $(( (4 * TRAIN_PROC) % 8 == 0 ))"
echo "  时间:      $(date)"
echo "============================================================"

if [ $((4 * TRAIN_PROC % 8)) -ne 0 ]; then
    echo "[FATAL] 整除失败: per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)) 必须整除 num_generations=8"
    exit 1
fi

notify "🚀 CVRP20 GRPO v5 启动" \
"reward: v5 (hardgate distance + cov/cons 加权)
LR 2e-5, warmup 5 step
基座: $MODEL_BASE
GPU: 1 vLLM + $TRAIN_PROC 训练
开始: $(date '+%Y-%m-%d %H:%M:%S')"

if ! start_vllm_server; then
    echo "[FATAL] vLLM server 启动失败"
    exit 1
fi

TRAIN_LOG="$LOG_DIR/train_${PROBLEM}${SIZE}_v5_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%H:%M:%S')] 启动训练 ($TRAIN_PROC 卡: GPU=$TRAIN_GPUS_CSV)"
echo "  log: $TRAIN_LOG"

PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_HOME="$CUDA_HOME" \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS_CSV" \
    python -m accelerate.commands.launch --num_processes "$TRAIN_PROC" "$WORK_DIR/train.py" \
    --problem "$PROBLEM" \
    --problem_size "$SIZE" \
    --num_train "$NUM_TRAIN" \
    --model "$MODEL_BASE" \
    --num_gpus "$TRAIN_PROC" \
    --zero_stage "$ZERO_STAGE" \
    --gradient_checkpointing \
    --output_dir "$OUTPUT_DIR_BASE" \
    --pomo_ckpt_dir "$POMO_CKPT_DIR" \
    --pomo_baseline_dir "$POMO_BASELINE_DIR" \
    --pipd_ckpt_dir "$PIPD_CKPT_DIR" \
    --pipd_dir "$PIPD_DIR" \
    --vllm_server_host "localhost" \
    --vllm_server_port "$VLLM_PORT" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EC=${PIPESTATUS[0]}

if [ $TRAIN_EC -eq 0 ]; then
    notify "✅ CVRP20 GRPO v5 训练完成" \
"output: $OUTPUT_DIR_BASE/${PROBLEM}_n${SIZE}/final_model
结束: $(date '+%Y-%m-%d %H:%M:%S')"
    TRAINING_COMPLETED=1
fi

stop_vllm_server

echo ""
echo "============================================================"
echo "  完成! exit=$TRAIN_EC  $(date)"
echo "  训练日志: $TRAIN_LOG"
echo "  vLLM 日志: $VLLM_LOG"
echo "  模型输出: $OUTPUT_DIR_BASE/${PROBLEM}_n${SIZE}/final_model"
echo "============================================================"

exit $TRAIN_EC
