#!/bin/bash
# run_grpo_cvrp20_v4_mask.sh — GRPO + POMO PRM · CVRP n=20 · reward_scheme=v4 + CVRP mask
#   7 卡 (1 vLLM + 6 训练) · v4 reward + vLLM logits_processor 强制 cov=1 不重复
#
# 设计:
#   - reward_scheme=v4 (simplified absolute PRM + repaired distance)
#   - mask 启用: vLLM 端 utils/vllm_serve_logprobs.py --mask_enabled --mask_n N
#                trainer 端 --use_mask --mask_n N (sanity check + log 区分)
#   - mask 保证: cov=1 (走完所有 customer), 不重复. cap 违例和 format 仍可能错,
#                由 v4 repaired_distance + cons 间接惩罚.
#   - 输出目录: output_v4_mask (跟 output_v4 / output_mask 隔离)
#   - 端口 8003 错开 v4 (8002), mask (8002), hardgate (8001), 6gpu (8000)
#
# Mask 规则: CVRP-LLM-Mask-完整规则.md (规则 1-5: select 强 mask / 禁结束 /
#                                         visited==n 禁 | / visited<n 禁 all / 禁 Verification)
# 训练端 IS 校正: mask 位置 ratio=1 (跳过 IS, 见 CVRP-LLM-Mask-完整规则.md 第 9 节)
#
# SBATCH 提交:
#   sbatch submit_grpo_cvrp20_v4_mask.sh
# 手动:
#   bash run_grpo_cvrp20_v4_mask.sh

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
# BASE_MODEL_TYPE: r1_distill (默认, 7B DeepSeek) 或 qwen3_thinking (4B Qwen3-Thinking)
# paths.sh 据此设 BASE_MODEL + 采样参数 GEN_TEMPERATURE/TOP_P/TOP_K, trainer 自动读 env
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-r1_distill}"
source "$(dirname "$_SELF_DIR")/paths.sh"

WORK_DIR="$MASK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/grpo_cvrp20_v4_mask_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# zhuoyi 拓扑无 NVLink, NCCL P2P + SHM 必须禁用
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
echo "[MODEL_BASE] $MODEL_BASE"

# GPU 分配: 7 卡 = 1 vLLM + 6 训练 (同 v4 / hardgate 脚本)
TOTAL_GPUS=7
VLLM_GPU=6
TRAIN_GPUS_CSV="0,1,2,3,4,5"
TRAIN_PROC=6

ZERO_STAGE=3
NUM_TRAIN=4000
# 输出隔离: v4+mask 独立目录, 不污染 v4 / mask 单独 run
OUTPUT_DIR_BASE="$WORK_DIR/output_v4_mask"

# 端口 8003 错开 hardgate (8001) / v4 (8002) / mask (8002) / 6gpu (8000)
VLLM_PORT=8003
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=5120
VLLM_DTYPE=bfloat16
VLLM_STARTUP_TIMEOUT=300

# ── Mask 超参 ─────────────────────────────────────────────────────────
MASK_ENABLED=1   # 1=启用 (本脚本默认), 0=退化为纯 v4
MASK_N=20        # CVRP customer 数
MASK_DEBUG=0     # 1=vLLM stderr 输出每次 mask 触发 (慢, 仅 debug 用)

SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

# ── 启动/关闭 vLLM server ─────────────────────────────────────────
VLLM_LOG="$LOG_DIR/vllm_${PROBLEM}${SIZE}_v4_mask_$(date +%Y%m%d_%H%M%S).log"
VLLM_PID=""

start_vllm_server() {
    echo "[$(date '+%H:%M:%S')] 启动 vLLM server | GPU=$VLLM_GPU | port=$VLLM_PORT | mask_enabled=$MASK_ENABLED"
    local mask_args=""
    if [ "$MASK_ENABLED" = "1" ]; then
        mask_args="--mask_enabled --mask_n $MASK_N"
        if [ "$MASK_DEBUG" = "1" ]; then
            mask_args="$mask_args --mask_debug"
        fi
    fi
    PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    CUDA_HOME="$CUDA_HOME" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
        python "$WORK_DIR/utils/vllm_serve_logprobs.py" \
        $mask_args \
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
        notify "❌ CVRP20 GRPO v4+Mask 异常退出" \
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
echo "  GRPO + POMO PRM · CVRP n=$SIZE · 7 卡 · reward_scheme=v4 + Mask"
echo "  BASE_MODEL_TYPE: $BASE_MODEL_TYPE  (T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K)"
echo "  基座模型:  $MODEL_BASE"
echo "  GPU:       1 vLLM (GPU $VLLM_GPU) + $TRAIN_PROC 训练 (GPU $TRAIN_GPUS_CSV)"
echo "  ZeRO:      stage $ZERO_STAGE | gradient_checkpointing on"
echo "  Reward:    v4 (simplified: absolute PRM + repaired distance)"
echo "             A_feas = parse+format only, A_outcome = repaired_distance"
echo "             PRM normal a_proc = 1.5 + tanh(R_step)"
echo "  Mask:      $([ "$MASK_ENABLED" = "1" ] && echo "ENABLED (n=$MASK_N, debug=$MASK_DEBUG) — 强制 cov=1, 不重复" || echo "DISABLED (退化为纯 v4)")"
echo "  IS skip:   mask 位置 ratio=1 (avoid post-mask logp pollution)"
echo "  输出目录:  $OUTPUT_DIR_BASE"
echo "  整除检查:  per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)),  整除 num_generations (8) ? $(( (4 * TRAIN_PROC) % 8 == 0 ))"
echo "  时间:      $(date)"
echo "============================================================"

if [ $((4 * TRAIN_PROC % 8)) -ne 0 ]; then
    echo "[FATAL] 整除失败: per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)) 必须整除 num_generations=8"
    exit 1
fi

notify "🚀 CVRP20 GRPO v4+Mask 启动" \
"reward: v4 simplified + CVRP mask
mask: enabled=$MASK_ENABLED, n=$MASK_N
基座: $MODEL_BASE
GPU: 1 vLLM + $TRAIN_PROC 训练
开始: $(date '+%Y-%m-%d %H:%M:%S')"

if ! start_vllm_server; then
    echo "[FATAL] vLLM server 启动失败"
    exit 1
fi

TRAIN_LOG="$LOG_DIR/train_${PROBLEM}${SIZE}_v4_mask_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%H:%M:%S')] 启动训练 ($TRAIN_PROC 卡: GPU=$TRAIN_GPUS_CSV)"
echo "  log: $TRAIN_LOG"

# Trainer 端 --use_mask 透传 config.use_mask, 让 sanity check 跟 vLLM server 配对.
# 即使 --use_mask 不传, mask 仍能工作 (vLLM 端启用就生效), 只是失去 trainer 侧的
# 一致性检查 / log 区分.
TRAIN_USE_MASK_FLAG=""
if [ "$MASK_ENABLED" = "1" ]; then
    TRAIN_USE_MASK_FLAG="--use_mask --mask_n $MASK_N"
fi

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
    --reward_scheme v4 \
    $TRAIN_USE_MASK_FLAG \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EC=${PIPESTATUS[0]}

if [ $TRAIN_EC -eq 0 ]; then
    notify "✅ CVRP20 GRPO v4+Mask 训练完成" \
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
