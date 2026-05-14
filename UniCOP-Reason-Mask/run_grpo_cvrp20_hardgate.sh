#!/bin/bash
# run_grpo_cvrp20_hardgate.sh — GRPO + POMO PRM · CVRP n=20
#   7 卡 (1 vLLM + 6 训练) · cov_gate 硬墙 reward (v3)
#
# 跟 run_grpo_cvrp20_6gpu.sh 的区别:
#   - GPU 配置完全相同 (1 vLLM @ GPU6 + 6 训练 @ GPU0-5)
#   - 输出目录隔离: output_hardgate (不污染 output_6gpu 的 6gpu run)
#   - reward 公式改造 (代码侧已切换, 此脚本仅环境/日志/输出隔离):
#       v2 旧: A_feas = w_p*parse + w_cc*(cov × cons) + w_f*format  (3.5 满分)
#               cov 离散 hinge 1/0, cov*cons 乘积让 "保守少走" vs "全覆盖" 等价
#       v3 新: A_feas = w_p*parse + w_cov*cov + w_cons*cons*gate + w_f*format  (4.0 满分)
#               cov 连续 = n_unique/max(n, n_total) (同时惩罚遗漏+重复)
#               cov_gate=1.0 硬墙: cov<1.0 时 cons 信号置 0,
#               强迫模型先把 cov 推到全覆盖+无重复, 才解锁 cons 信号
#
# 为什么这么改 (诊断依据):
#   - 6gpu run 跑到 gstep 55 (11% 进度), feas rate 卡在 0.30 涨不动
#   - R_coverage_rate 0.45 → 0.49 几乎不动, R_constraint 0.81 → 0.91 快饱和
#   - 旧 reward 让 "cov=0.5 + cons=1.0" 和 "cov=1.0 + cons=0.5" 等高,
#     模型选择风险低的"保守少走"路径, 不愿冲 cov
#
# SBATCH 提交:
#   sbatch submit_grpo_cvrp20_hardgate.sh
# 手动:
#   bash run_grpo_cvrp20_hardgate.sh

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$(dirname "$_SELF_DIR")/paths.sh"

WORK_DIR="$MASK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/grpo_cvrp20_hardgate_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# zhuoyi 拓扑无 NVLink, NCCL P2P + SHM 必须禁用 (NVIDIA/nccl#1838, #2079)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# ── 配置 ──────────────────────────────────────────────────────────────
PROBLEM="cvrp"
SIZE=20

MODEL_BASE="$DISTILL_DIR/output_sft_hybrid_cvrp20/final_model"
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ 基座模型不存在: $MODEL_BASE"
    exit 1
fi
echo "[MODEL_BASE] $MODEL_BASE"

# GPU 分配: 7 卡 = 1 vLLM + 6 训练 (同 6gpu 脚本)
TOTAL_GPUS=7
VLLM_GPU=6
TRAIN_GPUS_CSV="0,1,2,3,4,5"
TRAIN_PROC=6

ZERO_STAGE=3
NUM_TRAIN=4000
# 输出隔离: hardgate run 独立目录, 不污染 6gpu run
OUTPUT_DIR_BASE="$WORK_DIR/output_hardgate"

# 端口 8001 错开 6gpu run 的 8000, 防同节点调度时撞端口
VLLM_PORT=8001
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=5120
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

# ── 启动/关闭 vLLM server ─────────────────────────────────────────
VLLM_LOG="$LOG_DIR/vllm_${PROBLEM}${SIZE}_hardgate_$(date +%Y%m%d_%H%M%S).log"
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
        notify "❌ CVRP20 GRPO Hardgate 异常退出" \
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
echo "  GRPO + POMO PRM · CVRP n=$SIZE · 7 卡 · cov_gate 硬墙 reward"
echo "  基座模型:  $MODEL_BASE"
echo "  GPU:       1 vLLM (GPU $VLLM_GPU) + $TRAIN_PROC 训练 (GPU $TRAIN_GPUS_CSV)"
echo "  ZeRO:      stage $ZERO_STAGE | gradient_checkpointing on"
echo "  Reward:    v3 (cov 连续化 + cov_gate=1.0 硬墙)"
echo "  输出目录:  $OUTPUT_DIR_BASE"
echo "  整除检查:  per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)),  整除 num_generations (8) ? $(( (4 * TRAIN_PROC) % 8 == 0 ))"
echo "  时间:      $(date)"
echo "============================================================"

if [ $((4 * TRAIN_PROC % 8)) -ne 0 ]; then
    echo "[FATAL] 整除失败: per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)) 必须整除 num_generations=8"
    exit 1
fi

notify "🚀 CVRP20 GRPO Hardgate 启动" \
"reward: cov_gate=1.0 硬墙 (v3)
基座: $MODEL_BASE
GPU: 1 vLLM + $TRAIN_PROC 训练
开始: $(date '+%Y-%m-%d %H:%M:%S')"

if ! start_vllm_server; then
    echo "[FATAL] vLLM server 启动失败"
    exit 1
fi

TRAIN_LOG="$LOG_DIR/train_${PROBLEM}${SIZE}_hardgate_$(date +%Y%m%d_%H%M%S).log"
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
    notify "✅ CVRP20 GRPO Hardgate 训练完成" \
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
