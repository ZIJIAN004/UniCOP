#!/bin/bash
# run_grpo_cvrp20_6gpu.sh — GRPO + POMO PRM 训练 · CVRP n=20 · 7 卡 (1 vLLM + 6 训练)
#
# 跟 run_grpo_cvrp20.sh 的区别:
#   - 训练从 4 卡升到 6 卡, 总卡数从 5 升到 7
#   - 保持 gradient_checkpointing 开 (A5000 24G + Qwen2.5-7B 关 GC 必 OOM, 激活 22+ GB)
#   - 关键整除关系: per_device_batch (4) × num_gpus (6) = 24, 整除 num_generations (8) → 3 groups/micro-step ✓
#     (5/7 卡训练会崩, 因为 4×N 必须整除 8 → N 必须偶数)
#
# 预期收益 (相对 4 卡):
#   - 反向传播理论 4/6 ≈ 67% 时间, 通信开销随 N 增长抵消一部分, 实际加速 25-30%
#   - 每卡省 ~1.4 GB 显存 (模型权重 + optimizer ZeRO-3 分片更细), 缓解 cache flushes WARN
#
# SBATCH 提交:
#   sbatch submit_grpo_cvrp20_6gpu.sh
# 手动:
#   bash run_grpo_cvrp20_6gpu.sh

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$(dirname "$_SELF_DIR")/paths.sh"

WORK_DIR="$MASK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/grpo_cvrp20_6gpu_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL 环境不一致 / handshake 失败时立刻打 WARN, 不刷屏.
export NCCL_DEBUG=WARN
# SIGABRT/SIGSEGV 时强制打印 Python 调用栈, 否则只看到 'Signal 6 received'.
export PYTHONFAULTHANDLER=1
# NCCL 2.21.5 在 PCIe-only / 跨 PCIe Host Bridge 拓扑 (NODE/PHB/SYS) 下,
# P2P 不可用 + cuMem allocator 有已知 bug (NVIDIA/nccl#1838, #2079),
# ZeRO-3 init 第一次 collective 会进入 deadlock spin-wait。
# 兜底方案: 禁用 P2P + 禁用 SHM 共享内存, 强制走 sockets。
# 代价: 通信慢 30-50% (ZeRO-3 影响大, GRPO 因 generation 主导影响小)。
# zhuoyi 拓扑无 NVLink, 实测同 NUMA NODE 拓扑下也会触发, 所以必须开。
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# ── 配置 ──────────────────────────────────────────────────────────────
PROBLEM="cvrp"
SIZE=20

# 基座模型: 用户指定的 SFT-hybrid-cvrp20 产物 (run_sft_hybrid_cvrp20.sh Step 2 已合并 LoRA)
MODEL_BASE="$DISTILL_DIR/output_sft_hybrid_cvrp20/final_model"
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ 基座模型不存在: $MODEL_BASE"
    exit 1
fi
echo "[MODEL_BASE] $MODEL_BASE"

# GPU 分配: 7 卡 = 1 vLLM + 6 训练.
# vLLM 放最后一张 (GPU 6), 跟训练隔离避免显存竞争.
# 训练 GPU 0-5 中可能跨 NUMA (5 卡日志里 GPU0-3 同 NUMA, GPU4 跨 NUMA).
# 新 7 卡分配的拓扑等运行时 nvidia-smi topo -m 才能确认. 若发现 GPU0-3 同
# NUMA 且 GPU4-5 跨 NUMA, 通信慢一点但 NCCL_P2P_DISABLE 已经在走 sockets,
# 影响不大.
TOTAL_GPUS=7
VLLM_GPU=6
TRAIN_GPUS_CSV="0,1,2,3,4,5"
TRAIN_PROC=6

ZERO_STAGE=3
# num_train 2000 → 4000: 总 update 数 250 → 500, 加 update 频率, GRPO 主流
# (DAPO/DeepSeek-R1) 推荐数千 update, 250 偏少. 代价: 总训练时间翻倍 ~160h.
NUM_TRAIN=4000
OUTPUT_DIR_BASE="$WORK_DIR/output_6gpu"

# vLLM server 参数 (与 auto_train.sh 对齐)
VLLM_PORT=8000
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=5120
VLLM_DTYPE=bfloat16
VLLM_STARTUP_TIMEOUT=300

# Server 酱
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

# ── 启动/关闭 vLLM server (摘自 auto_train.sh) ─────────────────────
VLLM_LOG="$LOG_DIR/vllm_${PROBLEM}${SIZE}_6gpu_$(date +%Y%m%d_%H%M%S).log"
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
        notify "❌ CVRP20 GRPO 6GPU 异常退出" \
"退出码: $exit_code
时间: $(date '+%Y-%m-%d %H:%M:%S')
日志末尾:
$(tail -n 20 "$LOG_FILE" 2>/dev/null || echo '(无日志)')"
    fi
}
trap 'on_exit' EXIT INT TERM

cd "$WORK_DIR"

echo "============================================================"
echo "  GPU 拓扑 (诊断 NCCL 通信路径)"
echo "============================================================"
nvidia-smi topo -m 2>&1 || echo "(nvidia-smi topo unavailable)"
echo ""
echo "============================================================"
echo "  GRPO + POMO PRM · CVRP n=$SIZE · 6卡训练"
echo "  基座模型:  $MODEL_BASE"
echo "  GPU:       1 vLLM (GPU $VLLM_GPU) + $TRAIN_PROC 训练 (GPU $TRAIN_GPUS_CSV)"
echo "  ZeRO:      stage $ZERO_STAGE | gradient_checkpointing on"
echo "  整除检查:  per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)),  整除 num_generations (8) ? $(( (4 * TRAIN_PROC) % 8 == 0 ))"
echo "  时间:      $(date)"
echo "============================================================"

# 启动前硬检查: 4 * TRAIN_PROC 必须整除 num_generations=8, 否则 BATCH_DIAG num_groups 错乱
if [ $((4 * TRAIN_PROC % 8)) -ne 0 ]; then
    echo "[FATAL] 整除失败: per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)) 必须整除 num_generations=8"
    echo "        TRAIN_PROC 必须是偶数 (2/4/6/8...)"
    exit 1
fi

notify "🚀 CVRP20 GRPO 6GPU 启动" \
"基座: $MODEL_BASE
GPU: 1 vLLM + $TRAIN_PROC 训练
开始: $(date '+%Y-%m-%d %H:%M:%S')"

# ── Step 1: 启动 vLLM server ─────────────────────────────────────
if ! start_vllm_server; then
    echo "[FATAL] vLLM server 启动失败"
    exit 1
fi

# ── Step 2: 训练 ─────────────────────────────────────────────────
TRAIN_LOG="$LOG_DIR/train_${PROBLEM}${SIZE}_6gpu_$(date +%Y%m%d_%H%M%S).log"
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
    --reward_scheme v3 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EC=${PIPESTATUS[0]}

if [ $TRAIN_EC -eq 0 ]; then
    notify "✅ CVRP20 GRPO 6GPU 训练完成" \
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
