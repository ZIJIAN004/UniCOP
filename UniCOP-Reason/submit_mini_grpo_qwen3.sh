#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=2
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/mini_grpo_qwen3_%j.log

# Mini GRPO 真训练 (Qwen3-4B-Thinking · CVRP-10 · 5 step)
# 2 GPU: 1 vLLM + 1 train, 短 prompt 短 completion, ~15 分钟出结果
# 目的: 验证我们改的 GRPO 兼容性代码 (config / GRPOConfig / decode / _strip)
#       在端到端真训练流程上 forward+backward 健康, 第一个 step 不崩.

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
export BASE_MODEL_TYPE=qwen3_thinking
source "$(dirname "$_SELF_DIR")/paths.sh"

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# zhuoyi NCCL topology
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

WORK_DIR="$REASON_DIR"
cd "$WORK_DIR"

# 基座: SFT-Qwen3 合并产物 (训练好的 LoRA 已 merge)
MODEL_BASE="$DISTILL_DIR/output_sft_qwen3_template_cvrp20/final_model"
if [ ! -d "$MODEL_BASE" ]; then
    echo "⚠️  $MODEL_BASE 不存在, 退化用 raw base model 跑 GRPO (用于纯兼容性测试, reward 信号会差)"
    MODEL_BASE="$BASE_MODEL"
fi
echo "[MODEL_BASE] $MODEL_BASE"

OUTPUT_DIR_BASE="$WORK_DIR/output_mini_smoke"
mkdir -p "$OUTPUT_DIR_BASE"

VLLM_PORT=8800
VLLM_LOG="$OUTPUT_DIR_BASE/vllm_smoke_${SLURM_JOB_ID:-local}.log"
TRAIN_LOG="$OUTPUT_DIR_BASE/train_smoke_${SLURM_JOB_ID:-local}.log"

echo "============================================================"
echo "  Mini GRPO smoke (Qwen3-4B-Thinking)"
echo "  BASE_MODEL_TYPE: $BASE_MODEL_TYPE  T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K"
echo "  vLLM GPU=0  port=$VLLM_PORT  log=$VLLM_LOG"
echo "  train GPU=1  log=$TRAIN_LOG"
echo "  目的: 验证 GRPO 兼容 Qwen3 真训练流程 (forward+backward 健康)"
echo "============================================================"

# ── GPU 占用诊断 ──────────────────────────────────────────────────────
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
_min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -n "$_min_free" ] && [ "$_min_free" -lt 20000 ]; then
    echo "❌ 某卡 free<20GiB 别人占着, abort. 重提让 SLURM 换节点"
    exit 1
fi

# ── Step 1: 启动 vLLM server (GPU 0) ──────────────────────────────────
echo ">>> 启动 vLLM server..."
PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES=0 \
CUDA_HOME="$CUDA_HOME" \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
    python "$WORK_DIR/utils/vllm_serve_ngram.py" \
    --no_repeat_ngram_size 0 \
    --model "$MODEL_BASE" \
    --tensor_parallel_size 1 \
    --port "$VLLM_PORT" \
    --gpu_memory_utilization 0.85 \
    --max_model_len 8192 \
    --dtype bfloat16 \
    --enable_prefix_caching True \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# 等 vLLM 就绪 (最多 5 分钟)
echo "等 vLLM 就绪 (pid=$VLLM_PID)..."
for i in $(seq 1 100); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ vLLM 启动失败, log 末尾:"; tail -30 "$VLLM_LOG"; exit 1
    fi
    if curl -s "http://localhost:${VLLM_PORT}/health/" > /dev/null 2>&1; then
        echo "✓ vLLM 就绪 (用时 ${i}×3 = $((i*3))s)"
        break
    fi
    sleep 3
done

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "stopping vLLM (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Step 2: train.py 跑 5 step (GPU 1) ────────────────────────────────
# 限制非常激进, 保证 15 分钟内收敛到 5 step:
#   num_train=8 (8 prompts), num_generations=2 → 4 GRPO group
#   max_completion_length=1024 (短 think)
#   per_device_train_batch_size=1, grad_accum=2 → effective batch 2
#   max_steps 通过 hijack: num_train=8 → 1 epoch ≈ 4 step, 3 epoch ≈ 12 step
# 看前几个 log: loss 不 NaN + grad_norm 正常 + reward 不报错 = pass
echo ""
echo ">>> 启动训练 (GPU 1, 1 process)..."
PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
CUDA_HOME="$CUDA_HOME" \
CUDA_VISIBLE_DEVICES=1 \
    python -m accelerate.commands.launch --num_processes 1 \
        --main_process_port 29800 \
        "$WORK_DIR/train.py" \
    --problem cvrp \
    --problem_size 10 \
    --num_train 8 \
    --model "$MODEL_BASE" \
    --num_gpus 1 \
    --zero_stage 0 \
    --gradient_checkpointing \
    --no_repeat_ngram_size 0 \
    --output_dir "$OUTPUT_DIR_BASE" \
    --pomo_ckpt_dir "$POMO_CKPT_DIR" \
    --pomo_baseline_dir "$POMO_BASELINE_DIR" \
    --pipd_ckpt_dir "$PIPD_CKPT_DIR" \
    --pipd_dir "$PIPD_DIR" \
    --vllm_server_host "localhost" \
    --vllm_server_port "$VLLM_PORT" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EC=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "  Mini GRPO 完成! exit=$TRAIN_EC"
echo "  train log: $TRAIN_LOG"
echo "  vLLM log:  $VLLM_LOG"
echo ""
echo "  检查 train log 前几条 step:"
echo "    grep -E 'loss|grad_norm|reward|FAIL|ERROR|Traceback' $TRAIN_LOG | head -30"
echo "============================================================"

exit $TRAIN_EC
