#!/bin/bash
# run_grpo_cvrp20_mask.sh — GRPO + POMO PRM · CVRP n=20
#   7 卡 (1 vLLM + 6 训练) · CVRP constrained-decoding mask
#
# 跟 run_grpo_cvrp20_hardgate.sh 的区别:
#   - vLLM server 启用 mask processor (--mask_enabled --mask_n 20)
#   - 输出隔离到 output_mask (跟 hardgate 的 output_hardgate 独立)
#   - vLLM 端口 8002 (hardgate 用 8001, 6gpu 用 8000)
#   - 训练端会拿 mask_hits 跳过 mask 位置的 IS 校正 (方案 C)
#
# Reward 公式仍是 v3 hardgate (cov_gate=1.0 硬墙), 但:
#   - mask 让 cov 必然 = 1 (走完所有 customer 才能 EOS)
#   - cov_gate=1.0 自动满足, cons_signal 总是开门
#   - 所以 reward 实际只看 cons + ORM + PRM
#
# SBATCH 提交:
#   sbatch submit_grpo_cvrp20_mask.sh

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
# BASE_MODEL_TYPE: r1_distill (默认, 7B DeepSeek) 或 qwen3_thinking (4B Qwen3-Thinking)
# paths.sh 据此设 BASE_MODEL + 采样参数 GEN_TEMPERATURE/TOP_P/TOP_K, trainer 自动读 env
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-r1_distill}"
source "$(dirname "$_SELF_DIR")/paths.sh"

WORK_DIR="$MASK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/grpo_cvrp20_mask_$(date '+%Y%m%d_%H%M%S').log"
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
echo "[MODEL_BASE] $MODEL_BASE"

# GPU 分配: 7 卡 = 1 vLLM + 6 训练 (同 hardgate)
TOTAL_GPUS=7
VLLM_GPU=6
TRAIN_GPUS_CSV="0,1,2,3,4,5"
TRAIN_PROC=6

ZERO_STAGE=3
NUM_TRAIN=4000
OUTPUT_DIR_BASE="$WORK_DIR/output_mask"   # 跟 hardgate 隔离

# 端口 8002 错开 hardgate (8001) 和 6gpu (8000)
VLLM_PORT=8002
VLLM_GPU_MEM_UTIL=0.85
VLLM_MAX_MODEL_LEN=8192
VLLM_DTYPE=bfloat16
VLLM_STARTUP_TIMEOUT=300

# 单卡已用显存超过此值(MiB)即视为"非空闲",拒绝启动。
# 空闲卡通常只占几 MiB~几百 MiB；任何残留模型进程都是 GB 级,2000 足够区分。
GPU_MEM_USED_THRESHOLD_MIB="${GPU_MEM_USED_THRESHOLD_MIB:-2000}"

# ── Mask 超参 ─────────────────────────────────────────────────────────
MASK_ENABLED=1   # 1=启用, 0=不启用 (退化为 hardgate)
MASK_N=20        # CVRP customer 数
MASK_DEBUG=0    # 1=详细 stderr log

SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

# ── 启动/关闭 vLLM server ─────────────────────────────────────────
VLLM_LOG="$LOG_DIR/vllm_${PROBLEM}${SIZE}_mask_$(date +%Y%m%d_%H%M%S).log"
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

# ── GPU 空闲检查 (防止往被占用的卡上启 vLLM 导致 CUDA graph capture OOM) ──
# 根因: vLLM gpu_memory_utilization 是"按总显存×util"的全局上限,不看当前空闲量。
#       若目标卡已被别的进程(别人的 job / 上次 crash 残留)占用,vLLM 仍按 util×总显存
#       预算 KV cache,等到 CUDA graph capture 要额外 scratch 时物理空闲不够 → capture_end OOM。
# 在启 vLLM 前把"vLLM 卡 + 训练卡"全查一遍,非空闲直接 fail-fast。
check_gpu_idle() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] 找不到 nvidia-smi,跳过 GPU 空闲检查"
        return 0
    fi

    local smi_out
    smi_out="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -z "$smi_out" ]; then
        echo "[WARN] nvidia-smi 查询为空,跳过 GPU 空闲检查"
        return 0
    fi

    local gpus_to_check
    gpus_to_check="$(echo "${VLLM_GPU},${TRAIN_GPUS_CSV}" | tr ',' ' ')"
    echo "[$(date '+%H:%M:%S')] GPU 空闲检查 (阈值 ${GPU_MEM_USED_THRESHOLD_MIB} MiB) | 待用: GPU ${VLLM_GPU}(vLLM) + ${TRAIN_GPUS_CSV}(训练)"

    local busy=0 g used total
    for g in $gpus_to_check; do
        used="$(echo "$smi_out"  | awk -F',' -v idx="$g" '{gsub(/ /,"",$1);gsub(/ /,"",$2)} $1==idx {print $2}')"
        total="$(echo "$smi_out" | awk -F',' -v idx="$g" '{gsub(/ /,"",$1);gsub(/ /,"",$3)} $1==idx {print $3}')"
        if [ -z "$used" ]; then
            echo "  [FAIL] GPU $g 在 nvidia-smi 中不存在 (可见 GPU 数不足?)"
            busy=$((busy+1))
        elif [ "$used" -gt "$GPU_MEM_USED_THRESHOLD_MIB" ]; then
            echo "  [FAIL] GPU $g 非空闲: 已用 ${used} / ${total} MiB"
            busy=$((busy+1))
        else
            echo "  [OK  ] GPU $g 空闲: 已用 ${used} / ${total} MiB"
        fi
    done

    if [ "$busy" -ne 0 ]; then
        echo ""
        echo "[FATAL] $busy 张待用 GPU 非空闲,拒绝启动 (否则 vLLM 大概率在 CUDA graph capture 阶段 OOM)"
        echo "  当前 GPU 占用进程:"
        nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv 2>/dev/null | sed 's/^/    /' || true
        echo "  排查:"
        echo "    1) 上次 crash 残留僵尸进程? 清理: pkill -u \"\$USER\" -f vllm_serve_logprobs.py ; pkill -u \"\$USER\" -f accelerate.commands.launch"
        echo "    2) 手动跑落到别人占用的物理卡? 改用 sbatch submit_grpo_cvrp20_mask.sh,或指定空闲卡:"
        echo "       VLLM_GPU=<空闲> TRAIN_GPUS_CSV=<空闲列表> TRAIN_PROC=<卡数> bash $0"
        echo "    3) 确认是误判? 临时放宽: GPU_MEM_USED_THRESHOLD_MIB=<更大值> bash $0"
        notify "❌ CVRP20 GRPO Mask GPU 非空闲,未启动" \
"有 $busy 张待用 GPU 已被占用 (阈值 ${GPU_MEM_USED_THRESHOLD_MIB} MiB)
时间: $(date '+%Y-%m-%d %H:%M:%S')
详见日志: $LOG_FILE"
        return 1
    fi
    echo "[$(date '+%H:%M:%S')] ✓ 所有待用 GPU 空闲"
    return 0
}

TRAINING_COMPLETED=0
on_exit() {
    local exit_code=$?
    stop_vllm_server
    if [ "$TRAINING_COMPLETED" != "1" ] && [ "$exit_code" != "0" ]; then
        notify "❌ CVRP20 GRPO Mask 异常退出" \
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
echo "  GRPO + POMO PRM · CVRP n=$SIZE · 7 卡 · CVRP Mask"
echo "  BASE_MODEL_TYPE: $BASE_MODEL_TYPE  (T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K)"
echo "  基座模型:  $MODEL_BASE"
echo "  GPU:       1 vLLM (GPU $VLLM_GPU) + $TRAIN_PROC 训练 (GPU $TRAIN_GPUS_CSV)"
echo "  ZeRO:      stage $ZERO_STAGE | gradient_checkpointing on"
echo "  Reward:    hardgate v3 (cov_gate=1.0) + mask 强制 cov=1"
echo "  Mask:      $([ "$MASK_ENABLED" = "1" ] && echo "ENABLED (n=$MASK_N, debug=$MASK_DEBUG)" || echo "DISABLED (退化为 hardgate)")"
echo "  输出目录:  $OUTPUT_DIR_BASE"
echo "  整除检查:  per_device_batch (4) × num_gpus ($TRAIN_PROC) = $((4 * TRAIN_PROC)),  整除 num_generations (8) ? $(( (4 * TRAIN_PROC) % 8 == 0 ))"
echo "  时间:      $(date)"
echo "============================================================"

if [ $((4 * TRAIN_PROC % 8)) -ne 0 ]; then
    echo "[FATAL] 整除失败"
    exit 1
fi

if ! check_gpu_idle; then
    echo "[FATAL] GPU 空闲检查未通过,中止 (避免 vLLM CUDA graph capture OOM)"
    exit 1
fi

notify "🚀 CVRP20 GRPO Mask 启动" \
"mask: enabled=$MASK_ENABLED, n=$MASK_N
基座: $MODEL_BASE
GPU: 1 vLLM + $TRAIN_PROC 训练
开始: $(date '+%Y-%m-%d %H:%M:%S')"

if ! start_vllm_server; then
    echo "[FATAL] vLLM server 启动失败"
    exit 1
fi

TRAIN_LOG="$LOG_DIR/train_${PROBLEM}${SIZE}_mask_$(date +%Y%m%d_%H%M%S).log"
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
    notify "✅ CVRP20 GRPO Mask 训练完成" \
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
