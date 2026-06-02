#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_mem_hlr_%j.log

# ════════════════════════════════════════════════════════════════════
# HLR 显存对照 Smoke: ZeRO-3 vs ZeRO-2 每卡峰值/剩余空间 + ZeRO-2 是否 OOM
# ════════════════════════════════════════════════════════════════════
# 背景: 8442 速度审计里, "ZeRO-3→ZeRO-2" 是核验后最大的提速杠杆(消除 per-layer
#       参数 all-gather), 但 ZeRO-2 不分片参数 → 全 4B 基座每卡常驻, 可能 OOM.
#       本脚本各跑几个真实训练 optimizer-step, 打印每卡 peak_reserved / headroom,
#       并捕获 ZeRO-2 是否 OOM, 用数据决定能否上 ZeRO-2 (而非纸面拍板).
#
# 做法: 同一 profiled 数据 + 同样 GC + 同样 4 卡, 只换 --zero_stage 3 / 2 各跑一遍.
#       train.py 的 HLR_MEM_REPORT=1 在每卡打 [MEM] 行 (peak 含 init/load 瞬时占用).
#       HLR_MAX_STEPS 跑到第 N 个 optimizer-step 就停 (不保存 ckpt).
#
# 用法:
#   sbatch Latent-SFT/submit_smoke_mem_hlr.sh                               # Qwen3-4B 默认
#   BASE_MODEL_TYPE=r1_distill sbatch Latent-SFT/submit_smoke_mem_hlr.sh    # R1-Distill-7B
#   MEM_MAX_STEPS=6 MEM_LIMIT=512 sbatch Latent-SFT/submit_smoke_mem_hlr.sh # 跑更久/更多样本
#   MEM_STAGES="3" sbatch ...                                              # 只测 ZeRO-3
#
# ⚠ MEM_LIMIT 必须能被 4(卡数) 整除, 否则各 rank batch 数不齐 → barrier 死锁(踩坑).
#   默认 256 (÷4=64/卡), peak 由 batch 内最长样本决定, 不是样本数, 故 256 足够触顶.

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
# 缓解双 forward + output_hidden_states 高显存压力下的 allocator cache flush 抖动 (诊断 8442 反复出现).
# 纯 CUDA allocator 行为, 不改 collective/loss/B=1 假设. 与正式训练同配置, 对照才干净.
# ⚠ 这会降低碎片化, 让 OOM 更不易发生; 本 smoke 测的就是"加了这个旋钮后 ZeRO-2 还会不会 OOM".
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# zhuoyi 多卡 NCCL 拓扑必加 (无 NVLink, 走 socket transport, 踩坑 #29)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export DEEPSPEED_TIMEOUT=3600

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source paths.sh

MODEL_PATH="${HLR_MODEL:-$BASE_MODEL}"
RAW_DATA="UniCOP-Distill/data/chains_template_cvrp20.jsonl"
PROFILED_DATA="Latent-SFT/data/profiled_smoke_${BASE_MODEL_TYPE}_cvrp20.jsonl"
PROFILE_LIMIT="${SMOKE_PROFILE_LIMIT:-256}"   # ≥ MEM_LIMIT, 且 ÷4 整除
MEM_LIMIT="${MEM_LIMIT:-256}"                  # 训练用样本数 (÷4 整除!)
MEM_MAX_STEPS="${MEM_MAX_STEPS:-4}"           # 跑到第几个 optimizer-step 停
MEM_STAGES="${MEM_STAGES:-3 2}"               # 要对照的 stage (默认先 3 后 2)
OUTPUT_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_smoke_mem"
STAGE_LOG_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_mem_logs_${SLURM_JOB_ID:-local}"
mkdir -p "$STAGE_LOG_DIR"

echo "============================================================"
echo "  HLR 显存对照 Smoke: ZeRO-3 vs ZeRO-2"
echo "============================================================"
echo "  HOST            = $HOST_ID"
echo "  BASE_MODEL_TYPE = $BASE_MODEL_TYPE"
echo "  MODEL           = $MODEL_PATH"
echo "  PROFILED DATA   = $PROFILED_DATA (profile_limit=$PROFILE_LIMIT)"
echo "  对照 stages     = $MEM_STAGES   max_steps=$MEM_MAX_STEPS  limit=$MEM_LIMIT"
echo "  expandable_segments = ON (与正式训练一致)"
echo "============================================================"

# ── MEM_LIMIT ÷4 整除校验 (不齐会 barrier 死锁) ──
if [ $(( MEM_LIMIT % 4 )) -ne 0 ]; then
    echo "❌ MEM_LIMIT=$MEM_LIMIT 不能被 4 整除, 各 rank batch 数不齐会触发 barrier 死锁. 退出."
    exit 1
fi

# ── 脏节点自检 (显存被别人占会污染峰值/误判 OOM) ──
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader | sed 's/^/    /'
_min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -n "$_min_free" ] && [ "$_min_free" -lt 20000 ]; then
    echo "❌ 某卡 free ${_min_free} MiB < 20 GiB, 节点脏 (峰值/OOM 判读会失真), 自动 resubmit exclude"
    _bad_node="$SLURM_NODELIST"; _self_path="$(realpath "$0")"
    _new_bad="${BAD_NODES:+$BAD_NODES,}$_bad_node"
    sbatch --exclude="$_new_bad" --export=ALL,BAD_NODES="$_new_bad" "$_self_path"
    exit 0
fi
echo "============================================================"

# ── Step 1: profiled smoke 数据 (不存在则单卡 entropy_profile) ──
echo ""
echo ">>> Step 1: profiled smoke 数据 ($(date))"
if [ -f "$PROFILED_DATA" ] && [ "${SMOKE_FORCE_REPROFILE:-0}" != "1" ]; then
    echo "    ✓ 已存在: $PROFILED_DATA ($(wc -l < "$PROFILED_DATA") 行)"
else
    [ -f "$RAW_DATA" ] || { echo "❌ 原始 chains 不存在: $RAW_DATA"; exit 1; }
    mkdir -p "$(dirname "$PROFILED_DATA")"
    CUDA_VISIBLE_DEVICES=0 python Latent-SFT/entropy_profile.py \
        --model "$MODEL_PATH" --data "$RAW_DATA" \
        --output "$PROFILED_DATA" --limit "$PROFILE_LIMIT"
    [ $? -eq 0 ] && [ -f "$PROFILED_DATA" ] || { echo "❌ entropy_profile 失败"; exit 1; }
    echo "    ✓ 生成: $(wc -l < "$PROFILED_DATA") 行"
fi

# ════════════════════════════════════════════════════════════════════
# run_stage <zero_stage> <port>: 跑一遍, 回显 [MEM], 判 OK / OOM / FAIL
# 返回: 0=跑通  2=OOM  1=其它失败
# ════════════════════════════════════════════════════════════════════
run_stage() {
    local stage=$1
    local port=$2
    local slog="$STAGE_LOG_DIR/train_zero${stage}.log"

    echo ""
    echo "############################################################"
    echo "###  ZeRO-$stage  ($(date))"
    echo "############################################################"
    rm -rf "$OUTPUT_DIR"   # 隔离两次 run, 不让上一次残留干扰

    HLR_MEM_REPORT=1 HLR_TIMING=1 HLR_MAX_STEPS="$MEM_MAX_STEPS" \
    accelerate launch --num_processes 4 --main_process_port "$port" \
        Latent-SFT/train.py \
        --model "$MODEL_PATH" \
        --data "$PROFILED_DATA" \
        --zero_stage "$stage" \
        --gradient_checkpointing \
        --output_dir "$OUTPUT_DIR" \
        --epochs 1 \
        --logging_steps 2 --save_steps 1000000 \
        --limit "$MEM_LIMIT" \
        2>&1 | tee "$slog"
    local code=${PIPESTATUS[0]}

    echo ""
    echo "  ── ZeRO-$stage 每卡显存 (peak_reserved 最接近 OOM 天花板, headroom 越小越危险) ──"
    grep "\[MEM " "$slog" | sed 's/^/    /' || echo "    (无 [MEM] 行: 多半在 init/训练中就 OOM 了)"

    if [ $code -ne 0 ]; then
        if grep -qiE "out of memory|outofmemory|cuda error: out of memory|cublas_status_alloc_failed" "$slog"; then
            echo "  ❌ ZeRO-$stage: OOM (exit $code)"
            return 2
        else
            echo "  ❌ ZeRO-$stage: 失败但非 OOM (exit $code) — 看上面 traceback"
            return 1
        fi
    fi
    echo "  ✓ ZeRO-$stage: 跑通 $MEM_MAX_STEPS 步, 未 OOM"
    return 0
}

# ── 依次跑各 stage ──
declare -A STAGE_RESULT
_port=29850
for stage in $MEM_STAGES; do
    run_stage "$stage" "$_port"
    STAGE_RESULT[$stage]=$?
    _port=$(( _port + 1 ))
done

# ════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  汇总 (job ${SLURM_JOB_ID:-local})"
echo "============================================================"
for stage in $MEM_STAGES; do
    case "${STAGE_RESULT[$stage]}" in
        0) _verdict="✓ 跑通未 OOM" ;;
        2) _verdict="❌ OOM" ;;
        *) _verdict="❌ 失败(非OOM)" ;;
    esac
    echo "  ZeRO-$stage : $_verdict"
    # 抽 rank0 的 peak_reserved / headroom 给个一眼结论
    grep "\[MEM " "$STAGE_LOG_DIR/train_zero${stage}.log" 2>/dev/null \
        | grep "rank=0" | tail -1 | sed 's/^/      rank0: /'
done
echo ""
echo "  结论参考:"
echo "    - ZeRO-2 跑通且 headroom 仍有富余 → 可上 ZeRO-2 (最大提速杠杆, 见 8442 审计)"
echo "    - ZeRO-2 OOM 或 headroom 接近 0   → 留在 ZeRO-3; 或先压 max_length / 优化 output_hidden_states 再试"
echo "    - 对比两者 peak_reserved 差值 ≈ 全 4B 基座未分片的常驻代价"
echo "  详细每卡 [MEM]: $STAGE_LOG_DIR/train_zero*.log"
echo "============================================================"

# 退出码: 只要 ZeRO-3 跑通就算 smoke 成功 (ZeRO-2 OOM 是预期可能结果, 不算脚本失败)
if [ "${STAGE_RESULT[3]:-1}" = "0" ] || [ "${STAGE_RESULT[3]:-x}" = "x" ]; then
    exit 0
else
    exit 1
fi
