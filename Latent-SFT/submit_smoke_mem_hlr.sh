#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_mem_hlr_%j.log

# ════════════════════════════════════════════════════════════════════
# HLR Smoke: ZeRO-3 vs ZeRO-2 显存/OOM  +  batched LR 提速 trial
# ════════════════════════════════════════════════════════════════════
# 背景 (8442 速度审计):
#   - "ZeRO-3→ZeRO-2" 是最大提速杠杆(消 per-layer 参数 all-gather), 但 ZeRO-2 不分片
#     参数 → 全 4B 基座每卡常驻, 可能 OOM. 本脚本实测每卡 peak/headroom + 是否 OOM.
#   - "batched LR forward" 是另一条中风险优化(43 段串行 LR → 1 次 batched). 先用
#     test_batched_lr.py 做 allclose 数值对拍守门, 通过才在真实训练里 trial 它的 latent_loop 提速.
#
# 做法: 同 profiled 数据 + 同 GC + 同 4 卡, 按 MEM_CONFIGS 逐个跑几个真实 optimizer-step.
#   每个 config = "stage:batched" (batched=1 启 HLR_BATCHED_LR). train.py 的 HLR_MEM_REPORT=1
#   每卡打 [MEM] (peak_reserved 最接近 OOM 天花板, headroom=total-peak_reserved 即剩余空间).
#   HLR_MAX_STEPS 跑到第 N 个 optimizer-step 就停 (不存 ckpt).
#
# 用法:
#   sbatch Latent-SFT/submit_smoke_mem_hlr.sh                                  # Qwen3-4B 默认
#   BASE_MODEL_TYPE=r1_distill sbatch Latent-SFT/submit_smoke_mem_hlr.sh       # R1-Distill-7B
#   MEM_CONFIGS="3:0 3:1" sbatch ...                                          # 只比 baseline vs batched (跳过 ZeRO-2)
#   MEM_MAX_STEPS=6 MEM_LIMIT=512 sbatch ...                                  # 跑更久/更多样本
#
# ⚠ MEM_LIMIT 必须能被 4(卡数) 整除, 否则各 rank batch 数不齐 → barrier 死锁.

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
# 缓解双 forward + output_hidden_states 高显存压力下的 allocator cache flush 抖动 (诊断 8442 反复出现).
# 纯 CUDA allocator 行为, 与正式训练同配置, 对照才干净.
# ⚠ 它会降低碎片化让 OOM 更不易发生; 本 smoke 测的就是"加了这个旋钮后 ZeRO-2 还会不会 OOM".
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
# 对照配置: "stage:batched" 列表. 3:0=ZeRO-3 基线 | 3:1=ZeRO-3+batched LR | 2:0=ZeRO-2(测 OOM)
MEM_CONFIGS="${MEM_CONFIGS:-3:0 3:1 2:0}"
OUTPUT_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_smoke_mem"
STAGE_LOG_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_mem_logs_${SLURM_JOB_ID:-local}"
mkdir -p "$STAGE_LOG_DIR"

echo "============================================================"
echo "  HLR Smoke: ZeRO-3 vs ZeRO-2 显存/OOM + batched LR trial"
echo "============================================================"
echo "  HOST            = $HOST_ID"
echo "  BASE_MODEL_TYPE = $BASE_MODEL_TYPE"
echo "  MODEL           = $MODEL_PATH"
echo "  PROFILED DATA   = $PROFILED_DATA (profile_limit=$PROFILE_LIMIT)"
echo "  MEM_CONFIGS     = $MEM_CONFIGS   max_steps=$MEM_MAX_STEPS  limit=$MEM_LIMIT"
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

# ── Step 1.5: batched LR 数值等价 allclose 对拍 (correctness gate, CPU, ~秒级) ──
echo ""
echo ">>> Step 1.5: batched LR allclose 对拍 (correctness gate) ($(date))"
_batched_ok=0
if python Latent-SFT/test_batched_lr.py; then
    echo "    ✓ allclose 通过 → batched 配置(:1)保留, 可在训练里 trial"
    _batched_ok=1
else
    echo "    ❌ allclose 失败! batched LR 逻辑有 bug, 自动从 MEM_CONFIGS 剔除所有 :1 配置 (只跑 baseline)"
fi

# allclose 没过 → 过滤掉所有 batched(:1) 配置, 仍保留 ZeRO-3/2 显存对照
_RUN_CONFIGS=""
for cfg in $MEM_CONFIGS; do
    _b="${cfg#*:}"
    if [ "$_b" = "1" ] && [ "$_batched_ok" != "1" ]; then
        echo "    (跳过 $cfg: allclose 未通过)"
        continue
    fi
    _RUN_CONFIGS="$_RUN_CONFIGS $cfg"
done

# ════════════════════════════════════════════════════════════════════
# run_config <stage> <batched> <port>: 跑一遍, 回显 [MEM]/[TIMING], 判 OK/OOM/FAIL
# 返回: 0=跑通  2=OOM  1=其它失败
# ════════════════════════════════════════════════════════════════════
run_config() {
    local stage=$1
    local batched=$2
    local port=$3
    local label
    if [ "$batched" = "1" ]; then label="zero${stage}_batched"; else label="zero${stage}_base"; fi
    local slog="$STAGE_LOG_DIR/train_${label}.log"

    echo ""
    echo "############################################################"
    echo "###  $label  (ZeRO-$stage, batched_LR=$batched)  ($(date))"
    echo "############################################################"
    rm -rf "$OUTPUT_DIR"   # 隔离每次 run

    HLR_MEM_REPORT=1 HLR_TIMING=1 HLR_MAX_STEPS="$MEM_MAX_STEPS" HLR_BATCHED_LR="$batched" \
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
    echo "  ── $label 每卡显存 (peak_reserved≈OOM 天花板, headroom 越小越危险) ──"
    grep "\[MEM " "$slog" | sed 's/^/    /' || echo "    (无 [MEM] 行: 多半在 init/训练中就 OOM 了)"
    echo "  ── $label rank0 分段计时 (看 latent_loop: batched 应比 base 小) ──"
    grep "\[TIMING r0 " "$slog" | tail -1 | sed 's/^/    /' || echo "    (无 [TIMING])"

    if [ $code -ne 0 ]; then
        if grep -qiE "out of memory|outofmemory|cuda error: out of memory|cublas_status_alloc_failed" "$slog"; then
            echo "  ❌ $label: OOM (exit $code)"; return 2
        else
            echo "  ❌ $label: 失败但非 OOM (exit $code) — 看上面 traceback"; return 1
        fi
    fi
    echo "  ✓ $label: 跑通 $MEM_MAX_STEPS 步, 未 OOM"; return 0
}

# ── 依次跑各 config ──
declare -A CFG_RESULT
_port=29850
for cfg in $_RUN_CONFIGS; do
    _stage="${cfg%:*}"; _batched="${cfg#*:}"
    run_config "$_stage" "$_batched" "$_port"
    CFG_RESULT["$cfg"]=$?
    _port=$(( _port + 1 ))
done

# ════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  汇总 (job ${SLURM_JOB_ID:-local})"
echo "============================================================"
for cfg in $_RUN_CONFIGS; do
    _stage="${cfg%:*}"; _batched="${cfg#*:}"
    if [ "$_batched" = "1" ]; then _label="zero${_stage}_batched"; else _label="zero${_stage}_base"; fi
    case "${CFG_RESULT[$cfg]}" in
        0) _verdict="✓ 跑通未 OOM" ;;
        2) _verdict="❌ OOM" ;;
        *) _verdict="❌ 失败(非OOM)" ;;
    esac
    echo "  $cfg ($_label) : $_verdict"
    grep "\[MEM " "$STAGE_LOG_DIR/train_${_label}.log" 2>/dev/null \
        | grep "rank=0" | tail -1 | sed 's/^/      rank0 MEM: /'
    grep "\[TIMING r0 " "$STAGE_LOG_DIR/train_${_label}.log" 2>/dev/null \
        | tail -1 | grep -oE "latent_loop=[ ]*[0-9]+ms[^ ]*" | sed 's/^/      latent_loop: /'
done
echo ""
echo "  结论参考:"
echo "    [ZeRO-2 OOM 判定]   2:0 跑通且 headroom 有富余 → 可上 ZeRO-2 (最大杠杆); OOM/贴 0 → 留 ZeRO-3"
echo "    [ZeRO-3 剩余空间]   看 3:0 的 headroom = total - peak_reserved"
echo "    [基座常驻代价]      ZeRO-2 与 ZeRO-3 的 peak_reserved 差值 ≈ 全 4B 基座未分片代价"
echo "    [batched LR 提速]   比 3:1 vs 3:0 的 latent_loop (应下降) 与整步 ~s/micro"
echo "  详细每卡日志: $STAGE_LOG_DIR/train_*.log"
echo "============================================================"

# 退出码: 只要 baseline ZeRO-3 (3:0) 跑通就算 smoke 成功 (ZeRO-2 OOM 是预期可能结果, 不算失败)
if [ "${CFG_RESULT[3:0]:-x}" = "0" ] || [ "${CFG_RESULT[3:0]:-x}" = "x" ]; then
    exit 0
else
    exit 1
fi
