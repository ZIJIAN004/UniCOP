#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_hlr_%j.log

# HLR 全流程 Smoke Test (4 GPU + ZeRO-3 + GC, 镜像 submit_train_hlr.sh 配置)
#
# 自动执行:
#   Step 1: profiled smoke jsonl 不存在 → 跑 entropy_profile --limit 200
#           (单 GPU bf16, 约 5-10 min)
#   Step 2: accelerate launch 4 GPU 跑 smoke_test_hlr.py 7 stage
#           stage 5 真走 ZeRO-3 切片 + gradient_checkpointing, 验证训练流程
#
# 默认配置 (env 覆盖):
#   BASE_MODEL_TYPE       = r1_distill | qwen3_thinking (默认 r1_distill)
#   HLR_MODEL             = 主模型路径 (默认 $BASE_MODEL)
#   SMOKE_PROFILE_LIMIT   = entropy_profile 处理样本数 (默认 200)
#   SMOKE_FORCE_REPROFILE = 1 → 强制重跑 entropy_profile
#
# 用法:
#   sbatch Latent-SFT/submit_smoke_hlr.sh                                # R1-Distill 默认
#   BASE_MODEL_TYPE=qwen3_thinking sbatch Latent-SFT/submit_smoke_hlr.sh # Qwen3-4B-Thinking
#
# 4 GPU normal QOS, 总耗时 ~25-35 min

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
# 缓解双 forward + output_hidden_states 高显存压力下的 allocator cache flush 抖动 (诊断 8442 反复出现).
# 纯 CUDA allocator 行为, 不改 collective/loss/B=1 假设. smoke 正好用来确认 PyTorch/NCCL 不报错.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# zhuoyi 多卡 NCCL 拓扑必加 (无 NVLink, 走 socket transport, 踩坑 #29)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
source paths.sh

MODEL_PATH="${HLR_MODEL:-$BASE_MODEL}"
RAW_DATA="${RAW_DATA:-UniCOP-Distill/data/chains_template_cvrp20.jsonl}"
PROFILED_DATA="Latent-SFT/data/profiled_smoke_${BASE_MODEL_TYPE}_cvrp20.jsonl"
PROFILE_LIMIT="${SMOKE_PROFILE_LIMIT:-200}"
FORCE_REPROFILE="${SMOKE_FORCE_REPROFILE:-0}"

echo "============================================================"
echo "  HLR 全流程 Smoke Test (4 GPU + ZeRO-3 + GC)"
echo "============================================================"
echo "  HOST              = $HOST_ID"
echo "  BASE_MODEL_TYPE   = $BASE_MODEL_TYPE"
echo "  MODEL             = $MODEL_PATH"
echo "  RAW DATA          = $RAW_DATA"
echo "  PROFILED OUTPUT   = $PROFILED_DATA"
echo "  PROFILE_LIMIT     = $PROFILE_LIMIT"
echo "============================================================"

nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader \
    | sed 's/^/    /'
_min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -n "$_min_free" ] && [ "$_min_free" -lt 20000 ]; then
    echo "❌ free memory ${_min_free} MiB < 20 GiB, 节点脏, 自动 resubmit exclude"
    _bad_node="$SLURM_NODELIST"
    _self_path="$(realpath "$0")"
    if [ -n "${BAD_NODES:-}" ]; then
        _new_bad="$BAD_NODES,$_bad_node"
    else
        _new_bad="$_bad_node"
    fi
    sbatch --exclude="$_new_bad" --export=ALL,BAD_NODES="$_new_bad" "$_self_path"
    exit 0
fi
echo "============================================================"

# ═════════════════════════════════════════════════════════════════
# Step 1: 单 GPU 跑 entropy_profile (如不存在)
# ═════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: entropy_profile ($(date))"
if [ -f "$PROFILED_DATA" ] && [ "$FORCE_REPROFILE" != "1" ]; then
    _existing=$(wc -l < "$PROFILED_DATA")
    echo "✓ profiled jsonl 已存在: $PROFILED_DATA ($_existing 行)"
else
    if [ ! -f "$RAW_DATA" ]; then
        echo "❌ 原始 chains 不存在: $RAW_DATA"
        exit 1
    fi
    # 单 GPU 跑 profile (CUDA_VISIBLE_DEVICES=0 限制只用一张卡)
    CUDA_VISIBLE_DEVICES=0 python Latent-SFT/entropy_profile.py \
        --model "$MODEL_PATH" \
        --data "$RAW_DATA" \
        --output "$PROFILED_DATA" \
        --limit "$PROFILE_LIMIT"
    _ep_exit=$?
    if [ $_ep_exit -ne 0 ] || [ ! -f "$PROFILED_DATA" ]; then
        echo "❌ entropy_profile 失败 (exit $_ep_exit)"
        exit 1
    fi
fi

# ═════════════════════════════════════════════════════════════════
# Step 2: accelerate launch 4 GPU 跑 7 stage smoke (stage 5 真 ZeRO-3)
# ═════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: HLR smoke 4 GPU + ZeRO-3 + GC ($(date))"
echo ""

accelerate launch --num_processes 4 --main_process_port 29800 \
    Latent-SFT/smoke_test_hlr.py \
    --model "$MODEL_PATH" \
    --data "$PROFILED_DATA"
SMOKE_EXIT=$?

echo ""
echo "============================================================"
if [ $SMOKE_EXIT -eq 0 ]; then
    echo "✓ HLR Smoke Test 全部通过 ($(date))"
    echo "  下一步: 正式训练"
    echo "    sbatch Latent-SFT/submit_train_hlr.sh"
else
    echo "❌ HLR Smoke Test 失败 (exit $SMOKE_EXIT)"
fi
echo "============================================================"
exit $SMOKE_EXIT
