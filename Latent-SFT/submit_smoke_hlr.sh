#!/bin/bash
#SBATCH --qos express
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_hlr_%j.log

# HLR 全流程 Smoke Test (一气呵成, 不需要用户介入)
#
# 自动执行:
#   Step 1: 检查 profiled jsonl 是否存在; 不存在 → 用 base model 跑 entropy_profile
#           (limit 200 条, ~5-10 min, 1 GPU bf16, 输出 profiled_smoke_cvrp20.jsonl)
#   Step 2: 跑完整 7 stage smoke_test_hlr.py
#           - stage 1-3: config / 小组件 / LR forward+KV cache
#           - stage 4-6: HLRDataset + compute_hlr_loss + HLRInferenceEngine (需主模型 + profiled)
#           - stage 7:   LatentReasoner state_dict 保存/加载
#
# 默认配置 (env 覆盖):
#   BASE_MODEL_TYPE       = r1_distill | qwen3_thinking (默认 r1_distill, 跟随 paths.sh)
#   HLR_MODEL             = 主模型路径 (默认 $BASE_MODEL)
#   SMOKE_PROFILE_LIMIT   = entropy_profile 处理样本数 (默认 200, 0=全量)
#   SMOKE_FORCE_REPROFILE = 1 → 即使 profiled 已存在也强制重跑
#
# 用法:
#   sbatch Latent-SFT/submit_smoke_hlr.sh                                # 默认 R1-Distill
#   BASE_MODEL_TYPE=qwen3_thinking sbatch Latent-SFT/submit_smoke_hlr.sh # Qwen3-4B-Thinking
#
# 1 GPU express, 总耗时 ~15-20 min (含模型加载 + profile 200 条 + 7 stage smoke)
# 验证通过后, 用 submit_train_hlr.sh 启动正式训练 (会用全量数据 auto_rebuild)

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# zhuoyi 单卡 smoke 也加上 (entropy_profile 没多卡, 不强制需要, 但保持习惯一致)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
source paths.sh

# cwd = UniCOP/, 所有相对路径基于此

MODEL_PATH="${HLR_MODEL:-$BASE_MODEL}"
RAW_DATA="UniCOP-Distill/data/chains_template_cvrp20.jsonl"

# smoke 用专属 profiled jsonl, 不覆盖真实训练用的 profiled_cvrp20.jsonl
PROFILED_DATA="Latent-SFT/data/profiled_smoke_${BASE_MODEL_TYPE}_cvrp20.jsonl"

PROFILE_LIMIT="${SMOKE_PROFILE_LIMIT:-200}"
FORCE_REPROFILE="${SMOKE_FORCE_REPROFILE:-0}"

echo "============================================================"
echo "  HLR 全流程 Smoke Test"
echo "============================================================"
echo "  HOST              = $HOST_ID"
echo "  BASE_MODEL_TYPE   = $BASE_MODEL_TYPE"
echo "  MODEL             = $MODEL_PATH"
echo "  RAW DATA          = $RAW_DATA"
echo "  PROFILED OUTPUT   = $PROFILED_DATA"
echo "  PROFILE_LIMIT     = $PROFILE_LIMIT (0=全量)"
echo "  FORCE_REPROFILE   = $FORCE_REPROFILE"
echo "============================================================"

# ── GPU 占用诊断 + 脏节点自动 exclude ──
echo "  GPU diagnostic ($(date +%H:%M:%S))"
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
# Step 1: 生成 smoke profiled jsonl (如果不存在 / 强制重跑)
# ═════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: entropy_profile ($(date))"

if [ -f "$PROFILED_DATA" ] && [ "$FORCE_REPROFILE" != "1" ]; then
    _existing=$(wc -l < "$PROFILED_DATA")
    echo "✓ profiled jsonl 已存在: $PROFILED_DATA ($_existing 行)"
    echo "  跳过 entropy_profile (SMOKE_FORCE_REPROFILE=1 强制重跑)"
else
    if [ ! -f "$RAW_DATA" ]; then
        echo "❌ 原始 chains 数据不存在: $RAW_DATA"
        exit 1
    fi
    echo "  原始数据: $RAW_DATA"
    echo "  输出:     $PROFILED_DATA"
    echo "  limit:    $PROFILE_LIMIT"
    echo ""
    python Latent-SFT/entropy_profile.py \
        --model "$MODEL_PATH" \
        --data "$RAW_DATA" \
        --output "$PROFILED_DATA" \
        --limit "$PROFILE_LIMIT"
    _ep_exit=$?
    if [ $_ep_exit -ne 0 ] || [ ! -f "$PROFILED_DATA" ]; then
        echo "❌ entropy_profile 失败 (exit $_ep_exit)"
        exit 1
    fi
    _n_with_seg=$(grep -c '"latent_segments": \[{' "$PROFILED_DATA" 2>/dev/null || echo 0)
    _n_total=$(wc -l < "$PROFILED_DATA")
    echo ""
    echo "✓ profiled jsonl 生成完成: $_n_total 行, $_n_with_seg 行含 latent 段"
    if [ "$_n_with_seg" = "0" ]; then
        echo "⚠ 0 行含 latent 段! entropy_quantile 太严或 cot 都很乱, 检查阈值"
    fi
fi

# ═════════════════════════════════════════════════════════════════
# Step 2: 完整 7 stage smoke test
# ═════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: HLR smoke_test (7 stage) ($(date))"
echo ""

python Latent-SFT/smoke_test_hlr.py \
    --model "$MODEL_PATH" \
    --data "$PROFILED_DATA"
SMOKE_EXIT=$?

echo ""
echo "============================================================"
if [ $SMOKE_EXIT -eq 0 ]; then
    echo "✓ HLR Smoke Test 全部通过 ($(date))"
    echo ""
    echo "  Smoke profiled (200 条) 留在: $PROFILED_DATA"
    echo "  下一步: 正式训练 (全量数据 auto_rebuild + 4 GPU + ZeRO-3)"
    echo "    sbatch Latent-SFT/submit_train_hlr.sh"
    echo "  或跳过 auto_rebuild 用已有 profiled:"
    echo "    HLR_DATA=Latent-SFT/data/profiled_cvrp20.jsonl sbatch Latent-SFT/submit_train_hlr.sh"
else
    echo "❌ HLR Smoke Test 失败 (exit $SMOKE_EXIT, $(date))"
    echo "  看 stage X FAIL 的 traceback, 修后重跑"
fi
echo "============================================================"

exit $SMOKE_EXIT
