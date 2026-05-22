#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/train_hlr_%j.log

# HLR Stage 3 训练 — Qwen3-4B-Thinking 基座 (默认)
#
# 关键配置 + 已避开的坑:
#   1. ZeRO-3 + LoRA + gradient_checkpointing 三件套
#      → train.py 已硬编码 use_reentrant=True (踩坑 #14, trl#2514)
#   2. NCCL_P2P_DISABLE + NCCL_SHM_DISABLE
#      → zhuoyi 无 NVLink 拓扑必加, 否则 ZeRO init 第一次 broadcast 卡 30min (踩坑 #29)
#   3. tokenizer add_bos_token=False + pad_token 独立 (train.py 自动处理, 踩坑 #5/#27)
#   4. main_optimizer 单 param_group (不踩 PyTorch 2.10+ strict zip 坑 #16)
#   5. accelerator + 手动保存 ZeRO-3 gather state_dict (理论上避开 PEFT 空 adapter 坑 #23,
#      训练完必须用 ls -lh 验证 adapter_model.safetensors > 20MB)
#
# 资源:
#   - Qwen3-4B-Thinking + ZeRO-3 + GC + 4 GPU: 单卡显存约 15-25GB
#   - R1-Distill-7B 同配置: 单卡显存约 25-40GB
#   - large QOS (≤24 GPU/job); 4 GPU 足够, 用 normal 也行
#
# 流程:
#   - HLR_DATA 未设 → 训练前自动从 chains_template_cvrp20.jsonl 跑 entropy_profile (~1-2 hr)
#   - HLR_DATA 已设 → 跳过 entropy_profile, 直接训练
#
# 用法:
#   sbatch Latent-SFT/submit_train_hlr.sh                              # Qwen3-4B + auto rebuild
#   HLR_DATA=Latent-SFT/data/profiled_cvrp20.jsonl sbatch ...           # 跳过 profile
#   BASE_MODEL_TYPE=r1_distill sbatch Latent-SFT/submit_train_hlr.sh   # R1-Distill-7B
#   HLR_MODEL=/path/to/grpo_ckpt sbatch ...                            # 从 Stage 2 ckpt 起步

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── zhuoyi 多卡 NCCL 拓扑必加 (无 NVLink, 走 socket transport) ──
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source paths.sh

MODEL_PATH="${HLR_MODEL:-$BASE_MODEL}"
OUTPUT_DIR="${HLR_OUTPUT_DIR:-/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_${BASE_MODEL_TYPE}}"

echo "============================================================"
echo "  HLR Stage 3 训练 (Hierarchical Latent Reasoner)"
echo "============================================================"
echo "  HOST              = $HOST_ID"
echo "  BASE_MODEL_TYPE   = $BASE_MODEL_TYPE"
echo "  MODEL             = $MODEL_PATH"
echo "  OUTPUT_DIR        = $OUTPUT_DIR"
if [ -n "${HLR_DATA:-}" ]; then
    echo "  DATA (explicit)   = $HLR_DATA"
else
    echo "  DATA              = auto-rebuild from chains_template_cvrp20.jsonl (~1-2 hr profile)"
fi
echo "  GPUs              = 4 (ZeRO-3 + gradient_checkpointing + use_reentrant=True)"
echo "  采样参数 (推理参考) = T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K"
echo "============================================================"

# ── GPU 占用诊断 (排查 OOM 'luk 等用户绕开 SLURM 占卡' 场景) ──
echo "  GPU allocation diagnostic ($(date +%H:%M:%S))"
echo "    SLURM_JOB_ID          = ${SLURM_JOB_ID:-(not in slurm)}"
echo "    SLURM_JOB_GPUS        = ${SLURM_JOB_GPUS:-(unset)}"
echo "    CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES:-(unset)}"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader \
    | sed 's/^/      /'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | sed 's/^/      /' || echo "      (no processes)"
echo "============================================================"

# 任一卡 free < 20 GiB → exclude 当前节点 resubmit (脏节点自动规避)
_min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -n "$_min_free" ] && [ "$_min_free" -lt 20000 ]; then
    echo "❌ 某张卡 free memory 只剩 ${_min_free} MiB (<20 GiB), 别人在占!"
    _bad_node="$SLURM_NODELIST"
    _self_path="$(realpath "$0")"
    if [ -n "${BAD_NODES:-}" ]; then
        _new_bad="$BAD_NODES,$_bad_node"
    else
        _new_bad="$_bad_node"
    fi
    echo "  当前脏节点: $_bad_node, 累计 exclude: $_new_bad"
    sbatch --exclude="$_new_bad" --export=ALL,BAD_NODES="$_new_bad" "$_self_path"
    echo "  resubmit 完成, 当前 job 退出"
    exit 0
fi

# ── 训练参数 (sbatch 不显式覆盖 HLRConfig 默认, 让 config.py 单点管理) ──
EXTRA_DATA_FLAG=""
if [ -n "${HLR_DATA:-}" ]; then
    EXTRA_DATA_FLAG="--data $HLR_DATA"
fi

echo ""
echo ">>> HLR 训练 ($(date))"
echo "    epochs=3  batch=1 grad_accum=8 × 4 GPU = effective 32"
echo "    main_lr=2e-5 (LoRA)  latent_reasoner_lr=5e-5 (随机初始化, lr 稍高)"
echo "    LoRA r=64 alpha=128  scheduler=cosine warmup=0.05  wd=0.01"
echo "    Loss: α=1.0 (student CE) β=1.0 (align L1) γ=1.0 (teacher CE)"
echo "    Latent trigger: window=3 quantile=0.5 min=3 max=8 cooldown=24"
echo "    LR 架构 (auto from main config):"
echo "      Qwen3-4B: 36 main layers → 9 LR layers, hidden 2560/4=640"
echo "      R1-7B:    28 main layers → 7 LR layers, hidden 3584/4=896"
echo ""

accelerate launch --num_processes 4 --main_process_port 29700 \
    Latent-SFT/train.py --hlr \
    --model "$MODEL_PATH" \
    $EXTRA_DATA_FLAG \
    --zero_stage 3 \
    --gradient_checkpointing \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 10 --save_steps 200

TRAIN_EXIT=$?

# ── 训练后校验 (避开 PEFT ZeRO-3 空 adapter 坑) ──
echo ""
echo "============================================================"
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ HLR 训练异常退出 (exit $TRAIN_EXIT), 跳过后续校验"
    exit $TRAIN_EXIT
fi

FINAL_DIR="$OUTPUT_DIR/checkpoint-final"
if [ ! -f "$FINAL_DIR/latent_reasoner.pt" ]; then
    echo "❌ latent_reasoner.pt 未保存!"
    ls -la "$FINAL_DIR/" 2>/dev/null || echo "  (final dir 不存在)"
    exit 1
fi

LR_SIZE=$(stat -c '%s' "$FINAL_DIR/latent_reasoner.pt" 2>/dev/null || echo 0)
ADAPTER_SIZE=$(stat -c '%s' "$FINAL_DIR/adapter_model.safetensors" 2>/dev/null || echo 0)

echo "  HLR 训练完成! $(date)"
echo "  Checkpoint: $FINAL_DIR"
echo "    adapter_model.safetensors  : $((ADAPTER_SIZE / 1024 / 1024)) MB  (期望 >20 MB)"
echo "    latent_reasoner.pt          : $((LR_SIZE / 1024 / 1024)) MB  (期望 >50 MB)"
echo ""

# 空 adapter 防御 (踩坑 #23)
if [ "$ADAPTER_SIZE" -lt 1048576 ]; then  # < 1 MB
    echo "⚠️ adapter_model.safetensors 异常小 (<1MB)!"
    echo "   可能踩到 PEFT ZeRO-3 空权重坑 (踩坑 #23)"
    echo "   下次试 --zero_stage 2 或检查 PEFT/TRL 版本"
fi
if [ "$LR_SIZE" -lt 10485760 ]; then  # < 10 MB
    echo "⚠️ latent_reasoner.pt 异常小 (<10MB)!"
    echo "   LatentReasoner 应该 ~80M (Qwen3-4B) 或 ~108M (R1-7B) params × 4B fp32 ≈ 数百 MB"
fi

echo ""
echo "  下一步: 推理 sanity check"
echo "    python Latent-SFT/inference.py --model $FINAL_DIR --prompt ..."
echo "============================================================"
