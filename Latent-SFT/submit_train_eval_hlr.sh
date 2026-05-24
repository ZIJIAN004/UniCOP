#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/train_eval_hlr_%j.log

# HLR 一体化 Pipeline (single sbatch): 训练 → merge → baseline eval → HLR eval → 对比
#
# 资源: 4 GPU large QOS, 整个 job 占 4 卡;
#       Step 1 训练用 4 卡; Step 2 eval 用 1 卡 (闲置 3 卡换简化);
#       全程 ~3-7 hr (1 epoch / 3 epoch 不同).
#
# 流程 (mirror submit_sft_qwen3_full.sh 的一体化设计):
#   Step 0: entropy_profile (单 GPU, 与 ZeRO-3 init 隔离, 有缓存就跳)
#   Step 1: 训练            (4 GPU ZeRO-3 + GC + accelerate launch train.py)
#   Step 2: 产物校验         (adapter > 20MB, latent_reasoner.pt > 50MB)
#   Step 3: 对比 eval        (单 GPU 进程, 内部自动: merge → baseline → HLR → 对比表)
#
# 用法 (cwd 任意, 脚本内部 cd):
#   HLR_MODEL=/path/to/sft_final_model \
#   BASE_MODEL_TYPE=qwen3_thinking \
#   HLR_OUTPUT_DIR=/path/to/output_hlr_run1 \
#   HLR_EPOCHS=1 \
#       sbatch Latent-SFT/submit_train_eval_hlr.sh
#
# 必填 env:
#   HLR_MODEL          训练起点模型 (SFT merged model 路径)
#
# 可选 env:
#   BASE_MODEL_TYPE    qwen3_thinking | r1_distill   (默认 qwen3_thinking)
#   HLR_OUTPUT_DIR     训练输出目录                   (默认 output_hlr_${BASE_MODEL_TYPE})
#   HLR_EPOCHS         训练 epoch 数                  (默认 3, 推荐 1 减量)
#   HLR_DATA           profiled jsonl 路径 (跳过 Step 0 重跑)
#   EVAL_NUM_TEST      eval 实例数                    (默认 100)
#   EVAL_PROBLEM       eval 问题                      (默认 cvrp)
#   EVAL_PROBLEM_SIZE  eval 规模                      (默认 20)
#   EVAL_MAX_LEN       eval max_completion_length    (默认 4096)
#   EVAL_BATCH_SIZE    eval batch_size (两轮共用)    (默认 4)
#   EVAL_TEMPERATURE   eval temperature              (默认 0.0 贪心)

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# zhuoyi 多卡 NCCL 拓扑必加 (无 NVLink, 踩坑 #29)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source paths.sh

# ── 必填校验 ────────────────────────────────────────────────────────
if [ -z "${HLR_MODEL:-}" ]; then
    echo "❌ 必须设 HLR_MODEL (训练起点模型路径)"
    exit 1
fi

# ── 配置派生 ────────────────────────────────────────────────────────
MODEL_PATH="$HLR_MODEL"
OUTPUT_DIR="${HLR_OUTPUT_DIR:-/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_${BASE_MODEL_TYPE}}"
HLR_CHECKPOINT="$OUTPUT_DIR/checkpoint-final"

EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"
EVAL_PROBLEM="${EVAL_PROBLEM:-cvrp}"
EVAL_PROBLEM_SIZE="${EVAL_PROBLEM_SIZE:-20}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"

# ── Server 酱通知 ──────────────────────────────────────────────────
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}
trap 'notify "❌ HLR train+eval 失败 (line $LINENO)"' ERR

echo "============================================================"
echo "  HLR 一体化 Pipeline (train → eval)"
echo "============================================================"
echo "  HOST              = $HOST_ID"
echo "  BASE_MODEL_TYPE   = $BASE_MODEL_TYPE"
echo "  MODEL (起点)      = $MODEL_PATH"
echo "  OUTPUT_DIR        = $OUTPUT_DIR"
echo "  HLR_EPOCHS        = ${HLR_EPOCHS:-3 (default)}"
if [ -n "${HLR_DATA:-}" ]; then
    echo "  DATA (explicit)   = $HLR_DATA"
else
    echo "  DATA              = (Step 0 自动跑 entropy_profile, ~1-2 hr)"
fi
echo "  EVAL              = $EVAL_PROBLEM-$EVAL_PROBLEM_SIZE n=$EVAL_NUM_TEST "
echo "                       max_len=$EVAL_MAX_LEN  bs=$EVAL_BATCH_SIZE  T=$EVAL_TEMPERATURE"
echo "============================================================"

# ── GPU 占用诊断 (排查 luk 等绕开 SLURM 占卡) ────────────────────────
echo "  GPU allocation diagnostic ($(date +%H:%M:%S))"
echo "    SLURM_JOB_ID          = ${SLURM_JOB_ID:-(not in slurm)}"
echo "    SLURM_JOB_GPUS        = ${SLURM_JOB_GPUS:-(unset)}"
echo "    CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES:-(unset)}"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader \
    | sed 's/^/      /'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | sed 's/^/      /' || echo "      (no processes)"

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
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 0: entropy_profile (单 GPU, ~1-2 hr; 必须先于训练, 与 ZeRO-3 init 隔离)
# ══════════════════════════════════════════════════════════════════
# 为何独立? train.py 内部 inline 跑会触发 ZeRO-3 init 后 model.forward
# 报 'weight' must be 2-D (embedding.weight 已被 partition 成 1D 切片).
# 改为 sbatch 这里单 GPU 不带 DeepSpeed 跑, 训练时传 --data 跳过.
RAW_DATA="UniCOP-Distill/data/chains_template_cvrp20.jsonl"
PROFILED_DATA="${HLR_DATA:-Latent-SFT/data/profiled_${BASE_MODEL_TYPE}_cvrp20.jsonl}"

echo ""
echo ">>> Step 0: entropy_profile ($(date))"
echo "    raw:      $RAW_DATA"
echo "    output:   $PROFILED_DATA"

if [ -f "$PROFILED_DATA" ] && [ -z "${FORCE_REPROFILE:-}" ]; then
    _existing=$(wc -l < "$PROFILED_DATA")
    echo "    ✓ profiled jsonl 已存在 ($_existing 行), 跳过 (FORCE_REPROFILE=1 强制重跑)"
else
    if [ ! -f "$RAW_DATA" ]; then
        echo "❌ 原始 chains 不存在: $RAW_DATA"
        notify "❌ HLR profile 失败: 原始数据缺失"
        exit 1
    fi
    mkdir -p "$(dirname "$PROFILED_DATA")"
    CUDA_VISIBLE_DEVICES=0 python Latent-SFT/entropy_profile.py \
        --model "$MODEL_PATH" \
        --data "$RAW_DATA" \
        --output "$PROFILED_DATA"
    _ep_exit=$?
    if [ $_ep_exit -ne 0 ] || [ ! -f "$PROFILED_DATA" ]; then
        echo "❌ entropy_profile 失败 (exit $_ep_exit)"
        notify "❌ HLR entropy_profile 失败 (exit $_ep_exit)"
        exit 1
    fi
    echo "    ✓ profile 完成: $(wc -l < "$PROFILED_DATA") 行"
fi

# ══════════════════════════════════════════════════════════════════
# Step 1: HLR 训练 (4 GPU ZeRO-3 + GC + LoRA)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: HLR 训练 ($(date))"
echo "    epochs=${HLR_EPOCHS:-3}  batch=1 grad_accum=8 × 4 GPU = effective 32"
echo "    main_lr=2e-5 (LoRA)  latent_reasoner_lr=5e-5"
echo "    LoRA r=64 alpha=128  scheduler=cosine warmup=0.05 wd=0.01"
echo "    Loss: α=1.0 (student CE) β=1.0 (align L1) γ=1.0 (teacher CE)"
echo "    Latent trigger: window=3 quantile=0.5 min=3 max=8 cooldown=24"
echo "    profiled data: $PROFILED_DATA"
echo ""

EXTRA_EPOCHS_FLAG=""
if [ -n "${HLR_EPOCHS:-}" ]; then
    EXTRA_EPOCHS_FLAG="--epochs $HLR_EPOCHS"
fi

accelerate launch --num_processes 4 --main_process_port 29700 \
    Latent-SFT/train.py \
    --model "$MODEL_PATH" \
    --data "$PROFILED_DATA" \
    $EXTRA_EPOCHS_FLAG \
    --zero_stage 3 \
    --gradient_checkpointing \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 10 --save_steps 200

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Step 1 训练失败 (exit $TRAIN_EXIT), 跳过 eval"
    notify "❌ HLR train 失败 (exit $TRAIN_EXIT)" "$OUTPUT_DIR"
    exit $TRAIN_EXIT
fi

# ══════════════════════════════════════════════════════════════════
# Step 2: 训练产物校验 (避开 PEFT ZeRO-3 空 adapter 坑 #23)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 训练产物校验 ($(date))"

if [ ! -f "$HLR_CHECKPOINT/latent_reasoner.pt" ]; then
    echo "❌ $HLR_CHECKPOINT/latent_reasoner.pt 不存在!"
    ls -la "$HLR_CHECKPOINT/" 2>/dev/null || echo "  (final dir 不存在)"
    notify "❌ HLR 训练 latent_reasoner.pt 缺失"
    exit 1
fi
if [ ! -f "$HLR_CHECKPOINT/adapter_model.safetensors" ]; then
    echo "❌ $HLR_CHECKPOINT/adapter_model.safetensors 不存在!"
    notify "❌ HLR 训练 adapter 缺失"
    exit 1
fi

LR_SIZE=$(stat -c '%s' "$HLR_CHECKPOINT/latent_reasoner.pt")
ADAPTER_SIZE=$(stat -c '%s' "$HLR_CHECKPOINT/adapter_model.safetensors")
echo "  adapter_model.safetensors : $((ADAPTER_SIZE / 1024 / 1024)) MB  (期望 >20 MB)"
echo "  latent_reasoner.pt         : $((LR_SIZE / 1024 / 1024)) MB  (期望 >50 MB)"

if [ "$ADAPTER_SIZE" -lt 1048576 ]; then
    echo "❌ adapter 异常小 (<1MB), 可能 PEFT ZeRO-3 空权重 (踩坑 #23)"
    notify "❌ HLR adapter 异常小 ($((ADAPTER_SIZE / 1024)) KB)"
    exit 1
fi
if [ "$LR_SIZE" -lt 10485760 ]; then
    echo "⚠️ latent_reasoner.pt 异常小 (<10MB), 检查 Phase 1 限制"
    notify "⚠️ HLR latent_reasoner 异常小 ($((LR_SIZE / 1024 / 1024)) MB)"
fi

notify "✅ HLR 训练完成" "$HLR_CHECKPOINT  开始 eval..."

# ══════════════════════════════════════════════════════════════════
# Step 3: 对比 eval (内部: merge → baseline → HLR → 对比表)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: HLR vs baseline 对比 eval ($(date))"
echo "    (内部: auto-merge LoRA → baseline (local) → HLR (hlr backend) → compare.json)"
echo ""

python Latent-SFT/eval_hlr_compare.py \
    --hlr_checkpoint "$HLR_CHECKPOINT" \
    --problem "$EVAL_PROBLEM" \
    --problem_size "$EVAL_PROBLEM_SIZE" \
    --num_test "$EVAL_NUM_TEST" \
    --max_completion_length "$EVAL_MAX_LEN" \
    --temperature "$EVAL_TEMPERATURE" \
    --batch_size "$EVAL_BATCH_SIZE"

EVAL_EXIT=$?

echo ""
echo "============================================================"
if [ $EVAL_EXIT -eq 0 ]; then
    echo "✅ HLR Pipeline 完成 ($(date))"
    echo "  训练 ckpt:  $HLR_CHECKPOINT"
    echo "  对比结果:   $HLR_CHECKPOINT/compare_eval/compare.json"
    notify "✅ HLR train+eval 完成" "$HLR_CHECKPOINT/compare_eval/compare.json"
else
    echo "❌ Step 3 eval 失败 (exit $EVAL_EXIT)"
    notify "❌ HLR eval 失败 (exit $EVAL_EXIT)" "训练 ckpt 已保存: $HLR_CHECKPOINT"
fi
echo "============================================================"
exit $EVAL_EXIT
