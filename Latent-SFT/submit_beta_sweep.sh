#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/beta_sweep_%j.log

# HLR β (align loss) sweep: 0.5 / 1.5 / 2.0
# (β=1.0 已单独跑过，不重复)
#
# 流程:
#   Step 0: entropy_profile (4 GPU, 有缓存跳过)
#   对每个 β:
#     Step 1: 训练 (4 GPU ZeRO-3 + GC)
#     Step 2: eval_hlr_compare (1 GPU: merge → baseline → HLR → compare.json)
#   Step 3: 汇总所有 compare.json → 对比表
#
# 用法:
#   HLR_MODEL=/path/to/instruct_model \
#   HLR_DATA=Latent-SFT/data/profiled_..._10k.jsonl \
#       sbatch Latent-SFT/submit_beta_sweep.sh
#
# 可选 env:
#   HLR_MODEL          训练起点模型 (必填)
#   HLR_DATA           profiled jsonl (跳过 profile)
#   EVAL_NUM_TEST      eval 实例数 (默认 100)
#   EVAL_PROBLEM       eval 问题 (默认 cvrp)
#   EVAL_PROBLEM_SIZE  eval 规模 (默认 20)

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# zhuoyi 多卡 NCCL 拓扑
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export DEEPSPEED_TIMEOUT=3600
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20480
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=WARN

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP

export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source paths.sh

# ── 必填校验 ──
if [ -z "${HLR_MODEL:-}" ]; then
    echo "❌ 必须设 HLR_MODEL (训练起点模型路径)"
    exit 1
fi

MODEL_PATH="$HLR_MODEL"
HLR_EPOCHS="${HLR_EPOCHS:-1}"

# ── Eval 参数 (所有 β 共用, seed=9999 → 相同测试集) ──
EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"
EVAL_PROBLEM="${EVAL_PROBLEM:-cvrp}"
EVAL_PROBLEM_SIZE="${EVAL_PROBLEM_SIZE:-20}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"

# ── Sweep 配置 ──
BETAS="${BETAS:-0.5 1.5 2.0}"
HLR_ALPHA=1.0
HLR_GAMMA=1.0

# ── Server 酱 ──
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}
trap 'notify "❌ HLR β sweep 失败 (line $LINENO)"' ERR

echo "============================================================"
echo "  HLR β Sweep: ${BETAS}"
echo "============================================================"
echo "  HOST            = $HOST_ID"
echo "  MODEL           = $MODEL_PATH"
echo "  α=${HLR_ALPHA}  β sweep=${BETAS}  γ=${HLR_GAMMA}"
echo "  epochs=${HLR_EPOCHS}  batch=1  GA=8  ×4 GPU → eff=32"
echo "  Eval: ${EVAL_PROBLEM}-${EVAL_PROBLEM_SIZE}  n=${EVAL_NUM_TEST}  T=${EVAL_TEMPERATURE}"
echo "  Test-set seed   = 9999 (evaluate.py line 890, 固定, 所有 β 同集)"
echo "============================================================"

# ── GPU 诊断 ──
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader \
    | sed 's/^/    /'
_min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -n "$_min_free" ] && [ "$_min_free" -lt 20000 ]; then
    echo "❌ free ${_min_free} MiB < 20 GiB, 节点脏, resubmit exclude"
    _bad_node="$SLURM_NODELIST"
    _self_path="$(realpath "$0")"
    _new_bad="${BAD_NODES:+$BAD_NODES,}$_bad_node"
    sbatch --exclude="$_new_bad" --export=ALL,BAD_NODES="$_new_bad" "$_self_path"
    exit 0
fi
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 0: entropy_profile (4 GPU, 有缓存跳过)
# ══════════════════════════════════════════════════════════════════
RAW_DATA="${RAW_DATA:-UniCOP-Distill/data/chains_template_cvrp20_10k.jsonl}"
PROFILED_DATA="${HLR_DATA:-Latent-SFT/data/profiled_${BASE_MODEL_TYPE}_cvrp20_10k.jsonl}"

echo ""
echo ">>> Step 0: entropy_profile (4 GPU) ($(date))"
echo "    raw:      $RAW_DATA"
echo "    output:   $PROFILED_DATA"

if [ -f "$PROFILED_DATA" ] && [ -z "${FORCE_REPROFILE:-}" ]; then
    echo "    ✓ profiled jsonl 已存在 ($(wc -l < "$PROFILED_DATA") 行), 跳过"
else
    [ -f "$RAW_DATA" ] || { echo "❌ $RAW_DATA 不存在"; exit 1; }
    mkdir -p "$(dirname "$PROFILED_DATA")"

    PROFILE_TMP=$(mktemp -d -p "${TMPDIR:-/tmp}" entropy_profile.XXXXXX)
    echo "    临时 shard 目录: $PROFILE_TMP"
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python Latent-SFT/entropy_profile.py \
            --model "$MODEL_PATH" --data "$RAW_DATA" \
            --output "$PROFILE_TMP/shard_${i}.jsonl" \
            --num_shards 4 --shard_rank $i \
            > "$PROFILE_TMP/log_${i}.log" 2>&1 &
    done
    wait

    profile_failed=0
    for i in 0 1 2 3; do
        if [ ! -s "$PROFILE_TMP/shard_${i}.jsonl" ]; then
            echo "❌ shard $i 失败, log:"; cat "$PROFILE_TMP/log_${i}.log" | sed 's/^/      /'
            profile_failed=1
        else
            echo "    ✓ shard $i: $(wc -l < "$PROFILE_TMP/shard_${i}.jsonl") 行"
        fi
    done
    [ $profile_failed -ne 0 ] && { echo "❌ profile 失败"; notify "❌ HLR β sweep: profile 失败"; exit 1; }

    cat "$PROFILE_TMP/shard_"*.jsonl > "${PROFILED_DATA}.tmp"
    mv "${PROFILED_DATA}.tmp" "$PROFILED_DATA"
    rm -rf "$PROFILE_TMP"
    echo "    ✓ 合并完成: $(wc -l < "$PROFILED_DATA") 行"
fi

# ====================================================================
# 训练 + eval 循环
# ====================================================================
SUMMARY_TABLE="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/beta_sweep_summary_${SLURM_JOB_ID:-local}.txt"
echo "" > "$SUMMARY_TABLE"
echo "β | parse_rate | feasible_rate | avg_dist | CoT tokens | wall_hlr(s) | wall_baseline(s)" >> "$SUMMARY_TABLE"
echo "--|------------|---------------|----------|------------|-------------|-----------------" >> "$SUMMARY_TABLE"

for BETA in $BETAS; do
    echo ""
    echo "################################################################"
    echo "###  β = ${BETA}  ($(date))"
    echo "################################################################"

    OUTPUT_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_beta${BETA}"
    HLR_CHECKPOINT="$OUTPUT_DIR/checkpoint-final"

    # ── Step 1: 训练 ──
    echo ""
    echo ">>> Step 1 (β=${BETA}): HLR 训练 ($(date))"
    echo "    output_dir = $OUTPUT_DIR"
    echo ""

    accelerate launch --num_processes 4 --main_process_port 29700 \
        Latent-SFT/train.py \
        --model "$MODEL_PATH" \
        --data "$PROFILED_DATA" \
        --epochs "$HLR_EPOCHS" \
        --alpha "$HLR_ALPHA" --beta "$BETA" --gamma "$HLR_GAMMA" \
        --zero_stage 3 --gradient_checkpointing \
        --output_dir "$OUTPUT_DIR" \
        --logging_steps 10 --save_steps 200

    if [ $? -ne 0 ]; then
        echo "❌ β=${BETA} 训练失败, 跳过 eval 继续下一个"
        notify "❌ HLR β=${BETA} 训练失败"
        continue
    fi

    # ── Step 2: 产物校验 ──
    if [ ! -f "$HLR_CHECKPOINT/latent_reasoner.pt" ] || [ ! -f "$HLR_CHECKPOINT/adapter_model.safetensors" ]; then
        echo "❌ β=${BETA} checkpoint 不完整, 跳过 eval"
        notify "❌ HLR β=${BETA} ckpt 不完整"
        continue
    fi
    ADAPTER_SIZE=$(stat -c '%s' "$HLR_CHECKPOINT/adapter_model.safetensors")
    LR_SIZE=$(stat -c '%s' "$HLR_CHECKPOINT/latent_reasoner.pt")
    echo "  ✓ adapter=$(($ADAPTER_SIZE/1024/1024))MB  lr=$(($LR_SIZE/1024/1024))MB"

    # ── Step 3: 对比 eval (1 GPU) ──
    echo ""
    echo ">>> Step 3 (β=${BETA}): HLR vs baseline eval ($(date))"
    echo ""

    CUDA_VISIBLE_DEVICES=0 python Latent-SFT/eval_hlr_compare.py \
        --hlr_checkpoint "$HLR_CHECKPOINT" \
        --problem "$EVAL_PROBLEM" \
        --problem_size "$EVAL_PROBLEM_SIZE" \
        --num_test "$EVAL_NUM_TEST" \
        --max_completion_length "$EVAL_MAX_LEN" \
        --temperature "$EVAL_TEMPERATURE" \
        --batch_size "$EVAL_BATCH_SIZE"

    EVAL_EXIT=$?
    if [ $EVAL_EXIT -ne 0 ]; then
        echo "❌ β=${BETA} eval 失败"
        notify "❌ HLR β=${BETA} eval 失败"
        continue
    fi

    # ── 提取关键指标到汇总表 ──
    COMPARE_JSON="$HLR_CHECKPOINT/compare_eval/compare.json"
    if [ -f "$COMPARE_JSON" ]; then
        python -c "
import json
with open('$COMPARE_JSON') as f:
    d = json.load(f)
hlr = d.get('hlr', {})
baseline = d.get('baseline', {})
# hlr 和 baseline 都是 list[dict], 取 combo summary
def get_metrics(o):
    if isinstance(o, list) and len(o) > 0:
        return o[0]
    return o or {}
h = get_metrics(hlr)
b = get_metrics(baseline)
print(f\"  β=${BETA}\")
print(f\"    baseline: parse={b.get('parse_rate',0):.3f}  feas={b.get('global_feas_rate',0):.3f}  dist={b.get('avg_instance_distance',0):.2f}\")
print(f\"    HLR:      parse={h.get('parse_rate',0):.3f}  feas={h.get('global_feas_rate',0):.3f}  dist={h.get('avg_instance_distance',0):.2f}\")
cot_hlr = h.get('total_completion_tokens', 0)
cot_base = b.get('total_completion_tokens', 0)
wall_hlr = d.get('hlr_wall_seconds', 0)
wall_base = d.get('baseline_wall_seconds', 0)
print(f\"    CoT tokens: HLR={cot_hlr}  baseline={cot_base}\")
print(f\"    wall (s):   HLR={wall_hlr:.0f}  baseline={wall_base:.0f}\")
"
    fi

    notify "✅ HLR β=${BETA} 训练+eval 完成" "$HLR_CHECKPOINT/compare_eval/compare.json"
done

# ══════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  β Sweep 全部完成 ($(date))"
echo "============================================================"
echo "  各 β checkpoint:"
for BETA in $BETAS; do
    CKPT="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_beta${BETA}/checkpoint-final"
    COMP="$CKPT/compare_eval/compare.json"
    if [ -f "$COMP" ]; then
        echo "    β=${BETA}  ✓  $COMP"
    else
        echo "    β=${BETA}  ❌  (失败/跳过)"
    fi
done
echo "============================================================"
notify "✅ HLR β sweep (${BETAS}) 全部完成"
