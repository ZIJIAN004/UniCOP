#!/bin/bash
# submit_sweep_eval_hlr.sh
# ── HLR β sweep: merge+eval 多 checkpoint 并行 (1 pa = 1 卡) ──
#
#   背景: submit_beta_sweep.sh 提交的训练 job 是 train-only,
#   各 β 跑完只留下 HLR checkpoint (output_hlr_beta{β}/checkpoint-final),
#   本 submit 一个 job 补齐:
#     [1] 校验每个 checkpoint (adapter + latent_reasoner.pt 存在)
#     [2] HLR vs baseline eval (并行, 每 β 绑一张卡)
#         eval_hlr_compare.py 内部: merge → baseline → HLR → compare.json
#         卡数不足时分批。
#
#   结果: output_hlr_beta{β}/checkpoint-final/compare_eval/compare.json
#   幂等: compare.json 存在且有效则跳过该 β (重跑只补缺失的)
#
#   提交 (submit_beta_sweep.sh 自动带 dependency, 不需要手动):
#     sbatch --export=ALL,DIRS="output_hlr_beta0.5 output_hlr_beta1.5 output_hlr_beta2.0" \
#       submit_sweep_eval_hlr.sh
#   可覆盖:
#     EVAL_NUM_TEST=100  EVAL_GPUS=0,1,2,3  EVAL_PROBLEM=cvrp  EVAL_PROBLEM_SIZE=20

#SBATCH --qos=large
#SBATCH --gpus=4
#SBATCH --job-name=zijia_hlr_sweep_eval
#SBATCH --comment="zijianliu, HLR β sweep BO1 eval (4 pa parallel), do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/sweep_eval_hlr_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/sweep_eval_hlr_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

echo "[sweep_eval_hlr] 启动 $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none}"
set -uo pipefail   # 必须在 conda activate 之后, 不加 -e (参照 Reason-Mask)

cd /homes/zhuoyi/zijianliu/UniCOP

# ── 参数 ──────────────────────────────────────────────────────────
DIRS="${DIRS:?必须设 DIRS (checkpoint 目录列表, 空格分隔)}"
EVAL_NUM_TEST="${EVAL_NUM_TEST:-100}"
EVAL_PROBLEM="${EVAL_PROBLEM:-cvrp}"
EVAL_PROBLEM_SIZE="${EVAL_PROBLEM_SIZE:-20}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"

IFS=',' read -ra GPU_POOL <<< "$EVAL_GPUS"
echo "============================================================"
echo "  HLR Sweep Eval (并行, 每 pa 1 卡)"
echo "============================================================"
echo "  DIRS         = $DIRS"
echo "  Eval         = ${EVAL_PROBLEM}-${EVAL_PROBLEM_SIZE}  n=${EVAL_NUM_TEST}  T=${EVAL_TEMPERATURE}"
echo "  卡池         = [${EVAL_GPUS}] (每 pa 1 卡, 不足分批)"
echo "  seed         = 9999 (evaluate.py, 测试集固定)"
echo "============================================================"

# Server酱
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=${1:0:100}" \
        --data-urlencode "desp=${2:0:500}" > /dev/null 2>&1 || true
}

# ── GPU 占用预检 ──
export SUBMIT_SCRIPT="$(pwd)/Latent-SFT/submit_sweep_eval_hlr.sh"
export BASE_EXCLUDE=""
if [ -f "$(pwd)/preflight_gpu.sh" ]; then
    source "$(pwd)/preflight_gpu.sh"
    preflight_gpu_or_resubmit
fi

# ====================================================================
echo ""
echo "############## [1] 校验 checkpoint ##############  $(date '+%F %T')"
# ====================================================================
declare -A TAGS
declare -A CKPTS
EVAL_DIRS=""
EVAL_FAILED=""

for d in $DIRS; do
    d="${d%/}"
    ckpt="${d}/checkpoint-final"
    tag="beta${d##*beta}"

    echo ""
    echo "── [$tag] ──  $(date '+%F %T')"
    echo "   checkpoint: $ckpt"

    if [ ! -f "$ckpt/latent_reasoner.pt" ]; then
        echo "  [$tag] ❌ latent_reasoner.pt 不存在, 跳过"
        EVAL_FAILED="$EVAL_FAILED $tag(no-lr)"
        continue
    fi
    if [ ! -f "$ckpt/adapter_model.safetensors" ]; then
        echo "  [$tag] ❌ adapter_model.safetensors 不存在, 跳过"
        EVAL_FAILED="$EVAL_FAILED $tag(no-adapter)"
        continue
    fi

    ADAPTER_SIZE=$(stat -c '%s' "$ckpt/adapter_model.safetensors")
    LR_SIZE=$(stat -c '%s' "$ckpt/latent_reasoner.pt")
    echo "  [$tag] ✓ adapter=$(($ADAPTER_SIZE/1024/1024))MB  lr=$(($LR_SIZE/1024/1024))MB"

    TAGS["$tag"]="$tag"
    CKPTS["$tag"]="$ckpt"
    EVAL_DIRS="$EVAL_DIRS $tag"
done

if [ -z "$EVAL_DIRS" ]; then
    echo "[FATAL] 没有任何 checkpoint 通过校验, 退出。失败:$EVAL_FAILED"
    notify "❌ HLR sweep eval: 全部校验失败" "$EVAL_FAILED  $(date '+%F %T')"
    exit 1
fi

# ====================================================================
echo ""
echo "############## [2] HLR vs baseline eval (并行, 每 pa 1 卡) ##############  $(date '+%F %T')"
# ====================================================================
NGPU="${#GPU_POOL[@]}"
ALL_TAGS=($EVAL_DIRS)

# 幂等判定: compare.json 存在且 n_eval == EVAL_NUM_TEST → 跳过
COMPLETE_TAGS=""
TODO_TAGS=""
for tag in "${ALL_TAGS[@]}"; do
    comp_json="${CKPTS[$tag]}/compare_eval/compare.json"
    if [ -f "$comp_json" ] && python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('hlr') else 1)" "$comp_json" 2>/dev/null; then
        echo "  [$tag] ✓ compare.json 已存在, 跳过 (幂等)"
        COMPLETE_TAGS="$COMPLETE_TAGS $tag"
    else
        TODO_TAGS="$TODO_TAGS $tag"
    fi
done

if [ -z "$TODO_TAGS" ]; then
    echo "  全部已完成, 无需 eval"
else
    run_one_pa() {
        local tag="$1" ckpt="$2" gpu="$3"
        echo "[$tag][gpu$gpu] >>> HLR eval  $(date '+%F %T')"
        CUDA_VISIBLE_DEVICES="$gpu" python Latent-SFT/eval_hlr_compare.py \
            --hlr_checkpoint "$ckpt" \
            --problem "$EVAL_PROBLEM" \
            --problem_size "$EVAL_PROBLEM_SIZE" \
            --num_test "$EVAL_NUM_TEST" \
            --max_completion_length "$EVAL_MAX_LEN" \
            --temperature "$EVAL_TEMPERATURE" \
            --batch_size "$EVAL_BATCH_SIZE"
    }

    TODO_ARR=($TODO_TAGS)
    for (( base=0; base<${#TODO_ARR[@]}; base+=NGPU )); do
        pids=(); batch_tags=()
        for (( s=0; s<NGPU && base+s<${#TODO_ARR[@]}; s++ )); do
            idx=$((base+s)); tag="${TODO_ARR[$idx]}"; gpu="${GPU_POOL[$s]}"
            echo "  [batch] $tag → GPU $gpu"
            run_one_pa "$tag" "${CKPTS[$tag]}" "$gpu" > "${CKPTS[$tag]}/eval_driver.log" 2>&1 &
            pids+=($!); batch_tags+=("$tag")
        done
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}" || echo "  [batch] ⚠️ ${batch_tags[$i]} eval 非零退出 (详见 eval_driver.log)"
        done
    done
fi

# ── 结果校验 ──
DONE=""
for tag in "${ALL_TAGS[@]}"; do
    comp_json="${CKPTS[$tag]}/compare_eval/compare.json"
    if [ -f "$comp_json" ] && python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('hlr') else 1)" "$comp_json" 2>/dev/null; then
        DONE="$DONE $tag"
    else
        EVAL_FAILED="$EVAL_FAILED $tag(eval)"
    fi
done

# ── 汇总表 ──
echo ""
echo "============================================================"
echo "  HLR β Sweep Eval 汇总 ($(date '+%F %T'))"
echo "============================================================"
printf "  %-8s | %8s | %8s | %10s | %12s | %8s | %8s\n" \
    "β" "parse" "feas" "avg_dist" "CoT_tokens" "HLR_wall" "base_wall"
printf "  %-8s-|-%8s-|-%8s-|-%10s-|-%12s-|-%8s-|-%8s\n" \
    "--------" "--------" "--------" "----------" "------------" "--------" "--------"

for tag in "${ALL_TAGS[@]}"; do
    comp_json="${CKPTS[$tag]}/compare_eval/compare.json"
    if [ -f "$comp_json" ]; then
        python -c "
import json
with open('$comp_json') as f:
    d = json.load(f)
hlr = d.get('hlr', [{}])
b   = d.get('baseline', [{}])
if isinstance(hlr, list) and hlr: hlr = hlr[0]
if isinstance(b,   list) and b:   b   = b[0]
pr  = hlr.get('parse_rate', 0)
fr  = hlr.get('global_feas_rate', 0)
ad  = hlr.get('avg_instance_distance', 0)
cot_h = hlr.get('total_completion_tokens', 0)
cot_b = b.get('total_completion_tokens', 0)
wl_h = d.get('hlr_wall_seconds', 0)
wl_b = d.get('baseline_wall_seconds', 0)
print(f'  {tag:>7} | {pr:7.3f} | {fr:7.3f} | {ad:9.2f} | {cot_h:>6}/{cot_b:<6} | {wl_h:7.0f} | {wl_b:7.0f}')
"
    else
        echo "  ${tag:>7} | ❌ 无 compare.json"
    fi
done

echo ""
echo "  完成: ${DONE:-<无>}"
echo "  失败: ${EVAL_FAILED:-<无>}"
echo "============================================================"

if [ -z "$EVAL_FAILED" ]; then
    notify "✅ HLR sweep eval 全部完成" "完成:$DONE  $(date '+%F %T')"
else
    notify "⚠️ HLR sweep eval 部分失败" "完成:${DONE:-无}
失败:$EVAL_FAILED  $(date '+%F %T')"
    exit 1
fi
