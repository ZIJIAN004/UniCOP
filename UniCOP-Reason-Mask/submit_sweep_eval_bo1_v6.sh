#!/bin/bash
# submit_sweep_eval_bo1_v6.sh
# ── 给 v6 超参扫描 (train-only) 补: merge LoRA→SFT基座 + BO1 eval (4 pa 并行, 每 pa 一张卡) ──
#
#   背景: sweep_proc_alpha_v6.sh 转投的 submit_grpo_cvrp20_v6.sh 是 train-only,
#   各 pa 跑完只留下 LoRA adapter (output_v6_*pa*/cvrp_n20/final_model),
#   没 merge 也没 eval。本 submit 一个 job 补齐:
#     [1/2] merge (串行, CPU): 对每个扫参目录定位 adapter (兼容 <dir>/cvrp_n20/final_model |
#           <dir>/final_model | <dir> 三种布局), 合并到 SFT 基座
#           (UniCOP-Distill/output_sft_qwen3_template_cvrp20/final_model) → 同级 merged_model
#           + 权重非空校验。CPU 合并一个峰值 ~20GB 内存, 串行避免内存挤兑 (反正每个只要几分钟)。
#     [2/2] BO1 eval (并行): 每个 pa 绑一张卡各起一个 vLLM (TP=1), 4 卡同时跑 4 个 pa;
#           目录数 > 卡数时按波次分批 (每批 = 卡数个, wait 后再下一批)。
#           BO1 = greedy (evaluate.py:855 num_samples=1 强制 temp=0), 每模型 1 遍即可;
#           要量化 vLLM 运行间噪声可 N_REPS=3 重投, 已完成的 r1 幂等跳过, 只补 r2/r3。
#
#   结果 (每 pa 独立 tag, 不撞名):
#     eval_results_matrix/v6_lr2e-5_ep1_pa100_r1_RL_BO1.json
#     eval_logs_matrix/v6_lr2e-5_ep1_pa100_r1_RL_BO1.log   (evaluate.py 日志)
#     eval_logs_matrix/driver_<tag>.log                     (该 pa 的 run_eval_matrix 驱动日志)
#   幂等: merged_model 已存在跳 merge; 结果 JSON 有效跳 eval → 崩了/超时直接重投本 submit
#
#   提交 (集群登录节点, git pull 之后):
#     sbatch submit_sweep_eval_bo1_v6.sh
#   可覆盖 (sbatch --export=ALL,KEY=VAL,... ):
#     DIRS="output_v6_xxx output_v6_yyy"  默认 glob output_v6_*pa*
#     EVAL_NUM_TEST=1000 (默认; 与 optimal 冻结集 seed=9999 逐一对齐, 勿改小否则 gap 不对齐)
#     N_REPS=1  EVAL_GPUS=0,1,2,3 (并行卡池, 每 pa 占一张)  EVAL_GPU_MEM=0.8

#SBATCH --qos=large
#SBATCH --gpus=4
#SBATCH --job-name=zijia_v6_sweep_bo1
#SBATCH --comment="zijianliu, v6 sweep merge+BO1 eval (4 pa parallel), do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/sweep_eval_bo1_v6_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/sweep_eval_bo1_v6_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
export BASE_MODEL_TYPE=qwen3_thinking

# ⚠️ conda activate 前别开 set -u (activate.d 引用未设变量会挂, 同 v6_eval submit 的坑)
source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
set -uo pipefail
source ../paths.sh   # 注入 DISTILL_DIR 等 (本机 = astar-zhuoyi)

# ── 参数 ──────────────────────────────────────────────────────────
DIRS="${DIRS:-$(ls -d output_v6_*pa* 2>/dev/null | tr '\n' ' ')}"
SFT_BASE="${SFT_BASE:-$DISTILL_DIR/output_sft_qwen3_template_cvrp20/final_model}"
EVAL_NUM_TEST="${EVAL_NUM_TEST:-1000}"
N_REPS="${N_REPS:-1}"               # BO1 greedy 1 遍即可; 量化 vLLM 噪声再 N_REPS=3 重投(幂等补跑)
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"   # 并行卡池: 每个 pa 占一张卡 (TP=1), 同时跑卡数个 pa
EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.8}"

if [ -z "$DIRS" ]; then echo "[FATAL] 没找到任何 output_v6_*pa* 目录"; exit 1; fi
if [ ! -f "$SFT_BASE/config.json" ]; then
    echo "[FATAL] SFT 基座不存在或缺 config.json: $SFT_BASE"; exit 1
fi
IFS=',' read -ra GPU_POOL <<< "$EVAL_GPUS"
echo "扫参目录: $DIRS"
echo "SFT 基座: $SFT_BASE   NUM_TEST=$EVAL_NUM_TEST  REPS=$N_REPS  卡池=[${EVAL_GPUS}] (每 pa 1 卡并行)"

# Server酱通知 (复用 v6_eval submit 的 key 与写法)
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=${1:0:100}" \
        --data-urlencode "desp=${2:0:500}" > /dev/null 2>&1 || true
}

# ── GPU 占用预检: 分到的卡被占 → exclude 本节点重投 (幂等, 重投跳过已完成项) ──
export SUBMIT_SCRIPT="$(pwd)/submit_sweep_eval_bo1_v6.sh"
export BASE_EXCLUDE=""
source "$(pwd)/preflight_gpu.sh"
preflight_gpu_or_resubmit

# ====================================================================
echo "############## [1/2] merge (串行, CPU) ##############  $(date '+%F %T')"
# ====================================================================
TAGS=(); MERGEDS=(); FAILED=""
for d in $DIRS; do
    d="${d%/}"
    tag="${d##*/}"; tag="${tag#output_}"        # output_v6_lr2e-5_ep1_pa100 → v6_lr2e-5_ep1_pa100
    echo ""
    echo "── [$tag] ──  $(date '+%F %T')"

    # 定位 adapter: 三种布局按序探测
    ADAPTER=""
    for cand in "$d/cvrp_n20/final_model" "$d/final_model" "$d"; do
        if [ -f "$cand/adapter_config.json" ]; then ADAPTER="$cand"; break; fi
    done
    if [ -z "$ADAPTER" ]; then
        echo "[$tag] ❌ 找不到 adapter_config.json (探测过 $d{,/final_model,/cvrp_n20/final_model}), 跳过"
        FAILED="$FAILED $tag(no-adapter)"; continue
    fi
    # merged 落点: adapter 在 final_model 里 → 同级; adapter 直接在扫参目录根 → 目录内
    # (不能无脑 dirname: ADAPTER=$d 时 dirname 是所有 pa 共享的上级目录, 会互相覆盖)
    if [ "$ADAPTER" = "$d" ]; then MERGED="$d/merged_model"; else MERGED="$(dirname "$ADAPTER")/merged_model"; fi
    case "$MERGED" in /*) MERGED_ABS="$MERGED";; *) MERGED_ABS="$PWD/$MERGED";; esac   # DIRS 传绝对路径也兼容
    echo "[$tag] adapter: $ADAPTER  →  merged: $MERGED"

    # merge (幂等; 显式以 SFT 基座为底, 不依赖 adapter_config 里可能过期的绝对路径)
    if [ -d "$MERGED" ] && [ -f "$MERGED/config.json" ]; then
        echo "[$tag] merged_model 已存在, 跳过合并"
    else
        ADAPTER="$ADAPTER" MERGED="$MERGED" SFT_BASE="$SFT_BASE" python - <<'PY'
import os, json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
adapter = os.environ["ADAPTER"]; merged = os.environ["MERGED"]; base = os.environ["SFT_BASE"]
rec = json.load(open(os.path.join(adapter, "adapter_config.json"))).get("base_model_name_or_path", "")
if rec and os.path.realpath(rec) != os.path.realpath(base):
    print(f"[merge][warn] adapter_config 记录的 base 与指定 SFT 基座不同, 按指定基座合并:\n"
          f"  记录: {rec}\n  指定: {base}")
print(f"[merge] base = {base}")
print(f"[merge] adapter = {adapter}  ->  {merged}")
m  = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                          trust_remote_code=True, device_map="cpu")
pm = PeftModel.from_pretrained(m, adapter)
pm.merge_and_unload().save_pretrained(merged)
try:
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
except Exception:
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)   # adapter 目录没存 tokenizer 时回退
print("[merge] done")
PY
        if [ $? -ne 0 ]; then
            echo "[$tag] ❌ merge 失败, 跳过 eval"
            notify "❌ v6 sweep BO1: $tag merge 失败" "$(date '+%F %T')"
            FAILED="$FAILED $tag(merge)"; continue
        fi
    fi
    # 校验: config + 至少一个非空 *.safetensors (防 LoRA 存出空壳, 见 CLAUDE.md 代码自审)
    _W=$(find "$MERGED" -maxdepth 1 -name '*.safetensors' -size +0c 2>/dev/null | head -1)
    if [ ! -f "$MERGED/config.json" ] || [ -z "$_W" ]; then
        echo "[$tag] ❌ merged_model 校验失败 (缺 config.json 或权重为空)"
        ls -la "$MERGED" 2>/dev/null || true
        notify "❌ v6 sweep BO1: $tag merged 权重为空" "$MERGED  $(date '+%F %T')"
        FAILED="$FAILED $tag(empty-weights)"; continue
    fi
    TAGS+=("$tag"); MERGEDS+=("$MERGED_ABS")
done

if [ "${#TAGS[@]}" -eq 0 ]; then
    echo "[FATAL] 没有任何 pa merge 成功, 退出。失败:$FAILED"
    notify "❌ v6 sweep BO1: 全部 merge 失败" "$FAILED  $(date '+%F %T')"
    exit 1
fi

# ====================================================================
echo ""
echo "############## [2/2] BO1 eval (并行, 每 pa 1 卡) ##############  $(date '+%F %T')"
# ====================================================================
# 单个 pa 的 eval: 在指定卡上串行跑 N_REPS 遍 (默认 1)
run_one_pa() {  # run_one_pa <tag> <merged_abs> <gpu>
    local tag="$1" merged="$2" gpu="$3" rep
    for rep in $(seq 1 "$N_REPS"); do
        echo "[$tag][gpu$gpu] >>> BO1 rep $rep/$N_REPS  $(date '+%F %T')"
        RUN_PREFIX="${tag}_r${rep}_" \
        RL_MODEL="$merged" \
        ONLY=RL DO_BO1=1 DO_BO8WAVE=0 \
        NUM_TEST="$EVAL_NUM_TEST" \
        GPU="$gpu" TP=1 GPU_MEM="$EVAL_GPU_MEM" \
        MAXLEN_RL=6144 \
            bash run_eval_matrix.sh
    done
}

NGPU="${#GPU_POOL[@]}"
EVAL_FAILED=""
mkdir -p eval_logs_matrix eval_results_matrix   # driver log 重定向先于 run_eval_matrix 的 mkdir 发生
for (( base=0; base<${#TAGS[@]}; base+=NGPU )); do      # 波次: 每批最多 NGPU 个 pa 同时跑
    pids=(); batch_tags=()
    for (( s=0; s<NGPU && base+s<${#TAGS[@]}; s++ )); do
        idx=$((base+s)); tag="${TAGS[$idx]}"; gpu="${GPU_POOL[$s]}"
        echo "[batch] $tag → GPU $gpu  (driver log: eval_logs_matrix/driver_${tag}.log)"
        run_one_pa "$tag" "${MERGEDS[$idx]}" "$gpu" > "eval_logs_matrix/driver_${tag}.log" 2>&1 &
        pids+=($!); batch_tags+=("$tag")
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || echo "[batch] ⚠️ ${batch_tags[$i]} driver 非零退出 (详见 driver log)"
    done
done

# ── 结果校验: 每 pa × 每遍的 JSON 必须有效 ──
DONE=""
for tag in "${TAGS[@]}"; do
    ok=1
    for rep in $(seq 1 "$N_REPS"); do
        out_json="eval_results_matrix/${tag}_r${rep}_RL_BO1.json"
        if [ ! -f "$out_json" ] || ! python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('results') and d['results'][0].get('n_eval',0)>0 else 1)" "$out_json" 2>/dev/null; then
            echo "[$tag] ❌ rep $rep 结果 JSON 无效: $out_json (详见 eval_logs_matrix/${tag}_r${rep}_RL_BO1.log)"
            ok=0
        fi
    done
    if [ "$ok" = "1" ]; then DONE="$DONE $tag"; else EVAL_FAILED="$EVAL_FAILED $tag(eval)"; fi
done
FAILED="$FAILED$EVAL_FAILED"

echo ""
echo "============================================================"
echo "  完成: ${DONE:-<无>}"
echo "  失败: ${FAILED:-<无>}"
echo "  结果: eval_results_matrix/v6_*_r{1..$N_REPS}_RL_BO1.json   $(date '+%F %T')"
echo "============================================================"
if [ -z "$FAILED" ]; then
    notify "✅ v6 sweep merge+BO1 全部完成" "完成:$DONE
结果: eval_results_matrix/  $(date '+%F %T')"
else
    notify "⚠️ v6 sweep BO1 部分失败" "完成:${DONE:-无}
失败:$FAILED  $(date '+%F %T')"
    exit 1
fi
