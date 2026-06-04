#!/bin/bash
# submit_sweep_eval_bo1_v6.sh
# ── 给 v6 超参扫描 (train-only) 补: merge LoRA→SFT基座 + BO1 eval ×3 ──
#
#   背景: sweep_proc_alpha_v6.sh 转投的 submit_grpo_cvrp20_v6.sh 是 train-only,
#   各 pa 跑完只留下 LoRA adapter (output_v6_*pa*/cvrp_n20/final_model),
#   没 merge 也没 eval。本 submit 一个 job 串行补齐, 对每个扫参目录:
#     ① 定位 adapter (兼容 <dir>/cvrp_n20/final_model | <dir>/final_model | <dir> 三种布局)
#     ② merge 到 SFT 基座 (UniCOP-Distill/output_sft_qwen3_template_cvrp20/final_model)
#        → 同级 merged_model (镜像 submit_grpo_cvrp20_v6_eval.sh 的合并逻辑 + 权重非空校验)
#     ③ BO1 eval (greedy) 重复 3 遍 → 量化 vLLM 批调度/FP 噪声下 greedy 的运行间方差,
#        比较 pa 时看 3 遍均值, 差距小于运行间方差的结论不可信
#
#   结果 (每 pa × 每遍独立 tag, 不撞名):
#     eval_results_matrix/v6_lr2e-5_ep1_pa100_r1_RL_BO1.json  (r1/r2/r3)
#     eval_logs_matrix/v6_lr2e-5_ep1_pa100_r1_RL_BO1.log
#   幂等: merged_model 已存在跳 merge; 结果 JSON 有效跳 eval → 崩了/超时直接重投本 submit
#
#   提交 (集群登录节点, git pull 之后):
#     sbatch submit_sweep_eval_bo1_v6.sh
#   可覆盖 (sbatch --export=ALL,KEY=VAL,... ):
#     DIRS="output_v6_xxx output_v6_yyy"  默认 glob output_v6_*pa*
#     EVAL_NUM_TEST=1000 (默认; 与 optimal 冻结集 seed=9999 逐一对齐, 勿改小否则 gap 不对齐)
#     N_REPS=3  EVAL_GPU=0,1,2,3  EVAL_TP=4  EVAL_GPU_MEM=0.8

#SBATCH --qos=large
#SBATCH --gpus=4
#SBATCH --job-name=zijia_v6_sweep_bo1
#SBATCH --comment="zijianliu, v6 sweep merge+BO1 eval x3, do not cancel"
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
N_REPS="${N_REPS:-3}"            # BO1 每模型重复遍数 (greedy 的运行间噪声量化)
EVAL_GPU="${EVAL_GPU:-0,1,2,3}"
EVAL_TP="${EVAL_TP:-4}"          # Qwen3-4B-Thinking kv_heads 可整除 4
EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.8}"

if [ -z "$DIRS" ]; then echo "[FATAL] 没找到任何 output_v6_*pa* 目录"; exit 1; fi
if [ ! -f "$SFT_BASE/config.json" ]; then
    echo "[FATAL] SFT 基座不存在或缺 config.json: $SFT_BASE"; exit 1
fi
echo "扫参目录: $DIRS"
echo "SFT 基座: $SFT_BASE   NUM_TEST=$EVAL_NUM_TEST  REPS=$N_REPS  TP=$EVAL_TP"

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

FAILED=""; DONE=""
for d in $DIRS; do
    d="${d%/}"
    tag="${d##*/}"; tag="${tag#output_}"        # output_v6_lr2e-5_ep1_pa100 → v6_lr2e-5_ep1_pa100
    echo ""
    echo "############## [$tag] ##############  $(date '+%F %T')"

    # ① 定位 adapter: 三种布局按序探测
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

    # ② merge (幂等; 显式以 SFT 基座为底, 不依赖 adapter_config 里可能过期的绝对路径)
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

    # ③ BO1 eval × N_REPS (每遍独立 tag _r1/_r2/_r3; run_eval_matrix 幂等跳过已完成遍)
    ok=1
    for rep in $(seq 1 "$N_REPS"); do
        echo "[$tag] >>> BO1 rep $rep/$N_REPS  $(date '+%F %T')"
        RUN_PREFIX="${tag}_r${rep}_" \
        RL_MODEL="$MERGED_ABS" \
        ONLY=RL DO_BO1=1 DO_BO8WAVE=0 \
        NUM_TEST="$EVAL_NUM_TEST" \
        GPU="$EVAL_GPU" TP="$EVAL_TP" GPU_MEM="$EVAL_GPU_MEM" \
        MAXLEN_RL=6144 \
            bash run_eval_matrix.sh
        out_json="eval_results_matrix/${tag}_r${rep}_RL_BO1.json"
        if [ ! -f "$out_json" ] || ! python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('results') and d['results'][0].get('n_eval',0)>0 else 1)" "$out_json" 2>/dev/null; then
            echo "[$tag] ❌ rep $rep 结果 JSON 无效: $out_json (详见 eval_logs_matrix/${tag}_r${rep}_RL_BO1.log)"
            ok=0
        fi
    done
    if [ "$ok" = "1" ]; then DONE="$DONE $tag"; else FAILED="$FAILED $tag(eval)"; fi
done

echo ""
echo "============================================================"
echo "  完成: ${DONE:-<无>}"
echo "  失败: ${FAILED:-<无>}"
echo "  结果: eval_results_matrix/v6_*_r{1..$N_REPS}_RL_BO1.json   $(date '+%F %T')"
echo "============================================================"
if [ -z "$FAILED" ]; then
    notify "✅ v6 sweep merge+BO1×$N_REPS 全部完成" "完成:$DONE
结果: eval_results_matrix/  $(date '+%F %T')"
else
    notify "⚠️ v6 sweep BO1 部分失败" "完成:${DONE:-无}
失败:$FAILED  $(date '+%F %T')"
    exit 1
fi
