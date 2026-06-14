#!/bin/bash
# submit_eval_bo1_foarl_compare.sh
# ── 对 FOARL 两个产物各做一遍 BO1 eval, 固定测试集, SFT vs RL 同集可比 ──
#   待评模型:
#     foarl_SFT_BO1 = output_sft_foarl_cvrp20/merged        (Stage-1 SFT only)
#     foarl_RL_BO1  = output_grpo_foarl_cvrp20/merged_model (Stage-2 SFT+GRPO-RL, 完整 FOARL)
#   测试集固定: evaluate.py 硬编码 seed=9999, num_test=1000 → 与 Mask RL_BO1 / optimal
#     冻结集逐一对齐的同一批 1000 实例。SFT 与 RL 走同一 evaluate.py 同参数 → 同一批, 可比。
#
# ★ 为什么不沿用 run_grpo_foarl_cvrp.sh 里那段直调 evaluate.py 的 eval (它崩了):
#   那段 EVAL_TP 默认=4。但 vLLM 0.7.3 无 Qwen3 原生实现 → 回退 Transformers backend,
#   该 backend 在 tensor_parallel>1 的多 worker 下生成期崩溃
#   (日志: "Worker VllmWorkerProcess died, exit code: -15"; 对口 issue #17630 / #39774)。
#   Mask 的 run_eval_matrix.sh 默认就是 TP=1 (单卡不起 worker 子进程, 整类 worker 崩溃消失),
#   v6_eval 里的 TP=4 只是显式提速覆盖。本脚本照原版默认用 TP=1。
#   Qwen3-4B merged ~1.9GB, 单卡 24GB KV cache 实测 176× 并发, BO1 贪心单卡足够快;
#   TP 仅推理并行度, 不改贪心 token 选择 → 对结果零影响, 受控对比不受损。
#
# 学自: UniCOP-Reason-Mask/submit_sweep_eval_bo1_v6.sh + run_eval_matrix.sh
#   (TP=1 + spawn + 确定性 run_tag + n_eval==NUM_TEST 幂等 + 日志重定向 + exit code 立刻接住)
#
# 提交 (zhuoyi 登录节点, git pull 后): sbatch submit_eval_bo1_foarl_compare.sh
# 可覆盖 (sbatch --export=ALL,KEY=VAL):
#   EVAL_NUM_TEST=1000 (与冻结集对齐, 勿改小)  EVAL_GPU_MEM=0.8  EVAL_MAXLEN=1024
#   ENFORCE_EAGER=0 (=1 时加 --enforce_eager, 仅当 CUDA graph capture 报错才需要)

#SBATCH --qos=express
#SBATCH --gpus=1
#SBATCH --job-name=zijia_foarl_eval_bo1
#SBATCH --comment="zijianliu, FOARL SFT vs RL BO1 eval (fixed seed=9999 testset), do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/FOARL/eval_bo1_foarl_compare_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/FOARL/eval_bo1_foarl_compare_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ⚠️ conda 激活: 必须直接 source miniforge 的 conda.sh, 不能靠 source ~/.bashrc + conda hook
#    (非交互 sbatch shell 下 .bashrc 提前 return → conda hook 不生效 → conda: command not found,
#     与 submit_grpo_foarl_cvrp.sh 同款; 实测 09590 job 就栽在 .bashrc 方式上)。
#    在 conda activate 之前别开 set -u (activate.d 引用未设变量, nounset 下挂)。
__CONDA_SH="/homes/zhuoyi/miniforge3/etc/profile.d/conda.sh"
[ -f "$__CONDA_SH" ] || { echo "[FATAL] 找不到 conda.sh: $__CONDA_SH"; exit 1; }
source "$__CONDA_SH"
conda activate /homes/zhuoyi/miniforge3/envs/unicop
# fail-fast: 激活没成功就立刻退出, 不再静默"假完成" (上次 conda not found 但脚本仍跑到结束)
command -v python >/dev/null 2>&1 || { echo "[FATAL] python 不在 PATH, conda 未激活"; exit 1; }
python -c "import vllm" 2>/dev/null || { echo "[FATAL] import vllm 失败, unicop 环境未正确激活"; exit 1; }
FOARL_DIR=/homes/zhuoyi/zijianliu/UniCOP/FOARL
cd "$FOARL_DIR"
set -uo pipefail
source ../paths.sh   # 注入 MASK_DIR (evaluate.py 在那, import 其 config/problems)

# tp=1 不起 worker 子进程, spawn 对 tp=1 无副作用; 保留与 run_eval_matrix.sh 一致 (tp>1 时必需)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── 参数 ──────────────────────────────────────────────────────────────
EVAL_NUM_TEST="${EVAL_NUM_TEST:-1000}"
EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.8}"
EVAL_MAXLEN="${EVAL_MAXLEN:-1024}"          # FOARL 答案短(~百 tok), 1024 足够 (与 RL pipeline 一致)
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"         # 1 = 加 --enforce_eager (CUDA graph capture 报错时回退)
SAVE_DIR="$FOARL_DIR/eval_bo1_compare"
LOG_DIR="$SAVE_DIR/logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

SFT_MERGED="$FOARL_DIR/output_sft_foarl_cvrp20/merged"
RL_MERGED="$FOARL_DIR/output_grpo_foarl_cvrp20/merged_model"

EAGER_FLAG=""; [ "$ENFORCE_EAGER" = "1" ] && EAGER_FLAG="--enforce_eager"

SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=${1:0:100}" --data-urlencode "desp=${2:0:500}" >/dev/null 2>&1 || true
}

# run <tag> <merged_dir> : BO1 eval (greedy, num_samples=1 强制 temp=0), 幂等跳过已完成
run() {
    local tag="$1" model="$2"
    local out_json="$SAVE_DIR/${tag}.json"
    # 模型存在性 + 权重非空校验 (防 ZeRO-3+LoRA 空壳, 见 CLAUDE.md 代码自审)
    if [ ! -d "$model" ] || [ ! -f "$model/config.json" ]; then
        echo "[$tag] ❌ merged 模型不存在或缺 config.json: $model"; return 2
    fi
    local _w; _w=$(find "$model" -maxdepth 1 -name '*.safetensors' -size +0c 2>/dev/null | head -1)
    [ -n "$_w" ] || { echo "[$tag] ❌ 权重为空 (无非空 safetensors): $model"; return 2; }
    # 幂等: 结果 JSON 已存在且 n_eval == 当前 NUM_TEST → 跳过 (防把旧 smoke 结果当正式)
    if [ -f "$out_json" ] && python -c "import json,sys;d=json.load(open(sys.argv[1]));r=(d.get('results') or [{}])[0];sys.exit(0 if r.get('n_eval',0)==int(sys.argv[2]) else 1)" "$out_json" "$EVAL_NUM_TEST" 2>/dev/null; then
        echo "[$tag] ⏭️  已完成 (n_eval=$EVAL_NUM_TEST 一致), 跳过: $out_json"; return 0
    fi
    echo "[$tag] >>> BO1 eval (greedy) | model=$model  $(date '+%F %T')"
    cd "$MASK_DIR"
    python evaluate.py \
        --backend vllm --model_path "$model" --tp_size 1 \
        --vllm_gpu_mem_util "$EVAL_GPU_MEM" \
        --problem cvrp --problem_size 20 --num_test "$EVAL_NUM_TEST" \
        --prompt_mode foarl --model_type instruct \
        --num_samples 1 --max_completion_length "$EVAL_MAXLEN" \
        --save_dir "$SAVE_DIR" --run_tag "$tag" $EAGER_FLAG \
        > "$LOG_DIR/${tag}.log" 2>&1
    local ec=$?   # 必须立刻接住: 下一行 $(date) 命令替换会把 $? 重置为 0
    cd "$FOARL_DIR"
    echo "[$tag] <<< BO1 eval (exit $ec)  log: $LOG_DIR/${tag}.log"
    return $ec
}

echo "############## FOARL SFT vs RL BO1 eval (固定 seed=9999 × $EVAL_NUM_TEST) ##############  $(date '+%F %T')"
echo "  SFT merged: $SFT_MERGED"
echo "  RL  merged: $RL_MERGED"
echo "  TP=1 (避 Qwen3 fallback 多 worker 崩) | maxlen=$EVAL_MAXLEN | gpu_mem=$EVAL_GPU_MEM | eager=$ENFORCE_EAGER"

FAILED=""
run "foarl_SFT_BO1" "$SFT_MERGED"; ec=$?; [ "$ec" -eq 0 ] || FAILED="$FAILED foarl_SFT_BO1(ec=$ec)"
run "foarl_RL_BO1"  "$RL_MERGED";  ec=$?; [ "$ec" -eq 0 ] || FAILED="$FAILED foarl_RL_BO1(ec=$ec)"

echo "============================================================"
if [ -z "$FAILED" ]; then
    echo "  ✅ FOARL SFT/RL BO1 eval 全部完成  $(date '+%F %T')"
    echo "  结果: $SAVE_DIR/foarl_SFT_BO1.json , $SAVE_DIR/foarl_RL_BO1.json"
    echo "  对比: 同 seed=9999 × $EVAL_NUM_TEST 测试集, SFT vs RL 直接比 feas / avg_best_dist / gap"
    echo "  Quick view:"
    echo "    for t in foarl_SFT_BO1 foarl_RL_BO1; do jq '.results[0]|{tag:\"'\$t'\",feas:.global_feasibility_rate,dist:.avg_best_dist,parse:.format_match_rate}' $SAVE_DIR/\$t.json; done"
    notify "✅ FOARL SFT/RL BO1 eval 完成" "$SAVE_DIR  $(date '+%F %T')"
else
    echo "  ⚠️ 部分失败:$FAILED  (详见 $LOG_DIR/*.log)"
    notify "⚠️ FOARL BO1 eval 部分失败" "$FAILED  $(date '+%F %T')"
    exit 1
fi
echo "============================================================"
