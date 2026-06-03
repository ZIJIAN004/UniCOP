#!/bin/bash
# submit_grpo_cvrp20_v6_eval.sh
# ── 一个 job 串行: 训练 v6  →  merge LoRA  →  eval(BO1 + BO8 + wave) ──
#   阶段:
#     [1/3] 训练  : bash run_grpo_cvrp20_v6.sh (1 vLLM + 6 训练, 占满 7 卡)
#                   纯净 v6 (PRM 批级截尾标准化+sigmoid), LR=2e-5(对齐v5) EPOCHS=1
#     [2/3] merge : output_v6/cvrp_n20/final_model (LoRA adapter) → 同级 merged_model
#                   (镜像 evaluate.py:675-688 的合并逻辑, 提前显式做, 失败早暴露 + 校验权重非空)
#     [3/3] eval  : 只评 v6 RL 模型 (ONLY=RL), 跑 run_eval_matrix.sh 的 BO1 + BO8/wave
#                   训练阶段腾空的卡给 vLLM: TP=4 (GPU 0,1,2,3), 比默认 TP=1 快
#                   NUM_TEST=1000 + cvrp + size20 → evaluate.py 用 seed=9999 顺序生成,
#                   与 optimal 冻结集逐一对齐的同一批 1000 个实例 (gap 下游按下标对齐算)。
#
#   为什么训练+eval 同一个 job (而非两个 job + 依赖):
#     用户要求"跑完立刻 merge+eval"在一个 submit 里。本 job 独占整节点 7 卡, 训练完
#     卡自动释放回本 shell 的 SLURM 配额, eval 直接拿来用 (默认 TP=4)。eval 阶段用 4 卡,
#     另 3 卡空闲——可接受 (换取流程单一、无跨 job 依赖、无 QOS 二次排队)。
#
#   ⚠️ 训练那轴同时变了"信号(v6)"和"温和超参(LR/epoch)", gap 改善无法单独归因
#      (与 submit_grpo_cvrp20_v6.sh 同一口径)。
#
#   checkpoint 续跑: train.py 自动从 output_v6/cvrp_n20/checkpoint-* 最新一份恢复
#      (SAVE_STEPS=20)。本 job 内无 crash 自动重启 (--no-requeue); 崩了手动重投本
#      submit 即从最近 checkpoint 续训, 续完照样接 merge+eval。
#
#   提交: sbatch submit_grpo_cvrp20_v6_eval.sh

#SBATCH --qos=large
#SBATCH --gpus=7
#SBATCH --job-name=zijia_cvrp20_v6_te
#SBATCH --comment="zijianliu, v6 train+merge+eval(BO1/BO8/wave), do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v6_eval_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v6_eval_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 训练阶段覆盖项 (与 submit_grpo_cvrp20_v6.sh 完全一致) ──────────────
export BASE_MODEL_TYPE=qwen3_thinking   # Qwen3-4B SFT 产物作为 RL 起点
export LR="${LR:-2e-5}"                   # 对齐 v5 (上次 1e-6 致训练不足: grad_norm~0.03 下更新微乎其微, fully_feas 全程~0.3 不动)
export EPOCHS="${EPOCHS:-1}"              # 单 epoch
export SAVE_STEPS="${SAVE_STEPS:-20}"     # 每 20 step 存档 (供 checkpoint 续跑)
export NUM_TRAIN="${NUM_TRAIN:-1000}"     # 一个 epoch 的训练样本数 (run_grpo_v5 经 env 读); 与扫参目录命名一致
export PROC_ALPHA_V6="${PROC_ALPHA_V6:-200}"   # v6 PRM 段注入权重 (与扫参目录命名一致; train.py:180 env 覆盖)
# 输出目录带超参标注 → 不同 lr/epoch/proc_alpha/num_train 互不覆盖; 也避免误 resume 旧超参的 checkpoint
export OUTPUT_DIR_BASE="output_v6_lr${LR}_ep${EPOCHS}_pa${PROC_ALPHA_V6}_nt${NUM_TRAIN}"   # 如 output_v6_lr2e-5_ep1_pa400_nt500
# REWARD_SCHEME=v6 由 run_grpo_cvrp20_v6.sh 设

# ── eval 阶段参数 (可在 sbatch 前 export 覆盖) ─────────────────────────
EVAL_NUM_TEST="${EVAL_NUM_TEST:-1000}"   # 1000 = 与 optimal 对齐的冻结集; 勿改小否则不对齐
EVAL_GPU="${EVAL_GPU:-0,1,2,3}"          # 训练完腾空的卡给 vLLM; TP=4 用 GPU 0,1,2,3
EVAL_TP="${EVAL_TP:-4}"                   # tensor parallel=4 (Qwen3-4B-Thinking kv_heads 可整除 4)。
                                         # KV cache 切 4 份, 减抢占; 4B 模型提速主要来自 KV 余量。
EVAL_TEMP="${EVAL_TEMP:-0.6}"            # Qwen3-thinking 推荐 BO8 采样温度
EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.8}"      # 留余量给 CUDA graph + wave 的 POMO PRM

# ⚠️ 别在 conda activate 之前开 set -u: conda 的 activate.d/~cuda-nvcc_activate.sh 会引用
#    未设置的 NVCC_PREPEND_FLAGS, nounset 下直接 "unbound variable" 挂掉。原 v5/v6 submit
#    全程无 set -u 故无此坑; 这里先宽松 bootstrap, 激活完再开严格模式。
source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
set -uo pipefail

# 复用 v5 launcher 里的 Server酱 key, 给 merge/eval 阶段补通知 (训练阶段 run_grpo 已自带)
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=${1:0:100}" \
        --data-urlencode "desp=${2:0:500}" > /dev/null 2>&1 || true
}

# ── GPU 占用预检: 分到的卡被占 → exclude 本节点重投 (重投后从 checkpoint 续训) ──
#    无默认 exclude (用户决定: 不预排任何节点), 完全靠预检动态排除被占的坏节点。
export SUBMIT_SCRIPT="$(pwd)/submit_grpo_cvrp20_v6_eval.sh"
export BASE_EXCLUDE=""
source "$(pwd)/preflight_gpu.sh"
preflight_gpu_or_resubmit

OUT_DIR="${OUTPUT_DIR_BASE}/cvrp_n20"    # = $OUTPUT_DIR_BASE/{run_tag}_n{size} (train.py:436)
ADAPTER="$OUT_DIR/final_model"           # GRPO LoRA adapter
MERGED="$OUT_DIR/merged_model"           # 合并后的全量权重 (eval 吃这个)

# ====================================================================
echo "############## [1/3] 训练 v6 ##############  $(date '+%F %T')"
# ====================================================================
bash run_grpo_cvrp20_v6.sh
TRAIN_EC=$?
if [ "$TRAIN_EC" -ne 0 ]; then
    echo "[FATAL] 训练失败 (exit=$TRAIN_EC), 跳过 merge + eval"
    notify "❌ v6 train+eval: 训练失败" "exit=$TRAIN_EC  $(date '+%F %T')"
    exit "$TRAIN_EC"
fi
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "[FATAL] 训练号称成功但找不到 LoRA adapter: $ADAPTER/adapter_config.json"
    notify "❌ v6 train+eval: 找不到 adapter" "$ADAPTER  $(date '+%F %T')"
    exit 1
fi
echo "[1/3] ✓ 训练完成, adapter: $ADAPTER"

# ====================================================================
echo "############## [2/3] merge LoRA → merged_model ##############  $(date '+%F %T')"
# ====================================================================
# 显式 CPU 合并 (镜像 evaluate.py 的 auto-merge), 提前暴露失败 + 校验权重非空。
# 已存在 merged_model 则跳过 (幂等, 支持 eval 阶段失败后重投只重跑 eval)。
if [ -d "$MERGED" ] && [ -f "$MERGED/config.json" ]; then
    echo "[2/3] merged_model 已存在, 跳过合并: $MERGED"
else
    ADAPTER="$ADAPTER" MERGED="$MERGED" python - <<'PY'
import os, json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
adapter = os.environ["ADAPTER"]; merged = os.environ["MERGED"]
cfg  = json.load(open(os.path.join(adapter, "adapter_config.json")))
base = cfg["base_model_name_or_path"]
print(f"[merge] base = {base}")
print(f"[merge] adapter = {adapter}  ->  {merged}")
m  = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                          trust_remote_code=True, device_map="cpu")
pm = PeftModel.from_pretrained(m, adapter)
merged_m = pm.merge_and_unload()
merged_m.save_pretrained(merged)
AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
print("[merge] done")
PY
    if [ $? -ne 0 ]; then
        echo "[FATAL] merge 失败"
        notify "❌ v6 train+eval: merge 失败" "$(date '+%F %T')"
        exit 1
    fi
fi
# 校验: config + 至少一个非空 *.safetensors (防 ZeRO-3+LoRA 存出空壳, 见 CLAUDE.md 代码自审)
_W=$(find "$MERGED" -maxdepth 1 -name '*.safetensors' -size +0c 2>/dev/null | head -1)
if [ ! -f "$MERGED/config.json" ] || [ -z "$_W" ]; then
    echo "[FATAL] merged_model 校验失败 (缺 config.json 或权重为空): $MERGED"
    ls -la "$MERGED" 2>/dev/null || true
    notify "❌ v6 train+eval: merged 权重为空" "$MERGED  $(date '+%F %T')"
    exit 1
fi
echo "[2/3] ✓ merge 完成且权重非空: $MERGED"

# ====================================================================
echo "############## [3/3] eval BO1 + BO8 + wave (only v6 RL) ##############  $(date '+%F %T')"
# ====================================================================
notify "🚀 v6 eval 开始" "merged ok, 跑 BO1+BO8+wave on $EVAL_NUM_TEST 实例 (TP=$EVAL_TP)
$(date '+%F %T')"
RL_MODEL="$(pwd)/$MERGED" \
ONLY=RL \
NUM_TEST="$EVAL_NUM_TEST" \
TEMP="$EVAL_TEMP" \
GPU="$EVAL_GPU" \
TP="$EVAL_TP" \
GPU_MEM="$EVAL_GPU_MEM" \
MAXLEN_RL=6144 \
DO_BO1=1 DO_BO8WAVE=1 \
    bash run_eval_matrix.sh
EVAL_EC=$?

echo "============================================================"
if [ "$EVAL_EC" -eq 0 ]; then
    echo "  ✅ 全流程完成 (train→merge→eval)  $(date '+%F %T')"
    echo "  结果 JSON: eval_results_matrix/  (RL_BO1*, RL_BO8wave*)"
    echo "  eval 日志: eval_logs_matrix/RL_BO1.log, RL_BO8wave.log"
    notify "✅ v6 train+merge+eval 全部完成" "结果: eval_results_matrix/
$(date '+%F %T')"
else
    echo "  ⚠️ eval 阶段非零退出 (exit=$EVAL_EC), 详见 eval_logs_matrix/"
    notify "⚠️ v6 eval 非零退出" "exit=$EVAL_EC  $(date '+%F %T')"
fi
echo "============================================================"
exit "$EVAL_EC"
