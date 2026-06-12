#!/bin/bash
# submit_grpo_foarl_cvrp.sh — FOARL CVRP Stage-2 GRPO RL (zhuoyi SLURM)
#   基座: Stage-1 SFT 的 **merged** 模型 (LoRA 须先 merge 回基座)
#   奖励: foarl_reward_cvrp 规则奖励 R^P = R_f(可行性) + R_o(最优性), 无 reward model/PRM
#   栈:   accelerate ZeRO-3 多卡 + TRL GRPOTrainer + LoRA (与 SFT 同规格)
#
#   ⚠️ 必填: MODEL = merged SFT 模型目录 (脚本里留空, 提交前 export 或改下面默认值)。
#        LoRA 合并示例 (SFT 产物是 adapter):
#          python -c "from peft import AutoPeftModelForCausalLM; \
#            m=AutoPeftModelForCausalLM.from_pretrained('<SFT_OUT>/final_model'); \
#            m.merge_and_unload().save_pretrained('<MERGED_DIR>')"
#        再把 tokenizer 一并拷到 <MERGED_DIR>。
#   sanity 先跑: SANITY=1 MODEL=<merged> sbatch submit_grpo_foarl_cvrp.sh  (只取 64 条 + 看 reward 探针)
#   提交:        MODEL=<merged> sbatch submit_grpo_foarl_cvrp.sh

#SBATCH --qos=normal
#SBATCH --gpus=4
#SBATCH --job-name=zijia_foarl_grpo_cvrp
#SBATCH --comment="zijianliu, FOARL CVRP stage2 GRPO RL, do not cancel"
#SBATCH --exclude=canele1
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/FOARL/foarl_grpo_cvrp_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/FOARL/foarl_grpo_cvrp_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 可覆盖参数 ─────────────────────────────────────────────────────────
# ⚠️ MODEL 留空: 必须指向 Stage-1 SFT 的 merged 模型目录, 提交前 export MODEL=<路径>
MODEL="${MODEL:-}"
NUM_GPUS="${NUM_GPUS:-4}"
DATA="${DATA:-data/foarl_cvrp20.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_grpo_foarl_cvrp20}"
# GRPO 超参 (可 export 覆盖)
S="${S:-8}"                       # 组大小 num_generations
LR="${LR:-1e-6}"
BETA="${BETA:-0.04}"              # KL 系数
EPS="${EPS:-0.2}"
EPS_HIGH="${EPS_HIGH:-0.28}"
PDTB="${PDTB:-8}"                 # per-device prompt 数; 全局批=PDTB×NUM_GPUS×GA 须被 S 整除
GA="${GA:-4}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
# FOARL 奖励权重 (默认沿用既有配置, 对齐论文附录 A 时改这里)
ALPHA="${ALPHA:-0.5}"
W_PARSE="${W_PARSE:-0.2}"
W_COV="${W_COV:-0.3}"
W_CAP="${W_CAP:-0.3}"
W_FMT="${W_FMT:-0.2}"
SANITY="${SANITY:-0}"             # 1 = 只取 64 条做 sanity

# ⚠️ 先 conda activate 再 set -u (cuda-nvcc_activate.sh 引用未设的 NVCC_PREPEND_FLAGS,
#    nounset 下会 unbound variable 挂掉, 见主机配置库/vLLM踩坑)
# ⚠️ 不能靠 source ~/.bashrc 激活: 非交互 sbatch shell 下 .bashrc 会提前 return,
#    conda hook 不生效 → conda/accelerate 全 command not found。直接 source miniforge 的
#    conda.sh (zhuoyi 是 miniforge3, 见 address.md/paths.sh)。
__CONDA_SH="/homes/zhuoyi/miniforge3/etc/profile.d/conda.sh"
[ -f "$__CONDA_SH" ] || { echo "[FATAL] 找不到 conda.sh: $__CONDA_SH (确认 zhuoyi conda 安装路径)"; exit 1; }
source "$__CONDA_SH"
conda activate /homes/zhuoyi/miniforge3/envs/unicop
command -v accelerate >/dev/null 2>&1 || { echo "[FATAL] accelerate 不在 PATH, unicop 环境未激活成功"; exit 1; }
cd /homes/zhuoyi/zijianliu/UniCOP/FOARL
set -uo pipefail

# zhuoyi 多卡须禁 P2P/SHM 否则 ZeRO-3 init hang (reference_zhuoyi_nccl_topology)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

echo "############## FOARL CVRP GRPO RL ##############  $(date '+%F %T')"
echo "  MODEL=$MODEL | NUM_GPUS=$NUM_GPUS | DATA=$DATA | OUT=$OUTPUT_DIR"
echo "  GRPO: S=$S LR=$LR BETA=$BETA EPS=[$EPS,$EPS_HIGH] PDTB=$PDTB GA=$GA EPOCHS=$EPOCHS"
echo "  奖励: ALPHA=$ALPHA W(parse=$W_PARSE cov=$W_COV cap=$W_CAP fmt=$W_FMT) | SANITY=$SANITY"

if [ -z "$MODEL" ]; then
    echo "[FATAL] MODEL 为空。请 export MODEL=<merged SFT 模型目录> 后重投。"
    echo "        (Stage-1 SFT 产物是 LoRA adapter, 须先 merge_and_unload 再喂 RL)"
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "[FATAL] 基座不存在: $MODEL"
    exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "[FATAL] 数据不存在: $DATA (先跑 build_foarl_cvrp_data.py 生成, 须含 instance 字段)"
    exit 1
fi

SANITY_FLAG=""
LOG_STEPS=5
RESUME_FLAG="--resume_from_checkpoint auto"
if [ "$SANITY" = "1" ]; then
    SANITY_FLAG="--max_samples 64"
    LOG_STEPS=1
    RESUME_FLAG=""
    OUTPUT_DIR="${OUTPUT_DIR}_sanity"
    echo "[sanity] 输出改到独立目录: $OUTPUT_DIR (不续 ckpt)"
fi

# 全局批可整除性自检 (train_grpo_foarl.py 内部也会 assert, 这里提前拦截省排队时间)
GLOBAL_BATCH=$(( PDTB * NUM_GPUS * GA ))
if [ $(( GLOBAL_BATCH % S )) -ne 0 ]; then
    echo "[FATAL] 全局批 $GLOBAL_BATCH (=PDTB$PDTB×GPU$NUM_GPUS×GA$GA) 不能被 S=$S 整除, 调参后重投。"
    exit 1
fi
echo "[批检查] 全局生成批=$GLOBAL_BATCH, S=$S → OK"

accelerate launch --num_processes "$NUM_GPUS" --main_process_port 29611 \
    train_grpo_foarl.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank 64 --lora_alpha 128 \
    --num_generations "$S" \
    --lr "$LR" --beta "$BETA" --epsilon "$EPS" --epsilon_high "$EPS_HIGH" \
    --batch_size "$PDTB" --grad_accum "$GA" \
    --epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
    --max_prompt_length 1536 --max_completion_length 1024 \
    --alpha "$ALPHA" \
    --omega_parse "$W_PARSE" --omega_coverage "$W_COV" \
    --omega_capacity "$W_CAP" --omega_format "$W_FMT" \
    --zero_stage 3 --gradient_checkpointing \
    --save_steps 200 --logging_steps "$LOG_STEPS" \
    $RESUME_FLAG \
    $SANITY_FLAG
EC=$?

echo "============================================================"
if [ "$EC" -eq 0 ]; then
    echo "  ✅ FOARL GRPO 完成  $(date '+%F %T')  →  $OUTPUT_DIR/final_model"
    echo "  下一步: merge LoRA → eval (可行率/gap) 对比 SFT-only baseline"
else
    echo "  ⚠️ GRPO 非零退出 (exit=$EC)"
fi
echo "============================================================"
exit "$EC"
