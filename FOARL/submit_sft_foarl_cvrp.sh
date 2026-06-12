#!/bin/bash
# submit_sft_foarl_cvrp.sh — FOARL CVRP Stage-1 SFT (zhuoyi SLURM)
#   基座: Qwen3-4B-Instruct-2507 (非思维) | 数据: FOARL 原版内容 (无 think)
#   超参: 对齐 UniCOP-Distill 思维臂 (LoRA r64/α128, lr2e-5, 3ep, bs1·ga8, maxlen4864) → 消融只差"有无 think"
#   流程: (缺数据则先建) build_foarl_cvrp_data.py → accelerate ZeRO-3 多卡 train_sft_foarl.py
#
#   ⚠️ 需先在集群放好 Qwen3-4B-Instruct-2507 (paths.sh 现仅有 Thinking-2507):
#        默认找 /homes/zhuoyi/zijianliu/models/Qwen3-4B-Instruct-2507
#        没有就 export MODEL=<路径> 或先下载到该目录。
#   sanity 先跑: SANITY=1 sbatch submit_sft_foarl_cvrp.sh  (只取 256 条 + 看探针)
#   提交: sbatch submit_sft_foarl_cvrp.sh

#SBATCH --qos=normal
#SBATCH --gpus=4
#SBATCH --job-name=zijia_foarl_sft_cvrp
#SBATCH --comment="zijianliu, FOARL CVRP stage1 SFT, do not cancel"
#SBATCH --exclude=canele1
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/FOARL/foarl_sft_cvrp_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/FOARL/foarl_sft_cvrp_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 可覆盖参数 ─────────────────────────────────────────────────────────
MODEL="${MODEL:-/homes/zhuoyi/zijianliu/models/Qwen3-4B-Instruct-2507}"
NUM_GPUS="${NUM_GPUS:-4}"
DATA="${DATA:-data/foarl_cvrp20.jsonl}"
SRC="${SRC:-../UniCOP-Distill/data/chains_template_cvrp20.jsonl}"   # 与思维臂同源(取 </think> 后答案)
OUTPUT_DIR="${OUTPUT_DIR:-./output_sft_foarl_cvrp20}"
SANITY="${SANITY:-0}"   # 1 = 只取 256 条做 sanity

# ⚠️ 先 conda activate 再 set -u: conda activate.d/~cuda-nvcc_activate.sh 引用未设的
#    NVCC_PREPEND_FLAGS, nounset 下会 "unbound variable" 挂掉 (见 vLLM踩坑/主机配置库)。
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

echo "############## FOARL CVRP SFT ##############  $(date '+%F %T')"
echo "  MODEL=$MODEL | NUM_GPUS=$NUM_GPUS | DATA=$DATA | OUT=$OUTPUT_DIR | SANITY=$SANITY"

if [ ! -d "$MODEL" ]; then
    echo "[FATAL] 基座不存在: $MODEL"
    echo "  请先把 Qwen3-4B-Instruct-2507 下到该目录, 或 export MODEL=<路径> 重投。"
    exit 1
fi

# 缺数据则先转换 (幂等: 已存在则跳过)
if [ ! -f "$DATA" ]; then
    echo "[data] $DATA 不存在, 从 $SRC 转换..."
    python build_foarl_cvrp_data.py --src "$SRC" --out "$DATA" --k_nn 2
fi
if [ ! -f "$DATA" ]; then
    echo "[FATAL] 数据仍不存在: $DATA (检查 SRC=$SRC)"
    exit 1
fi

SANITY_FLAG=""
LOG_STEPS=10
RESUME_FLAG="--resume_from_checkpoint auto"
# sanity: 少量样本 + 每步打 loss + 独立目录不续 ckpt (避免与全量互相污染)
if [ "$SANITY" = "1" ]; then
    SANITY_FLAG="--max_samples 512"
    LOG_STEPS=1
    RESUME_FLAG=""
    OUTPUT_DIR="${OUTPUT_DIR}_sanity"
    echo "[sanity] 输出改到独立目录: $OUTPUT_DIR (不续 ckpt)"
fi

accelerate launch --num_processes "$NUM_GPUS" --main_process_port 29610 \
    train_sft_foarl.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank 64 --lora_alpha 128 \
    --lr 2e-5 --epochs 3 \
    --batch_size 1 --grad_accum 8 \
    --warmup_ratio 0.05 \
    --max_length 4864 --max_output_length 1024 \
    --zero_stage 3 --gradient_checkpointing \
    --save_steps 500 --logging_steps "$LOG_STEPS" \
    $RESUME_FLAG \
    $SANITY_FLAG
EC=$?

echo "============================================================"
if [ "$EC" -eq 0 ]; then
    echo "  ✅ FOARL SFT 完成  $(date '+%F %T')  →  $OUTPUT_DIR/final_model"
    echo "  下一步: merge LoRA → eval(可行率/gap) → FOARL RL(rl_train 移植)"
else
    echo "  ⚠️ SFT 非零退出 (exit=$EC)"
fi
echo "============================================================"
exit "$EC"
