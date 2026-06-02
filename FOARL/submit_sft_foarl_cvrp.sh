#!/bin/bash
# submit_sft_foarl_cvrp.sh — FOARL CVRP Stage-1 SFT (zhuoyi SLURM)
#   基座: Qwen3-4B-Instruct-2507 (非思维) | 数据: FOARL 原版内容 (无 think) | LoRA r64/α64, lr2e-4, 1ep
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
source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
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
[ "$SANITY" = "1" ] && SANITY_FLAG="--max_samples 256"

accelerate launch --num_processes "$NUM_GPUS" --main_process_port 29610 \
    train_sft_foarl.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank 64 --lora_alpha 64 \
    --lr 2e-4 --epochs 1 \
    --batch_size 4 --grad_accum 4 \
    --max_length 4096 --max_output_length 1024 \
    --zero_stage 3 --gradient_checkpointing \
    --save_steps 200 --logging_steps 10 \
    --resume_from_checkpoint auto \
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
