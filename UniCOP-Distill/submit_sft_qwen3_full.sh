#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/sft_qwen3_full_%j.log
#SBATCH --time=12:00:00

# Stage 2 SFT — Qwen3-4B-Thinking 正式跑: 4 卡 ZeRO-2 / 3 epoch / chains_template_cvrp20
# 用法: sbatch UniCOP-Distill/submit_sft_qwen3_full.sh

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# zhuoyi 多卡 NCCL topology 必加(否则 ZeRO init hang)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE=qwen3_thinking
source paths.sh

echo "============================================================"
echo "  Stage 2 SFT [FULL] — Qwen3-4B-Thinking, 4 GPU ZeRO-2"
echo "  BASE_MODEL = $BASE_MODEL"
echo "  data       = chains_template_cvrp20.jsonl (filter cvrp / size 20)"
echo "  epochs=3  batch=1 grad_accum=8  lr=1e-4  zero=2  LoRA r=16"
echo "============================================================"

OUTPUT_DIR=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_template_cvrp20

echo ">>> Step 1: SFT 训练"
accelerate launch --num_processes 4 --main_process_port 29600 \
    UniCOP-Distill/stage2_reasoning/train_sft_stage2.py \
    --model "$BASE_MODEL" \
    --data UniCOP-Distill/data/chains_template_cvrp20.jsonl \
    --filter_problems cvrp \
    --filter_sizes 20 \
    --epochs 3 \
    --batch_size 1 --grad_accum 8 \
    --lr 1e-4 \
    --max_length 8192 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 10 --save_steps 200

if [ ! -f "$OUTPUT_DIR/final_model/adapter_config.json" ]; then
    echo "ERROR: 训练未保存 LoRA adapter, 跳过 merge"
    exit 1
fi

echo ""
echo ">>> Step 2: 合并 LoRA adapter 到基座"
python UniCOP-Distill/stage1_solution/merge_adapter.py \
    --adapter_path "$OUTPUT_DIR/final_model"

echo ""
echo "============================================================"
echo "  完成! 训练 + merge"
echo "  合并模型: $OUTPUT_DIR/final_model"
echo "  Eval 请单独提交 submit_eval_qwen3.sh"
echo "============================================================"
