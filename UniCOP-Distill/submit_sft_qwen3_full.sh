#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=4
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/sft_qwen3_full_%j.log

# Stage 2 SFT — Qwen3-4B-Thinking 一体化: 训练 → merge → eval
#   Step 1: 4 GPU ZeRO-2 SFT (3 epoch, 全量数据)
#   Step 2: merge LoRA adapter (1 GPU 进程)
#   Step 3: HF backend eval 100 条 (1 GPU 进程, batch=4),
#           输出 JSON 含 6 条样本 completion (3 parsed + 3 unparsed)
#
# 注: SLURM 占 4 卡整 job 时长, step 2/3 只用 1 卡, 闲置 3 卡换简单可靠。
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

OUTPUT_DIR=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_template_cvrp20
EVAL_DIR="$OUTPUT_DIR/eval_results"

echo "============================================================"
echo "  Qwen3-4B-Thinking 一体化 Pipeline"
echo "  Step 1 SFT → Step 2 merge → Step 3 eval"
echo "============================================================"
echo "  BASE_MODEL   = $BASE_MODEL"
echo "  OUTPUT_DIR   = $OUTPUT_DIR"
echo "  EVAL_DIR     = $EVAL_DIR"
echo "  采样参数     = T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Step 1: SFT 训练 (4 GPU ZeRO-2, 3 epoch)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: SFT 训练 ($(date))"
echo "    epochs=3  batch=1 grad_accum=8 × 4 GPU = effective 32"
echo "    lr=1e-4  zero=3 (LoRA + GC use_reentrant=True 自动)"
echo "    LoRA r=64 alpha=128  scheduler=cosine warmup=0.05  wd=0.01"
echo "    max_length=8192 (prompt+completion 总长 上限)"
echo "    max_output_length=4096 (completion 单段过滤; 数据集实测 ≤3100 安全)"
echo ""

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
    --lora_rank 64 --lora_alpha 128 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --resume_from_checkpoint auto \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 10 --save_steps 200

if [ ! -f "$OUTPUT_DIR/final_model/adapter_config.json" ]; then
    echo "ERROR: 训练未保存 LoRA adapter, 跳过 merge 和 eval"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════
# Step 2: 合并 LoRA adapter 到基座
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 合并 LoRA adapter ($(date))"
python UniCOP-Distill/stage1_solution/merge_adapter.py \
    --adapter_path "$OUTPUT_DIR/final_model"

if [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: merge 后未生成 config.json, 跳过 eval"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════
# Step 3: HF backend eval (100 条 CVRP-20, 输出含 6 条样本 completion)
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: Evaluate (100 条 CVRP-20, HF backend) ($(date))"
echo "    输出: $EVAL_DIR (含 examples 字段, 默认 3 parsed + 3 unparsed completion)"
echo ""

cd UniCOP-Reason
python evaluate.py \
    --backend local \
    --model_path "$OUTPUT_DIR/final_model" \
    --problem cvrp --problem_size 20 \
    --model_type reasoning \
    --max_completion_length 4096 \
    --num_test 100 --num_samples 1 \
    --batch_size 4 \
    --temperature "${GEN_TEMPERATURE:-0.6}" \
    --save_dir "$EVAL_DIR"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  训练 ckpt:  $OUTPUT_DIR/final_model"
echo "  合并模型:   $OUTPUT_DIR/final_model (覆盖)"
echo "  Eval 结果: $EVAL_DIR/*.json (含 examples 字段)"
echo ""
echo "  Quick view:"
echo "    jq '.results[0] | {parse_rate:.format_match_rate, "
echo "         feas:.global_feasibility_rate, dist:.avg_best_dist}' $EVAL_DIR/*.json"
echo "============================================================"
