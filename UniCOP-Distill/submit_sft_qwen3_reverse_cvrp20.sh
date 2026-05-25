#!/bin/bash
#SBATCH --qos=large
#SBATCH --gpus=4
#SBATCH --job-name=zijia_sft_reverse_cvrp20
#SBATCH --comment="zijianliu, do not cancel"
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/slurm_%x_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/slurm_%x_%j.err
#SBATCH --no-requeue
#SBATCH --open-mode=append
#
# Reverse-CoT SFT —— 在 forward 模型基础上，用 1 万条 depot-insertion 反向数据训 3 epoch。
#   基座    = output_sft_qwen3_template_cvrp20/final_model (forward 已训 + merge 的全量模型)
#   数据    = data/chains_reverse_cvrp20_10k.jsonl (build_reverse_chains.py --sample 10000 --seed 42)
#   超参    = 完全照搬 submit_sft_qwen3_full.sh 模板原值 (lr1e-4 / max_len8192 / LoRA r64a128 / ZeRO-3)
#   Step1 SFT → Step2 merge LoRA adapter
# 用法: sbatch UniCOP-Distill/submit_sft_qwen3_reverse_cvrp20.sh
#
# ⚠️ 注意（two-stage 遗忘风险）: 在已训好的 forward 模型上再用 lr=1e-4 训 3 epoch 反向数据，
#    可能侵蚀 forward 能力。本脚本按"沿用模板超参"默认 lr=1e-4；若想更保守地减少遗忘，
#    可手动把下面 --lr 调成 2e-5（这是建议，未擅自改动模板值）。

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# zhuoyi 多卡 NCCL topology 必加（否则 ZeRO-3 init hang ~30min）
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP

MODEL=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_template_cvrp20/final_model
DATA=UniCOP-Distill/data/chains_reverse_cvrp20_10k.jsonl
OUTPUT_DIR=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_reverse_cvrp20
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

notify() {
    curl -s "https://sctapi.ftqq.com/$SCKEY.send" \
        -d "title=${1:0:100}" -d "desp=${2:0:500}" > /dev/null 2>&1 || true
}
trap 'notify "Reverse SFT 失败: line $LINENO (job $SLURM_JOB_ID)"' ERR

echo "============================================================"
echo "  Reverse-CoT SFT (depot insertion, 10k, 3 epoch)"
echo "  MODEL      = $MODEL"
echo "  DATA       = $DATA"
echo "  OUTPUT_DIR = $OUTPUT_DIR"
echo "  $(date)"
echo "============================================================"

# ── 前置检查 ──────────────────────────────────────────────────────────
if [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: 基座模型不存在或未 merge: $MODEL/config.json"; exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "ERROR: 数据不存在: $DATA"
    echo "       先生成: python UniCOP-Distill/build_reverse_chains.py \\"
    echo "         --input UniCOP-Distill/data/solutions_cvrp20.jsonl \\"
    echo "         --output $DATA --sample 10000 --seed 42"
    exit 1
fi

# ── GPU 占用诊断 + 外部占卡自助 exclude（沿用 full 脚本经验，7679/7841 教训）──
echo "  GPU diagnostic ($(date +%H:%M:%S)):"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/    /'
_min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -n "$_min_free" ] && [ "$_min_free" -lt 20000 ]; then
    echo "❌ 某卡 free<20GiB (=${_min_free}MiB)，别人占着；自助 exclude 重投"
    _new_bad="${BAD_NODES:+$BAD_NODES,}$SLURM_NODELIST"
    sbatch --exclude="$_new_bad" --export=ALL,BAD_NODES="$_new_bad" "$(realpath "$0")"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════
# Step 1: SFT 训练（4 GPU ZeRO-3, 3 epoch）
# ══════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1: SFT 训练 ($(date))"
echo "    epochs=3  batch=1 grad_accum=8 × 4 GPU = effective 32"
echo "    lr=1e-4  zero=3  LoRA r=64 alpha=128  max_length=8192"
echo ""

accelerate launch --num_processes 4 --main_process_port 29600 \
    UniCOP-Distill/stage2_reasoning/train_sft_stage2.py \
    --model "$MODEL" \
    --data "$DATA" \
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
    echo "ERROR: 训练未保存 LoRA adapter，跳过 merge"; exit 1
fi
notify "Reverse SFT Step1 完成" "job $SLURM_JOB_ID, output=$OUTPUT_DIR"

# ══════════════════════════════════════════════════════════════════════
# Step 2: 合并 LoRA adapter 到基座
# ══════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: 合并 LoRA adapter ($(date))"
python UniCOP-Distill/stage1_solution/merge_adapter.py \
    --adapter_path "$OUTPUT_DIR/final_model"

if [ ! -f "$OUTPUT_DIR/final_model/config.json" ]; then
    echo "ERROR: merge 后未生成 config.json"; exit 1
fi
notify "Reverse SFT Step2 完成 (merge)" "模型: $OUTPUT_DIR/final_model"

# ══════════════════════════════════════════════════════════════════════
# Step 3: 评估 —— forward CVRP-20 上对比 baseline(forward 模型) vs reverse 训练后
#   evaluate.py 测试实例用固定 seed=9999 → 两个模型评同一批 100 实例，可直接对比。
#   看"提升了多少"：parse=format_match_rate, feas=global_feasibility_rate,
#   dist=avg_best_dist(越低越好；只在可行实例上算)。
# ══════════════════════════════════════════════════════════════════════
trap - ERR   # eval 阶段失败不算训练失败（模型已保存），不再触发失败通知
echo ""
echo ">>> Step 3: 评估 forward CVRP-20 (baseline vs reverse) ($(date))"

REASON_DIR=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason
EVAL_DIR="$OUTPUT_DIR/eval_results"
EVAL_TEMP=0.6   # 与 submit_sft_qwen3_full.sh 评估协议一致

cd "$REASON_DIR"

echo "  [eval] baseline = forward 模型 ($MODEL)"
python evaluate.py --backend local \
    --model_path "$MODEL" \
    --problem cvrp --problem_size 20 --model_type reasoning \
    --max_completion_length 4096 --num_test 100 --num_samples 1 \
    --batch_size 4 --temperature $EVAL_TEMP \
    --save_dir "$EVAL_DIR/baseline"

echo "  [eval] after reverse SFT = $OUTPUT_DIR/final_model"
python evaluate.py --backend local \
    --model_path "$OUTPUT_DIR/final_model" \
    --problem cvrp --problem_size 20 --model_type reasoning \
    --max_completion_length 4096 --num_test 100 --num_samples 1 \
    --batch_size 4 --temperature $EVAL_TEMP \
    --save_dir "$EVAL_DIR/reverse"

cd /homes/zhuoyi/zijianliu/UniCOP

# ── 对比打印（同一批 100 实例）──
_jqfmt='.results[0] | "parse=\(.format_match_rate)  feas=\(.global_feasibility_rate)  dist=\(.avg_best_dist)"'
_b=$(jq -r "$_jqfmt" "$EVAL_DIR/baseline"/*.json 2>/dev/null | head -1)
_r=$(jq -r "$_jqfmt" "$EVAL_DIR/reverse"/*.json  2>/dev/null | head -1)

echo ""
echo "============================================================"
echo "  完成! $(date)  —— forward CVRP-20, 同一批 100 实例 (seed=9999)"
echo "  合并模型: $OUTPUT_DIR/final_model"
echo "  ----------------------------------------------------------"
echo "  before (forward baseline) : $_b"
echo "  after  (reverse SFT)      : $_r"
echo "  (dist 越低越好; parse/feas 越高越好)"
echo "============================================================"

notify "Reverse SFT + eval 全部完成" "baseline: $_b || reverse: $_r"
