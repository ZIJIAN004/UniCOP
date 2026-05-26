#!/bin/bash
#SBATCH --qos=express
#SBATCH --gpus=1
#SBATCH --job-name=zijia_wave_bon_cvrp20
#SBATCH --comment="zijianliu, do not cancel"
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/slurm_%x_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/slurm_%x_%j.err
#SBATCH --no-requeue
#SBATCH --open-mode=append
#
# 波次式(wave, 我们的方法) vs 朴素 best-of-N · CVRP n=20 · 单 GPU 推理评估
# ───────────────────────────────────────────────────────────────────────
# 一次 evaluate.py 同时跑 --bestofn + --wave: 共享同一批 completion, 两个分析
# 用完全相同的样本 → 在【同算力(解码 token 总数)】下对比, 最干净.
#   --bestofn : 朴素 best-of-k (k=1..N) 的算力-质量 scaling 曲线 (POMO-free 基线)
#   --wave    : 1/4 客户检查点剪枝 (方案A: 25%只硬过滤, 50%/75% POMO 留一半)
# 结果都写进 save_dir 的 JSON: results["bestofn"]["scaling_curve"] / results["wave"].
#
# 用法:  sbatch UniCOP-Reason/submit_wave_vs_bon_cvrp20.sh
# 提交前可改下面的 MODEL / NUM_TEST / NUM_SAMPLES 等. express=1卡/1天/最高优先级.
#
# ⚠️ set -euo pipefail 故意【不放在这里】: conda 激活钩子 (cuda-nvcc) 引用未定义的
#    NVCC_PREPEND_FLAGS, set -u 下 'unbound variable' 直接崩 (job 8167 教训);
#    且 .bashrc/conda 激活在 set -e 下也可能因非零返回提前退出. 故先做环境/conda/
#    paths 设置, 激活完毕后再开严格模式 (对齐 run_grpo 的成功模式).

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

# paths.sh: 自动识别主机, 导出 REASON_DIR / POMO_CKPT_DIR / POMO_BASELINE_DIR / BASE_MODEL 等
cd /homes/zhuoyi/zijianliu/UniCOP
source ./paths.sh
cd "$REASON_DIR"

# 环境/conda/paths 都就绪后再开严格模式 (见顶部说明, 避开 conda 钩子的 unbound/非零返回)
set -eo pipefail

# ── 配置 (按需改) ─────────────────────────────────────────────────────────
# 要评估的模型: 默认用 SFT 产物 (forward). 想测 reverse/GRPO 模型就改这里.
#   qwen3_thinking → output_sft_qwen3_template_cvrp20/final_model
#   r1_distill     → output_sft_hybrid_cvrp20/final_model
MODEL="${MODEL:-$DISTILL_DIR/output_sft_qwen3_template_cvrp20/final_model}"

# 后端: vllm 强烈推荐 —— best-of-N(num_samples 大)必须用 vLLM 的 paged attention,
#   本地 HF (local) 会一次性并行 num_samples 条序列, KV cache 直接 OOM (job 8168 教训).
#   vLLM 生成完成后, 波次式的 POMO PRM (n=20, 很小) 在剩余显存里跑, 一般够.
BACKEND="${BACKEND:-vllm}"

NUM_TEST=50            # 测试实例数 (先 50 跑通, 稳定后可上 100)
NUM_SAMPLES=32         # 每实例采样数 = best-of-N 的 N / 波次式起始池. 16+ 才有意义
TEMP=0.6               # 采样温度 (>0 才有多样性; Qwen3-Thinking 官方推荐 0.6)
MAX_LEN=4096           # 生成长度上限 (CVRP20 链一般 2-4k, 4096 够且省时)
BATCH=1                # 仅 local 后端用 (prompt batch); vLLM 走 continuous batching, 忽略此值
KEEP_FRAC=0.5          # 波次式每个 halve 检查点保留比例

SAVE_DIR="$REASON_DIR/eval_wave_bon_cvrp20_$(date '+%Y%m%d_%H%M%S')"
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

notify() {
    curl -s "https://sctapi.ftqq.com/$SCKEY.send" \
        -d "title=${1:0:100}" -d "desp=${2:0:500}" > /dev/null 2>&1 || true
}
trap 'notify "wave-vs-bon CVRP20 失败: line $LINENO (job ${SLURM_JOB_ID:-NA})"' ERR

echo "============================================================"
echo "  wave vs best-of-N · CVRP-20"
echo "  HOST       = ${HOST_ID:-?}"
echo "  MODEL      = $MODEL"
echo "  NUM_TEST   = $NUM_TEST   NUM_SAMPLES = $NUM_SAMPLES   TEMP = $TEMP   MAX_LEN = $MAX_LEN"
echo "  POMO_CKPT  = $POMO_CKPT_DIR"
echo "  POMO_BASE  = $POMO_BASELINE_DIR"
echo "  SAVE_DIR   = $SAVE_DIR"
echo "  $(date)"
echo "============================================================"

# ── 前置检查 ──────────────────────────────────────────────────────────────
if [ ! -f "$MODEL/config.json" ]; then
    echo "❌ 模型不存在或未 merge: $MODEL/config.json"; exit 1
fi
# POMO CVRP n=20 checkpoint 约定: {POMO_CKPT_DIR}/*POMO_CVRP_n20/MODEL_FINAL.pt
POMO_CVRP_CKPT=$(ls -d ${POMO_CKPT_DIR}/*POMO_CVRP_n20 2>/dev/null | head -1 || true)
if [ -z "$POMO_CVRP_CKPT" ]; then
    echo "❌ 找不到 CVRP n=20 POMO checkpoint: ${POMO_CKPT_DIR}/*POMO_CVRP_n20"
    echo "   (--wave 需要它; 仅想跑 baseline 可临时去掉 --wave 相关行)"
    exit 1
fi
echo "  POMO CVRP-20 ckpt: $POMO_CVRP_CKPT"
mkdir -p "$SAVE_DIR"

echo ""
echo "  GPU ($(date +%H:%M:%S)):"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/    /'

# ── 评估: 一次生成, 同时算 baseline best-of-N 曲线 + 波次式剪枝 ──────────────
echo ""
echo ">>> evaluate.py --bestofn --wave  (backend=$BACKEND) ($(date))"
python evaluate.py --backend "$BACKEND" \
    --model_path "$MODEL" \
    --problem cvrp --problem_size 20 \
    --model_type reasoning \
    --num_test "$NUM_TEST" --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMP" \
    --max_completion_length "$MAX_LEN" \
    --batch_size "$BATCH" \
    --vllm_gpu_mem_util 0.8 \
    --save_dir "$SAVE_DIR" \
    --bestofn \
    --wave \
    --pomo_ckpt_dir "$POMO_CKPT_DIR" \
    --pomo_baseline_dir "$POMO_BASELINE_DIR" \
    --wave_keep_frac "$KEEP_FRAC"

echo ""
echo "============================================================"
echo "  完成! $(date)"
echo "  结果 JSON 在: $SAVE_DIR"
echo "  关键字段: results[*].bestofn.scaling_curve  (best-of-k 曲线)"
echo "            results[*].wave  (波次式: wave_C/baseline_C/省比例/同算力距离对比)"
echo "============================================================"

trap - ERR
notify "wave-vs-bon CVRP20 完成" "job ${SLURM_JOB_ID:-NA}, 结果在 $SAVE_DIR"
