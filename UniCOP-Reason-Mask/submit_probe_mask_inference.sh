#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/probe_mask_inference_%j.log

# Mask Inference Probe - 不训练, 用 vLLM 跑 5 个 CVRP prompt 对比 mask vs no-mask
# 5-10 分钟出结果, 直接看 mask 是否让模型 degenerate.
#
# 提交:
#   sbatch submit_probe_mask_inference.sh
# 查看输出:
#   cat probe_mask_inference_<job_id>.log
#   cat probe_mask_completions.txt
#   cat probe_mask_completions.json

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# NCCL (单卡 probe 不需要, 保留无害)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# ── 关键: conda env 激活 + PYTHONPATH (跟 run_grpo_cvrp20_v5.sh 一致) ──
source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

# source paths.sh 拿 BASE_MODEL / DISTILL_DIR / CUDA_HOME
source /homes/zhuoyi/zijianliu/UniCOP/paths.sh
export MODEL="$DISTILL_DIR/output_sft_hybrid_cvrp20/final_model"

# Env 诊断: 第一次失败时 (No module named 'vllm') 让我们能立刻定位
echo "============================================================"
echo "Mask Inference Probe"
echo "Time:  $(date)"
echo "Host:  $HOSTNAME"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "------------------------------------------------------------"
echo "Env 诊断:"
echo "  which python: $(which python)"
echo "  which conda:  $(which conda)"
echo "  CONDA_PREFIX: $CONDA_PREFIX"
echo "  CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "  PYTHONPATH:   ${PYTHONPATH:-(unset)}"
python -c "import sys; print('  sys.executable:', sys.executable)" || echo "  ✗ python -c 调用失败"
python -c "import vllm; print('  vllm version:', vllm.__version__)" || echo "  ✗ vllm import 失败! 检查 conda env"
echo "============================================================"

if [ ! -d "$MODEL" ]; then
    echo "❌ MODEL not found: $MODEL"
    exit 1
fi

# 关键: 加 PYTHONPATH 让 utils.* / problems.* import 工作 (跟 run_grpo_cvrp20_v5.sh 一致)
PYTHONPATH="/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask:${PYTHONPATH:-}" \
CUDA_HOME="$CUDA_HOME" \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
    python probe_mask_inference.py

echo ""
echo "============================================================"
echo "Probe 完成. 输出文件:"
echo "  - stdout log:                  probe_mask_inference_${SLURM_JOB_ID}.log"
echo "  - 10 completion 全文 (txt):    probe_mask_completions.txt"
echo "  - 10 completion 全文 (json):   probe_mask_completions.json"
echo "============================================================"
