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
#   cat probe_mask_inference_<job_id>.log  (stdout, 含 AUTO-DIAGNOSIS)
#   cat probe_mask_completions.txt          (全部 10 个 completion 全文)
#   cat probe_mask_completions.json         (含 metadata)

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# NCCL 兼容 zhuoyi 拓扑 (跟训练脚本一致, 虽然单卡不需要 NCCL 但保留无害)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

# 模型路径 (从 paths.sh 拿 DISTILL_DIR)
source /homes/zhuoyi/zijianliu/UniCOP/paths.sh
export MODEL="$DISTILL_DIR/output_sft_hybrid_cvrp20/final_model"

echo "============================================================"
echo "Mask Inference Probe"
echo "Time:  $(date)"
echo "Host:  $HOSTNAME"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "============================================================"

if [ ! -d "$MODEL" ]; then
    echo "❌ MODEL not found: $MODEL"
    exit 1
fi

python probe_mask_inference.py

echo ""
echo "============================================================"
echo "Probe 完成. 关键输出文件:"
echo "  - stdout log:                  probe_mask_inference_${SLURM_JOB_ID}.log"
echo "  - 10 completion 全文 (txt):    probe_mask_completions.txt"
echo "  - 10 completion 全文 (json):   probe_mask_completions.json"
echo "============================================================"
