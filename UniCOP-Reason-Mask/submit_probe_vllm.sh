#!/bin/bash
#SBATCH --qos normal
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/probe_vllm_%j.log

# Probe vLLM 0.7.3 logits_processor 行为, 验证 stateless mask 设计的关键假设.
# 跑 5-10 分钟即结束 (单 GPU, 短 generation).
#
# 提交:
#   sbatch submit_probe_vllm.sh
# 查看输出:
#   cat probe_vllm_<job_id>.log
#   cat probe_records.txt  # 详细 records

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# NCCL 兼容 zhuoyi 拓扑 (跟训练脚本一致)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

# source paths.sh 让 BASE_MODEL 自动设置
source /homes/zhuoyi/zijianliu/UniCOP/paths.sh

echo "============================================"
echo "vLLM Probe - logits_processor 行为验证"
echo "Time: $(date)"
echo "Host: $HOSTNAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $BASE_MODEL"
echo "============================================"

python probe_vllm_logits_processor.py
