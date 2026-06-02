#!/bin/bash
# submit_smoke_nrn.sh — zhuoyi SLURM 冒烟测: 验证 no-repeat-ngram logits processor 真生效
#   单卡, 几分钟。默认用已下好的 Qwen3-4B-Instruct-2507 (任意模型都行, 功能验证与模型无关)。
#   提交: sbatch submit_smoke_nrn.sh
#   换模型: MODEL=<路径> sbatch submit_smoke_nrn.sh

#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --job-name=zijia_nrn_smoke
#SBATCH --comment="zijianliu, no-repeat-ngram smoke test"
#SBATCH --exclude=canele1
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/nrn_smoke_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/nrn_smoke_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

MODEL="${MODEL:-/homes/zhuoyi/zijianliu/models/Qwen3-4B-Instruct-2507}"
GPU_MEM="${GPU_MEM:-0.8}"

# 先 conda activate 再 set -u (activate.d/~cuda-nvcc 会引用未设的 NVCC_PREPEND_FLAGS)
source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
set -uo pipefail

echo "############## no-repeat-ngram 冒烟测 ##############  $(date '+%F %T')"
echo "  MODEL=$MODEL"
if [ ! -d "$MODEL" ]; then
    echo "[FATAL] 模型不存在: $MODEL (用 MODEL=<路径> sbatch 指定一个已下好的模型)"
    exit 1
fi

GPU_MEM="$GPU_MEM" python smoke_no_repeat_ngram.py "$MODEL"
EC=$?
echo "============================================================"
echo "  冒烟测退出码: $EC  (0=PASS, 非0=FAIL)  $(date '+%F %T')"
echo "============================================================"
exit "$EC"
