#!/bin/bash
#SBATCH --qos express
#SBATCH --gpus=1
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_hlr_%j.log

# HLR Smoke Test — 严格检查 Hierarchical Latent Reasoner 7 个 stage
#   [1-3, 7] 不依赖主模型 (LR forward / KV cache 一致性等)
#   [4-6]    需要主模型 + profiled jsonl
#
# 默认配置 (可通过环境变量覆盖):
#   BASE_MODEL_TYPE - 基座选 r1_distill | qwen3_thinking (默认 paths.sh 用 r1_distill)
#                     LR 架构会从主模型 config 自动 1/4 缩放, 不需要手动调
#   HLR_MODEL       - 主模型路径 (默认 $BASE_MODEL, 跟随 BASE_MODEL_TYPE)
#   HLR_DATA        - profiled jsonl 路径 (默认 Latent-SFT/data/profiled_cvrp20.jsonl)
#
# 用法:
#   sbatch Latent-SFT/submit_smoke_hlr.sh                              # R1-Distill-7B 默认
#   BASE_MODEL_TYPE=qwen3_thinking sbatch Latent-SFT/submit_smoke_hlr.sh # 换 Qwen3-4B-Thinking
#   HLR_MODEL=/path/to/grpo_checkpoint sbatch Latent-SFT/submit_smoke_hlr.sh
#
# 1 GPU express, 5-10 分钟 (主要是加载基座模型 + forward/backward)

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
source paths.sh

# cwd = UniCOP/, 所有相对路径基于此 (与 config.py 默认值对齐)

# $BASE_MODEL 由 paths.sh 根据 BASE_MODEL_TYPE 派生
MODEL_PATH="${HLR_MODEL:-$BASE_MODEL}"
DATA_PATH="${HLR_DATA:-Latent-SFT/data/profiled_cvrp20.jsonl}"
# profiled jsonl 不存在时 smoke test 自动降级到 stage 1-3 + 7;
# 要跑完整 7 stage, 先在 zhuoyi 上手动跑一次:
#   python Latent-SFT/train.py --hlr   (会 auto_rebuild_data 把 chains_template_cvrp20.jsonl 转成 profiled)
# 或直接调:
#   python Latent-SFT/entropy_profile.py --model $BASE_MODEL \
#     --data UniCOP-Distill/data/chains_template_cvrp20.jsonl \
#     --output Latent-SFT/data/profiled_cvrp20.jsonl

echo "============================================================"
echo "  HLR Smoke Test"
echo "  HOST              = $HOST_ID"
echo "  BASE_MODEL_TYPE   = $BASE_MODEL_TYPE"
echo "  MODEL             = $MODEL_PATH"
echo "  DATA              = $DATA_PATH"
echo "  (LR 架构自动从主模型 config 推断, 无需手动调超参)"
echo "============================================================"

if [ -f "$DATA_PATH" ]; then
    echo "✓ profiled jsonl 存在, 跑完整 7 stage"
    python Latent-SFT/smoke_test_hlr.py --model "$MODEL_PATH" --data "$DATA_PATH"
else
    echo "⚠ profiled jsonl 不存在 ($DATA_PATH)"
    echo "  跑 stage 1-3 + 7 (跳过需要数据的 stage 4-6)"
    echo "  先跑 entropy_profile.py 生成 profiled jsonl 后再用完整模式"
    python Latent-SFT/smoke_test_hlr.py --no_main_model
fi
