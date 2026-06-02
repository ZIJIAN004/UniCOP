#!/bin/bash
#SBATCH --qos express
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/smoke_repro_%j.log

# Mini repro: 完整训练路径 (train.py 原代码不动), dataset 截前 32 条.
# 目的: 5-10 min 内复现 train.py 的 hang, 不用等 16 min HLRDataset 全量加载.
#
# 32 条 = GA(8) × 4 GPU = effective batch 32 一次, 跑过 step 0-3 (前 3 步打细 stamp),
# 第 8 step 触发 sync_gradients 路径 (但 32 / 4 / 1 = 8 micro-steps, 第 8 必 sync).
#
# 跑法:
#   sbatch Latent-SFT/submit_smoke_repro.sh
#
# 日志关键 stamp (按顺序):
#   step 0: batch fetched
#   step 0: BEFORE compute_hlr_loss        ← 第一次 forward 入口
#   step 0: AFTER compute_hlr_loss         ← 第一次 forward 通过 (gather+forward+forward)
#   step 0: AFTER accelerator.backward     ← 第一次 backward 通过 (GC 重做 forward)
#   step 1: BEFORE compute_hlr_loss        ← 第二步入口 (sync_grad 仍 False)
#   ...
#   step 7: AFTER accelerator.backward     ← 第 8 步触发 sync_gradients=True
#   step 7: optimizer.step                 ← 第一次真 update
#
# 如果某步缺 "AFTER" 就是 hang 在那一步.

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
# 缓解双 forward + output_hidden_states 高显存压力下的 allocator cache flush 抖动 (诊断 8442 反复出现).
# 纯 CUDA allocator 行为, 不改 collective/loss/B=1 假设. 首次上线先在 smoke 跑 1 步确认 PyTorch/NCCL 不报错.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# zhuoyi 必加 (踩坑 #29)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# 诊断 env
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900   # smoke 短跑, 900s 够
export NCCL_TIMEOUT=900
export DEEPSPEED_TIMEOUT=900
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20480
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=WARN
# smoke 默认开详细 stamp (train.py/model.py 的 _stamp/_loss_stamp)
export HLR_DEBUG=1

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop

cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source paths.sh

MODEL_PATH="${HLR_MODEL:-/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_template_cvrp20/final_model}"
PROFILED_DATA="${HLR_DATA:-Latent-SFT/data/profiled_qwen3_thinking_cvrp20.jsonl}"
LIMIT="${SMOKE_LIMIT:-32}"

echo "============================================================"
echo "  Mini smoke: train.py 完整路径, dataset 截 $LIMIT 条"
echo "  目的: 5-10 min 复现 hang, 不读 50000 全量"
echo "============================================================"
echo "  MODEL:    $MODEL_PATH"
echo "  DATA:     $PROFILED_DATA"
echo "  LIMIT:    $LIMIT (= GA(8) × 4 GPU = 1 sync cycle)"
echo "============================================================"

if [ ! -f "$PROFILED_DATA" ]; then
    echo "❌ profiled jsonl 不存在: $PROFILED_DATA"
    exit 1
fi

# 直接 reuse train.py, 加 --limit 截 dataset, 1 epoch
accelerate launch --num_processes 4 --main_process_port 29702 \
    Latent-SFT/train.py \
    --model "$MODEL_PATH" \
    --data "$PROFILED_DATA" \
    --epochs 1 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --output_dir /tmp/smoke_repro_$SLURM_JOB_ID \
    --logging_steps 1 --save_steps 100000 \
    --limit "$LIMIT"

EC=$?
echo ""
echo "============================================================"
echo "  smoke 退出 code=$EC"
if [ $EC -eq 0 ]; then
    echo "  ✓ 全部 step 通过, train.py 完整路径无 hang"
else
    echo "  ✗ smoke 失败 / hang, 看 log 最后一个 [STAMP] 定位"
fi
echo "============================================================"

# 清掉临时输出
rm -rf /tmp/smoke_repro_$SLURM_JOB_ID 2>/dev/null || true
exit $EC
