#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=4
#SBATCH --exclude=canele1
#SBATCH --no-requeue
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/ablate_hlr_%j.log

# ── backward 三路消融诊断 ────────────────────────────────────────────────
# 一个 job 顺序跑 3 个 loss 配置, 每个只跑 ABLATE_MAX_STEPS 个 optimizer step 就停,
# 对比各配置 [TIMING] 的 backward 段, 差分出 teacher CE / align / student 哪路吃了 backward。
#   1) FULL       α=1 β=1 γ=1  (= 当前真实训练, backward 基线)
#   2) NO_TEACHER α=1 β=1 γ=0  (去 teacher CE 路径)
#   3) NO_ALIGN   α=1 β=0 γ=1  (去 align/LR 反向路径)
# 差分:  teacher 路径 ≈ FULL.backward − NO_TEACHER.backward
#        align   路径 ≈ FULL.backward − NO_ALIGN.backward
#        student 基底 ≈ NO_TEACHER 与 NO_ALIGN 的共同剩余
# 前提: model.py 已改为 β/γ=0 时真把该项踢出计算图 (否则 0×loss 仍反向遍历, 测不准)。
# 配置完全照搬正式训练 (ZeRO-3 + GC + 4 卡), 只动 loss 项, 保证可比。

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
export PYTHONUNBUFFERED=1

# 无 NVLink NCCL (同 submit_train_hlr)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export DEEPSPEED_TIMEOUT=3600
export NCCL_DEBUG=WARN

# 诊断: 分段计时 + 每配置只跑几个 optimizer step
export HLR_TIMING=1
ABLATE_MAX_STEPS="${ABLATE_MAX_STEPS:-4}"   # logging_steps=2 → step2/step4 各报一次, 取 step4 (避开前几步 warmup)

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source paths.sh

MODEL_PATH="${HLR_MODEL:-$BASE_MODEL}"
PROFILED_DATA="${HLR_DATA:-Latent-SFT/data/profiled_${BASE_MODEL_TYPE}_cvrp20.jsonl}"
OUT_DIR="/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT/output_hlr_ablate"

if [ ! -f "$PROFILED_DATA" ]; then
    echo "❌ profiled 数据不存在: $PROFILED_DATA"
    echo "   消融诊断需已有 profiled 数据 (先跑正常 train 生成, 或设 HLR_DATA=<已有路径>)"
    exit 1
fi

echo "============================================================"
echo "  backward 三路消融诊断"
echo "  model = $MODEL_PATH"
echo "  data  = $PROFILED_DATA"
echo "  每配置 $ABLATE_MAX_STEPS optimizer step | ZeRO-3 + GC + 4 卡 (同正式训练)"
echo "============================================================"

run_ablate() {
    local name=$1 a=$2 b=$3 g=$4
    echo ""
    echo "##################################################################"
    echo "###  ABLATE: $name   (α=$a β=$b γ=$g)   $(date '+%H:%M:%S')"
    echo "##################################################################"
    HLR_MAX_STEPS="$ABLATE_MAX_STEPS" \
    accelerate launch --num_processes 4 --main_process_port 29701 \
        Latent-SFT/train.py \
        --model "$MODEL_PATH" \
        --data "$PROFILED_DATA" \
        --zero_stage 3 \
        --gradient_checkpointing \
        --alpha "$a" --beta "$b" --gamma "$g" \
        --output_dir "$OUT_DIR" \
        --logging_steps 2 --save_steps 100000 \
        || echo "[ablate] $name accelerate 退出 (诊断 break 后正常)"
    sleep 5
}

# 三配置顺序 (各自独立 ZeRO-3 init, 互不污染)
run_ablate FULL       1 1 1
run_ablate NO_TEACHER 1 1 0
run_ablate NO_ALIGN   1 0 1

echo ""
echo "##################################################################"
echo "###  消融完成。三段 [TIMING] 都在本 log。看每段最后一行 backward 值:"
echo "###    teacher 路径 ≈ FULL.backward − NO_TEACHER.backward"
echo "###    align   路径 ≈ FULL.backward − NO_ALIGN.backward"
echo "###    student 基底 ≈ NO_TEACHER 与 NO_ALIGN 的共同剩余"
echo "###  提取: grep -E 'ABLATE:|TIMING r0 ' <本 log>"
echo "##################################################################"
