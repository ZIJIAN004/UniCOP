#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=7
#SBATCH --exclude=canele1                 # 易挂节点(用户称 cancel1)，排除避免训练中途挂掉
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_sweep_%j.log

# ── GPU 数 sweet-spot 扫描: N=2 / 4 / 6 顺序各跑几步, 一个 job 全包 ──
# 目的: 6 卡 bwd 大头是无 NVLink 主机中转的梯度同步; 通信随 N 恶化、并行随 N 提升 → 找平衡点。
# 全部 B=4 (per_device_batch=4)、num_gen=8 (信号不变)、ZeRO-2; 整除 4×N%8=0 (N 偶数) ✓。
# 每个 config 只跑 ~3 个 optimizer step (NUM_TRAIN=6N), profiler 在第 2 步打印"精细化阶段耗时"表后
# 该 config 自然结束 (trap on_exit 杀 vLLM), 进下一个。三张表都在本 log 里。
# 比较: 每-completion = (fwd+bwd) / (4 × N × grad_accum8) = (fwd+bwd)/(32N); 最低的 N 就是 sweet spot。
# 基线参考: 6 卡 ZeRO-2 之前是 5.70s/completion。
export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

WORK=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

run_one() {
    local N=$1 train_csv=$2 vllm_gpu=$3 num_train=$4
    echo ""
    echo "##################################################################"
    echo "###  SWEEP: ${N} 卡  (train GPU=$train_csv, vLLM GPU=$vllm_gpu, NUM_TRAIN=$num_train ~3 step)"
    echo "##################################################################"
    rm -rf "$WORK/output_v5_sweep_${N}gpu"
    VLLM_GPU="$vllm_gpu" TRAIN_GPUS_CSV="$train_csv" TRAIN_PROC="$N" \
      PER_DEVICE_BATCH=4 ZERO_STAGE=2 NUM_TRAIN="$num_train" \
      OUTPUT_DIR_BASE="$WORK/output_v5_sweep_${N}gpu" \
      bash run_grpo_cvrp20_v5.sh || echo "[sweep] ${N} 卡 run 退出 (正常结束或报错, 继续下一个)"
    # 清残留, 释放 GPU 显存 + vLLM 端口, 再进下一个
    pkill -u "$USER" -f vllm_serve_logprobs.py 2>/dev/null || true
    pkill -u "$USER" -f accelerate.commands.launch 2>/dev/null || true
    sleep 30
}

# N=2: train 0,1 | vLLM 2   (NUM_TRAIN=12 → 3 step)
run_one 2 "0,1"         2 12
# N=4: train 0,1,2,3 | vLLM 4  (NUM_TRAIN=24 → 3 step)
run_one 4 "0,1,2,3"     4 24
# N=6: train 0,1,2,3,4,5 | vLLM 6  (NUM_TRAIN=36 → 3 step)
run_one 6 "0,1,2,3,4,5" 6 36

echo ""
echo "##################################################################"
echo "###  SWEEP 完成。三张 profiler 表(2/4/6 卡)都在本 log 里。"
echo "###  各自算 每-completion = (fwd+bwd)/(32×N), 最低的 N = sweet spot。"
echo "##################################################################"
