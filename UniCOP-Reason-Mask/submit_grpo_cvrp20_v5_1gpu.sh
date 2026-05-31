#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=2
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_1gpu_%j.log

# ── 单卡诊断实验 (1 训练 GPU + 1 vLLM GPU, ZERO_STAGE=0 无分片无 DeepSpeed = 零跨卡通信) ──
# 目的: 隔离"6 卡 ZeRO-2 的 bwd=960s 到底是通信还是计算"。
#   单卡没有任何跨卡梯度同步 → 若 bwd 掉到 ~300-400s, 实锤那 ~890s 是 6×卡主机中转的梯度同步;
#   若 bwd 仍 ~900s, 则是算力(我判错)。A5000 ≈ 3090 同代算力, 可比。
# 配置 (关键: 用 B=4 避免 OOM 污染计时, B=8 逼近 OOM 会让反向虚高 3-5×):
#   NUM_GEN=4 + PER_DEVICE_BATCH=4  → 单卡 B=4 (整除 4×1%4=0), 显存 ~17G/24G 宽松
#   GRAD_ACCUM=8                    → 32 completion/step
#   ZERO_STAGE=0                    → 纯单卡无 DeepSpeed, 零跨卡通信
# 读 profiler 表后, 算【每 completion 耗时 = 单步fwd+bwd / 32】, 跟 6 卡基线 5.70s/completion 比。
# 跑出表即 scancel (不用跑完)。
export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 单卡设置 (手动指定 GPU 绕过 run script 的 TRAIN_PROC>=2 限制) ──
export VLLM_GPU=1                 # SLURM 分配的第 2 张 (索引 1) 做 vLLM
export TRAIN_GPUS_CSV=0           # 第 1 张 (索引 0) 做训练
export TRAIN_PROC=1               # 单训练进程
export ZERO_STAGE=0               # 无分片无 DeepSpeed → 零跨卡通信
# 用 num_gen=4 + per_device_batch=4 → 单卡 B=4, 显存宽松(~17G/24G), 避免 OOM 污染计时。
# (B=8 会逼近 OOM → cudaMalloc 碎片回收停顿 → 反向虚高 3-5×, 计时不可信)
export NUM_GEN=4
export PER_DEVICE_BATCH=4
export GRAD_ACCUM=8              # 8 micro × B=4 = 32 completion/step; profiler 按每-completion 比
export OUTPUT_DIR_BASE=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/output_v5_1gpu

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
bash run_grpo_cvrp20_v5.sh
