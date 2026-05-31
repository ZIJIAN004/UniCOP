#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=2
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_1gpu_%j.log

# ── 单卡诊断实验 (1 训练 GPU + 1 vLLM GPU, ZERO_STAGE=0 无分片无 DeepSpeed = 零跨卡通信) ──
# 目的: 隔离"6 卡 ZeRO-2 的 bwd=960s 到底是通信还是计算"。
#   单卡没有任何跨卡梯度同步 → 若 bwd 掉到 ~300-400s, 实锤那 ~890s 是 6×卡主机中转的梯度同步;
#   若 bwd 仍 ~900s, 则是算力(我判错)。A5000 ≈ 3090 同代算力, 可比。
# 配置:
#   PER_DEVICE_BATCH=8  (单卡整除约束: 8×1 % num_gen(8) = 0)
#   GRAD_ACCUM=24       (有效 batch = 8×1×24 = 192, 与 6 卡 4×6×8=192 逐步可比)
#   LOGP_CHUNK=128      (保单卡 B=8 不 OOM, A5000/3090 24G)
#   ZERO_STAGE=0        (make_deepspeed_config 返回 None → 纯单卡, 无 DeepSpeed/通信)
# 跑出 profiler 那张"精细化阶段耗时"表后 scancel 即可 (不用跑完)。
# 想更快出结果可把 GRAD_ACCUM 降到 4 (步更短, 按每-completion 归一化比较)。
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
export PER_DEVICE_BATCH=8
export GRAD_ACCUM=24
export LOGP_CHUNK=128
export OUTPUT_DIR_BASE=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/output_v5_1gpu

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
bash run_grpo_cvrp20_v5.sh
