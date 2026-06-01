#!/bin/bash
# submit_grpo_cvrp20_v5_z2_5gpu.sh — zhuoyi 上的 ZeRO-2 不分片提速实验(对照 yangzhihan)
#   5 卡 = 1 vLLM + 4 训练 · ZeRO-2(不分片→无跨卡 all-gather) · batch2 · num_gen8(8 路对比不变)
#   batch2 × 4训练 × accum24 ÷ num_gen8 = 24 prompts/step, 等效 batch 与原 6 卡配置一致 → 公平对比
#
# 提交: sbatch submit_grpo_cvrp20_v5_z2_5gpu.sh
# 5 卡 > normal/long 上限(≤4) → 必须用 large(≤24)。

#SBATCH --qos large
#SBATCH --gpus=5
#SBATCH --job-name=zijia_v5_z2_5gpu
#SBATCH --comment="zijianliu, do not cancel"
#SBATCH --exclude=canele1                 # 跳过易挂节点 canele1, SLURM 自动挑空闲节点
#SBATCH --no-requeue                       # 【强制】节点故障直接 FAIL, 不重排截断日志
#SBATCH --open-mode=append                 # 【强制】日志追加, 杜绝被覆盖
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_z2_5gpu_%x_%j.log

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask

# SLURM 分配的 5 张卡在 job 内编号 0-4 → vLLM=GPU4, 训练=GPU0-3。
# NCCL 不设(用脚本默认 P2P/SHM 都关 = zhuoyi 良配, 防 init hang)。
# 注: ZeRO-2 用 FusedAdam(GPU), 首次 JIT 编译需 cuda_runtime.h;
#     paths.sh 已对 $CUDA_HOME/targets/x86_64-linux/include 做永久补全, 若 zhuoyi 该 env
#     无此布局而编译失败, 同样需补 CUDA dev 头(见 rules-detail "DeepSpeed CUDA_HOME")。
ZERO_STAGE=2 PER_DEVICE_BATCH=2 GRAD_ACCUM=24 \
VLLM_GPU=4 TRAIN_GPUS_CSV=0,1,2,3 TRAIN_PROC=4 \
bash run_grpo_cvrp20_v5.sh
