#!/bin/bash
#SBATCH --qos large
#SBATCH --gpus=7
#SBATCH --exclude=canele1                 # 跳过易挂节点 canele1, SLURM 自动挑空闲节点
#SBATCH --no-requeue                      # canele1 易挂; 故障直接 FAIL 不重排, 避免日志截断/白跑
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/grpo_cvrp20_v5_noprm_%j.log

# 消融实验脚手架: v5 关闭 POMO PRM 过程奖励 (只用 A_feas + A_outcome) 跟主 v5 对比。
# ✅ 已启用 DISABLE_PRM=1 → 真正的 no-PRM 消融 (不加载 POMO, 只用 A_feas + A_outcome)。
# 跟主实验隔离: 独立输出目录 output_v5_noprm，独立 log。epochs=2 (config 全局默认已改)。
# GPU: 1 vLLM + 6 卡训练 (run 脚本默认动态挑卡), 跟主实验同拓扑, 对比才干净。
export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 消融开关 ──
export DISABLE_PRM=1                                                            # 关 POMO PRM, 只用 A_feas + A_outcome (no-PRM 消融)
export OUTPUT_DIR_BASE=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask/output_v5_noprm  # 独立输出目录
export SAVE_STEPS=10                                                            # 每 10 step 存一次 checkpoint (默认 50)

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
bash run_grpo_cvrp20_v5.sh
