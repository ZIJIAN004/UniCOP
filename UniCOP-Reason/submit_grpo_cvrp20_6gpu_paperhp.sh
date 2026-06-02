#!/bin/bash
# submit_grpo_cvrp20_6gpu_paperhp.sh
# ── 对照实验: RL 超参对齐论文 (Jiang et al., NeurIPS 2025, arXiv:2509.16865) ──
#   目的: 看 CVRP20 的 optimality gap 会不会随"温和 RL"下降
#   相对基线 (config.py 默认) 的改动 —— 仅这三个超参:
#       learning_rate : 1e-5 → 1e-6   (论文值)
#       kl_coef(beta) : 0.01 → 0.05   (论文值, 更强约束防漂移)
#       num_train_epochs: 3 → 1        (论文值)
#   其余 (reward_mode=prm / clip 0.20-0.28 / num_generations=8) 保持不变。
#   基座: qwen3_thinking (4B);  规模: 快速探针 num_train=1000。
#   输出与基线隔离: output_paperhp (不覆盖 output_6gpu)。
#
# 提交:  sbatch submit_grpo_cvrp20_6gpu_paperhp.sh

#SBATCH --qos=large
#SBATCH --gpus=7
#SBATCH --job-name=zijia_cvrp20_paperhp
#SBATCH --comment="zijianliu, paper-HP ablation, do not cancel"
#SBATCH --exclude=canele1
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/grpo_cvrp20_paperhp_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/grpo_cvrp20_paperhp_%j.err

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

# ── 本次对照实验的覆盖项 (run 脚本读这些 env) ──────────────────────
export BASE_MODEL_TYPE=qwen3_thinking
export NUM_TRAIN=1000
export LR=1e-6
export KL_COEF=0.05
export EPOCHS=1
export OUTPUT_DIR_BASE=/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason/output_paperhp

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate unicop
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason
bash run_grpo_cvrp20_6gpu.sh
