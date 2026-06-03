#!/bin/bash
# sweep_proc_alpha_v6.sh — 扫 v6 主轴 proc_alpha_v6 ∈ {100,200,400,600} (train-only)
#
#   每个取值 = 一个独立 sbatch job, 复用 submit_grpo_cvrp20_v6.sh (train-only, 不 eval),
#   只把 PROC_ALPHA_V6 经 --export 注入 → train.py:180 覆盖 config.proc_alpha_v6
#   → grpo_prm_trainer.py:1630 在训练时读取 (同一 config 单例, 已验证链路通)。
#
#   输出目录各自隔离: output_v6_lr<LR>_ep<EPOCHS>_pa<PA>/cvrp_n20 (submit 里拼)。
#   互不覆盖, 也不会误 resume 别的 pa 的 checkpoint。
#
#   并发: 你账号一次最多并 2 个 7 卡 job → 4 个 job 一次性 sbatch, SLURM 自动排队,
#         跑完 2 个再上 2 个, 无需手动盯。
#
#   为什么 train-only (不接 eval): 先看 Tier-0 训练曲线 (reward / fully_feas_rate /
#   grad_norm) 选方向 —— 平的直接 scancel, 别浪费 merge+eval。幸存的 pa 再单独跑:
#       RL_MODEL=$(pwd)/output_v6_lr2e-5_ep1_pa400/cvrp_n20/merged_model \
#       ONLY=RL NUM_TEST=200 DO_BO1=1 DO_BO8WAVE=0 GPU=0,1,2,3 TP=4 bash run_eval_matrix.sh
#   (注: train-only 不自动 merge; eval 那步用的 run_eval_matrix 吃的是 merged_model,
#    所以幸存者要先 merge —— 或直接改投 submit_grpo_cvrp20_v6_eval.sh 带 --export=ALL,PROC_ALPHA_V6=<pa>
#    一条龙 train→merge→eval。)
#
#   用法 (集群登录节点, git pull 之后):
#       bash sweep_proc_alpha_v6.sh
#   可覆盖: LR=2e-5 EPOCHS=1 PA_LIST="100 200 400 600" bash sweep_proc_alpha_v6.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1}"
PA_LIST="${PA_LIST:-100 200 400 600}"

echo "############## sweep proc_alpha_v6 ##############"
echo "  LR=$LR  EPOCHS=$EPOCHS  PA_LIST=[$PA_LIST]  (train-only)"
echo "  你账号一次并 2 个 job, 其余 SLURM 排队。"
echo

for PA in $PA_LIST; do
    OUT="output_v6_lr${LR}_ep${EPOCHS}_pa${PA}"
    echo ">>> 提交 proc_alpha_v6=$PA  →  $OUT/cvrp_n20"
    sbatch --job-name="zijia_v6_pa${PA}" \
           --export="ALL,PROC_ALPHA_V6=${PA},LR=${LR},EPOCHS=${EPOCHS}" \
           submit_grpo_cvrp20_v6.sh
done

echo
echo "全部已提交。查看队列: squeue -u \$USER"
squeue -u "${USER:-$(whoami)}" 2>/dev/null || true
