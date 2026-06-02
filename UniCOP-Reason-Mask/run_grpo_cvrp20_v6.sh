#!/bin/bash
# run_grpo_cvrp20_v6.sh — 纯净 v6 (PRM 批级截尾标准化 + sigmoid, proc_alpha_v6=200)
#
# v6 vs v5: A_out (A_feas+A_outcome) 完全复用 v5; 只把 PRM per-step a_proc 从
#   v5 的 (prm_base + tanh(R_step)) 换成「批级截尾标准化 + sigmoid」:
#     gather 全批 fully-feas normal step 的原始 R_step → 按 |R| 剔最大 5% →
#     mu=mean(bulk), s=clamp(std(bulk)) → a_proc = sigmoid((R-mu)/s) ∈ (0,1)。
#   注入沿用 v5 mean 模式 (proc_alpha_v6 * a_proc / seg_len)。
#
# 纯净: 不传 --use_mask (use_mask 默认 False), 跟"纯净 v5"同口径。
# 本脚本是 v5 launcher 的薄封装, 仅经 env 切 reward_scheme=v6 / 输出 / 端口;
#   LR/EPOCHS 等温和超参由 submit_grpo_cvrp20_v6.sh 设 (LR=1e-6 EPOCHS=1)。
#
# 手动:  bash run_grpo_cvrp20_v6.sh
# SBATCH: sbatch submit_grpo_cvrp20_v6.sh

set -euo pipefail
_D="$(cd "$(dirname "$0")" && pwd)"

export REWARD_SCHEME="${REWARD_SCHEME:-v6}"
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$_D/output_v6}"   # 跟 output_v5 隔离
export VLLM_PORT="${VLLM_PORT:-8006}"                         # 错开 v5 的 8004

exec bash "$_D/run_grpo_cvrp20_v5.sh"
