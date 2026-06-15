#!/bin/bash
# run_grpo_cvrp20_v6.sh — 纯净 v6 (PRM 批级截尾标准化 + sigmoid, proc_alpha_v6 默认 1000)
#
# v6 vs v5: A_out (A_feas+A_outcome) 完全复用 v5; 只把 PRM per-step a_proc 从
#   v5 的 (prm_base + tanh(R_step)) 换成「批级截尾标准化 + sigmoid」:
#     gather 全批 fully-feas normal step 的原始 R_step → 按 |R| 剔最大 5% →
#     mu=mean(bulk), s=clamp(std(bulk)) → a_proc = sigmoid((R-mu)/s) ∈ (0,1)。
#   注入沿用 v5 mean 模式 (proc_alpha_v6 * a_proc / seg_len)。
#
# 纯净: 不传 --use_mask (use_mask 默认 False), 跟"纯净 v5"同口径。
# 本脚本是 v5 launcher 的薄封装, 仅经 env 切 reward_scheme=v6 / 输出目录;
#   LR/EPOCHS 等温和超参由 submit_grpo_cvrp20_v6.sh 设 (LR=1e-6 EPOCHS=1)。
#
# 手动:  bash run_grpo_cvrp20_v6.sh
# SBATCH: sbatch submit_grpo_cvrp20_v6.sh
#
# ── vLLM 端口为什么沿用 v5 默认 8004 (不再覆盖) ──────────────────────
#   每个 job 独占整节点(--gpus=7, 节点最多 8 卡装不下第二个 7 卡 job), 所以 v5/v6
#   永远不在同一节点 → 跟 v5 共用 8004 不可能冲突。
#   且 8004 在 net.ipv4.ip_local_port_range(临时端口范围)之下, 不会被 vLLM 初始化
#   那 ~19s 里建的 outgoing 连接(NCCL/torch.dist/vLLM 内部)当源端口抢占。
#   之前把端口挪到 8006/8261/8262 高端口反而踩坑: 落进临时端口范围, uvicorn LISTEN
#   时该端口已被 outgoing socket 占走 → address already in use (干净节点也复现)。
#   对比 v5/noprm(都用 8004)从无此问题, 已定位为唯一差异。

set -euo pipefail
_D="$(cd "$(dirname "$0")" && pwd)"

export REWARD_SCHEME="${REWARD_SCHEME:-v6}"
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$_D/output_v6}"   # 跟 output_v5 隔离
# VLLM_PORT 不设 → run_grpo_cvrp20_v5.sh 用默认 8004 (低端口, 安全)。

echo "[v6] vLLM 端口沿用 v5 默认 8004; ephemeral 起点=$(awk 'NR==1{print $1}' /proc/sys/net/ipv4/ip_local_port_range 2>/dev/null || echo '?') (>8004 即印证: 高端口才会被临时端口范围抢占)"

exec bash "$_D/run_grpo_cvrp20_v5.sh"
