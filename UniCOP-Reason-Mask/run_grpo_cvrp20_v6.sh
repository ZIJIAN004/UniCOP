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

# ── 自动挑一个空闲 vLLM 端口 ───────────────────────────────────────
#   背景: 上一次 v6 跑挂后残留进程占着 8006, 新进程 bind 失败 → [Errno 98]。
#   策略: 从基准端口 (默认 8006, 可 env 覆盖) 向上扫 50 个, 取第一个没人 listen 的。
_port_busy() {   # 返回 0=被占用, 非0=空闲
    local p="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${p}\$"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1
    else
        # 退化方案: 能连上 127.0.0.1:p 说明有人 listen = 占用
        (exec 3<>"/dev/tcp/127.0.0.1/$p") >/dev/null 2>&1 && { exec 3>&- 3<&-; return 0; }
        return 1
    fi
}

_base_port="${VLLM_PORT:-8006}"          # 错开 v5 的 8004
_pick_port="$_base_port"
for _i in $(seq 0 49); do
    _try=$(( _base_port + _i ))
    if _port_busy "$_try"; then
        echo "[v6] port $_try 被占用, 试下一个..."
    else
        _pick_port="$_try"; break
    fi
done
export VLLM_PORT="$_pick_port"
echo "[v6] 选定 VLLM_PORT=$VLLM_PORT (基准 $_base_port)"

exec bash "$_D/run_grpo_cvrp20_v5.sh"
