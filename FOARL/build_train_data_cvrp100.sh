#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# build_train_data_cvrp100.sh — 从 solver 解构建 UniCOP(思维臂) + FOARL(无推理臂)
#                               两臂 CVRP100 训练数据 (受控对比, 实例集逐条 1:1 对齐)
#
# 前置: 先有 solver 解 (run_generate_foarl_cvrp100.sh SOLUTIONS_ONLY=1 跑完):
#         $DISTILL_DIR/data/solutions_cvrp100.jsonl
#
# 管线:
#   [1] build_think_chains.py  solutions → chains (stride=5 块决策, 压缩思维链)
#         → $DISTILL_DIR/data/chains_template_cvrp100.jsonl   (= UniCOP 思维臂 SFT 数据)
#   [2] build_foarl_cvrp_data.py  chains(取</think>后答案) → FOARL 原版格式 (k_nn=2)
#         → $FOARL_DIR/data/foarl_cvrp100.jsonl               (= FOARL 无推理臂 SFT 数据)
#
#   FOARL 源 = chains(非 solutions): 保证两臂实例集逐条一致(建链失败的样本两臂同步丢)。
#   stride=5 / k_nn=2 均已验证: stride 逻辑定量测过(cap算术/覆盖/select全过);
#   k_nn=2 是 FOARL 官方值(论文§A.1.2 + repo CVRPEnv.py, 不随规模变)。
#
# 可覆盖: STRIDE(默认5) K_NN(默认2)
# ═══════════════════════════════════════════════════════════════════════════════
set -eo pipefail

SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
notify() {
    curl -s "https://sctapi.ftqq.com/${SCKEY}.send" \
        --data-urlencode "title=$1" --data-urlencode "desp=${2:-}" >/dev/null 2>&1 || true
}

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_SCRIPT_DIR/../paths.sh"          # DISTILL_DIR / UNICOP_ROOT
FOARL_DIR="$UNICOP_ROOT/FOARL"

for CONDA_SH in \
    /Data04/yangzhihan/anaconda3/etc/profile.d/conda.sh \
    /Data04/yangzhihan/miniconda3/etc/profile.d/conda.sh \
    /Data04/yangzhihan/miniforge3/etc/profile.d/conda.sh \
    ~/anaconda3/etc/profile.d/conda.sh \
    ~/miniconda3/etc/profile.d/conda.sh; do
    [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; break; }
done
command -v conda >/dev/null 2>&1 || { echo "[FATAL] 找不到 conda"; exit 1; }
conda activate "$CUDA_HOME" 2>/dev/null || conda activate unicop || { echo "[FATAL] conda activate 失败"; exit 1; }
python -c "import numpy" 2>/dev/null || { echo "[FATAL] 缺 numpy"; exit 1; }

STRIDE="${STRIDE:-5}"
K_NN="${K_NN:-2}"
SOL="$DISTILL_DIR/data/solutions_cvrp100.jsonl"
CHAINS="$DISTILL_DIR/data/chains_template_cvrp100.jsonl"
FOARL_OUT="$FOARL_DIR/data/foarl_cvrp100.jsonl"

[ -f "$SOL" ] || { echo "[FATAL] 找不到 solver 解: $SOL (先跑 run_generate_foarl_cvrp100.sh SOLUTIONS_ONLY=1)"; exit 1; }
_n_sol=$(wc -l < "$SOL")

echo "════════════════════════════════════════════════════════════"
echo "  CVRP100 两臂训练数据构建  $(date '+%F %T')"
echo "  solver 解: $_n_sol 条  →  $SOL"
echo "  stride=$STRIDE  k_nn=$K_NN"
echo "════════════════════════════════════════════════════════════"

# ── [1] UniCOP 思维臂: solutions → chains (stride=5) ─────────────────────────────
cd "$DISTILL_DIR"
echo "[1/2] $(date '+%F %T') 构建 UniCOP 思维链 (stride=$STRIDE)..."
python build_think_chains.py --input "$SOL" --output "$CHAINS" --stride "$STRIDE"
_n_chains=$(wc -l < "$CHAINS")

# ── [2] FOARL 无推理臂: chains → FOARL 格式 (源=chains, 实例集对齐) ────────────────
cd "$FOARL_DIR"
echo "[2/2] $(date '+%F %T') 构建 FOARL 数据 (源=chains, k_nn=$K_NN)..."
python build_foarl_cvrp_data.py --src "$CHAINS" --out "$FOARL_OUT" --k_nn "$K_NN"
_n_foarl=$(wc -l < "$FOARL_OUT")

echo "════════════════════════════════════════════════════════════"
echo "  ✅ 完成  $(date '+%F %T')"
echo "  UniCOP 思维臂 : $_n_chains 条  →  $CHAINS"
echo "  FOARL 无推理臂: $_n_foarl 条  →  $FOARL_OUT"
echo "  两臂条数应一致 (实例集 1:1 对齐)"
echo "════════════════════════════════════════════════════════════"
if [ "$_n_chains" != "$_n_foarl" ]; then
    echo "  ⚠️ 两臂条数不一致 ($_n_chains vs $_n_foarl), 检查 FOARL parse 失败"
fi
notify "CVRP100 两臂训练数据完成 ✅" "UniCOP=$_n_chains, FOARL=$_n_foarl 条 (stride=$STRIDE,k_nn=$K_NN)。"
