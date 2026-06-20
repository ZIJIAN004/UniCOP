#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_generate_foarl_cvrp100.sh — 在 zhihan 上生成 FOARL CVRP100 SFT 数据
#
# 管线 (FOARL 不需要思维链, 直接 solver 解 → FOARL 格式):
#   [1] generate_solutions.py  PyVRP/HGS 解 CVRP100 (48 core 并行)
#         → $DISTILL_DIR/data/solutions_cvrp100.jsonl
#   [2] build_foarl_cvrp_data.py  转 FOARL 原版 instruction/input/output 格式
#         → $FOARL_DIR/data/foarl_cvrp100.jsonl   (= 训练读取的数据)
#
# ⚠️ zhihan 直连 SSH 无 SLURM, 必须挂 tmux (SIGHUP 会杀掉脱离的进程):
#       tmux new -s gen100
#       bash run_generate_foarl_cvrp100.sh
#       (Ctrl+b d 脱离; tmux attach -t gen100 重连)
#
# ⚠️ 耗时由 TIMEOUT 决定 (HGS 用 MaxRuntime, 即使早收敛也会空转到 timeout):
#       均匀 CVRP100 HGS 通常几秒即近最优, 180s 是浪费 → 默认 TIMEOUT=20s(留足余量)。
#       50000 ÷ 48 × 20s ≈ 5.8 小时 (TIMEOUT=180 则 ~52h, 不必要)。
#       脚本可断点续跑 (按 id 去重), 中途挂掉重跑自动从断点继续, 不重复解已完成实例。
#
# 可覆盖环境变量:
#   NUM_SAMPLES (默认 50000, 对齐 cvrp20)  WORKERS (默认 48)  SEED (默认 42)  K_NN (默认 2)
#   TIMEOUT (默认 20, HGS 每实例秒数; 想更快设 10/5, 想更稳设 30)
# 例: NUM_SAMPLES=50 bash run_generate_foarl_cvrp100.sh   # 先小批试跑验证管线
# ═══════════════════════════════════════════════════════════════════════════════
set -eo pipefail

# ── Server 酱通知 (长任务规则: 完成/异常/中断都推送) ────────────────────────────
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
notify() {
    curl -s "https://sctapi.ftqq.com/${SCKEY}.send" \
        --data-urlencode "title=$1" --data-urlencode "desp=${2:-}" >/dev/null 2>&1 || true
}
_t0=$(date +%s)
on_exit() {
    ec=$?
    dur=$(( ($(date +%s) - _t0) / 60 ))
    if [ "$ec" -eq 0 ]; then
        :  # 正常完成由下方显式 notify, 这里不重复
    elif [ "$ec" -eq 130 ]; then
        notify "FOARL CVRP100 数据生成被中断(Ctrl-C)" "已运行 ${dur} 分钟, 可重跑断点续传。"
    else
        notify "FOARL CVRP100 数据生成异常退出 exit=$ec" "已运行 ${dur} 分钟, 检查日志后重跑(断点续传)。"
    fi
}
trap on_exit EXIT

# ── 路径 / 环境 ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_SCRIPT_DIR/../paths.sh"          # 取 DISTILL_DIR / UNICOP_ROOT / LKH_BIN / CUDA_HOME
FOARL_DIR="$UNICOP_ROOT/FOARL"

# 激活 conda (非交互 shell 须显式 source conda.sh, 不能靠 .bashrc)
for CONDA_SH in \
    /Data04/yangzhihan/anaconda3/etc/profile.d/conda.sh \
    /Data04/yangzhihan/miniconda3/etc/profile.d/conda.sh \
    /Data04/yangzhihan/miniforge3/etc/profile.d/conda.sh \
    ~/anaconda3/etc/profile.d/conda.sh \
    ~/miniconda3/etc/profile.d/conda.sh; do
    [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; break; }
done
command -v conda >/dev/null 2>&1 || { echo "[FATAL] 找不到 conda, 手动 source conda.sh 后重跑"; exit 1; }
conda activate "$CUDA_HOME" 2>/dev/null || conda activate unicop || { echo "[FATAL] conda activate unicop 失败"; exit 1; }
python -c "import pyvrp, numpy, tqdm" 2>/dev/null || { echo "[FATAL] unicop 环境缺 pyvrp/numpy/tqdm"; exit 1; }
export LKH_BIN="$LKH_BIN"

# ── 参数 ────────────────────────────────────────────────────────────────────────
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
WORKERS="${WORKERS:-48}"
SEED="${SEED:-42}"
K_NN="${K_NN:-2}"
TIMEOUT="${TIMEOUT:-20}"
SOLUTIONS_ONLY="${SOLUTIONS_ONLY:-0}"   # 1 = 只跑 Step1 生成 solver 解, 跳过 FOARL 格式转换
SOL_OUT="$DISTILL_DIR/data/solutions_cvrp100.jsonl"
FOARL_OUT="$FOARL_DIR/data/foarl_cvrp100.jsonl"

_eta_h=$(python -c "print(f'{${NUM_SAMPLES}*${TIMEOUT}/${WORKERS}/3600:.1f}')" 2>/dev/null || echo "?")
echo "════════════════════════════════════════════════════════════"
echo "  FOARL CVRP100 SFT 数据生成   $(date '+%F %T')   host=$HOST_ID"
echo "  样本数=$NUM_SAMPLES  并行=$WORKERS core  seed=$SEED  k_nn=$K_NN"
echo "  解器=PyVRP/HGS  timeout=${TIMEOUT}s/实例  →  预计 Step1 ≈ ${_eta_h} 小时(满量)"
echo "  [1] solver 解 → $SOL_OUT"
if [ "$SOLUTIONS_ONLY" = "1" ]; then
    echo "  [2] FOARL 格式 → (SOLUTIONS_ONLY=1, 本次跳过)"
else
    echo "  [2] FOARL 格式 → $FOARL_OUT"
fi
echo "════════════════════════════════════════════════════════════"

# ── Step 1: PyVRP/HGS 解 CVRP100 (断点续跑, 自带 Server酱完成通知) ──────────────
cd "$DISTILL_DIR"
echo "[1/2] $(date '+%F %T') 生成 solver 解 (CVRP100)..."
python stage1_solution/generate_solutions.py \
    --problems cvrp --sizes 100 \
    --num_samples "$NUM_SAMPLES" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    --lkh_bin "$LKH_BIN" \
    --output "$SOL_OUT"

_n_sol=$(wc -l < "$SOL_OUT" 2>/dev/null || echo 0)
echo "[1/2] 完成: solutions 累计 $_n_sol 条"

# ── SOLUTIONS_ONLY: 只生成 solver 解, 到此为止 ──────────────────────────────────
if [ "$SOLUTIONS_ONLY" = "1" ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "  ✅ Step1 完成 (SOLUTIONS_ONLY=1, 跳过 FOARL 转换)  $(date '+%F %T')"
    echo "  solutions_cvrp100.jsonl : $_n_sol 条  →  $SOL_OUT"
    echo "  转 FOARL 格式: bash run_generate_foarl_cvrp100.sh  (会复用已有 solutions, 仅跑 Step2)"
    echo "════════════════════════════════════════════════════════════"
    notify "CVRP100 solver 解生成完成 ✅" "solutions=$_n_sol 条。下一步可转 FOARL 格式(去掉 SOLUTIONS_ONLY 重跑即可)。"
    exit 0
fi

# ── Step 2: 转 FOARL 原版格式 (内含可行性/距离/正则三重自检) ──────────────────────
cd "$FOARL_DIR"
echo "[2/2] $(date '+%F %T') 转 FOARL 格式..."
python build_foarl_cvrp_data.py \
    --src "$SOL_OUT" \
    --out "$FOARL_OUT" \
    --k_nn "$K_NN"

_n_foarl=$(wc -l < "$FOARL_OUT" 2>/dev/null || echo 0)
echo "════════════════════════════════════════════════════════════"
echo "  ✅ 全部完成  $(date '+%F %T')"
echo "  solutions_cvrp100.jsonl : $_n_sol 条"
echo "  foarl_cvrp100.jsonl     : $_n_foarl 条  →  $FOARL_OUT"
echo "════════════════════════════════════════════════════════════"
notify "FOARL CVRP100 数据生成完成 ✅" "solutions=$_n_sol, foarl=$_n_foarl 条。可开始 SFT(DATA=data/foarl_cvrp100.jsonl)。"
