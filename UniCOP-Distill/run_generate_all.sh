#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 一键数据生成脚本（A*STAR-Zhihan 主机）
# 功能：生成 HGS/LKH solver 解 → 构造思维链 SFT 数据
#
# 使用方法：
#   bash run_generate_all.sh                    # 全部问题类型 + n=20
#   bash run_generate_all.sh --problems cvrp    # 只跑 CVRP
#   bash run_generate_all.sh --sizes 20 50      # n=20 和 n=50
#   bash run_generate_all.sh --num_samples 1000 # 每组合 1000 条（调试用）
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ── 环境配置 ──────────────────────────────────────────────────────────────────
PROJ_DIR="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill"
CONDA_ENV="/Data04/yangzhihan/envs/unicop"
LKH_BIN="/Data04/yangzhihan/lzj/LKH-3.0.9/LKH"

# 激活 conda（尝试常见路径）
for CONDA_SH in \
    /Data04/yangzhihan/anaconda3/etc/profile.d/conda.sh \
    /Data04/yangzhihan/miniconda3/etc/profile.d/conda.sh \
    ~/anaconda3/etc/profile.d/conda.sh \
    ~/miniconda3/etc/profile.d/conda.sh \
    /opt/conda/etc/profile.d/conda.sh; do
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        break
    fi
done
command -v conda >/dev/null 2>&1 || { echo "ERROR: 找不到 conda，请手动 source conda.sh"; exit 1; }
conda activate "$CONDA_ENV" 2>/dev/null || conda activate unicop || { echo "ERROR: conda activate 失败"; exit 1; }

export LKH_BIN="$LKH_BIN"
cd "$PROJ_DIR"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
PROBLEMS="cvrp tsptw vrptw"
SIZES="20"
NUM_SAMPLES=50000
WORKERS=32
REJECT_RATIO=0.3
SEED=42

# ── 解析命令行参数 ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --problems)  shift; PROBLEMS="$1"; shift ;;
        --sizes)     shift; SIZES="$1"; shift ;;
        --num_samples) shift; NUM_SAMPLES="$1"; shift ;;
        --workers)   shift; WORKERS="$1"; shift ;;
        --reject_ratio) shift; REJECT_RATIO="$1"; shift ;;
        --seed)      shift; SEED="$1"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "════════════════════════════════════════════════════════════"
echo "  UniCOP 数据生成流水线"
echo "  问题类型: $PROBLEMS"
echo "  节点规模: $SIZES"
echo "  每组合样本数: $NUM_SAMPLES"
echo "  并行进程: $WORKERS"
echo "  Reject 比例: $REJECT_RATIO"
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p data

# ── Step 1: 生成 solver 解 ────────────────────────────────────────────────────
echo "[Step 1/2] 生成 solver 解 (HGS/LKH)..."
echo ""

for pt in $PROBLEMS; do
    for sz in $SIZES; do
        OUTPUT_SOL="data/solutions_${pt}${sz}.jsonl"
        echo "  → $pt n=$sz → $OUTPUT_SOL"
        python stage1_solution/generate_solutions.py \
            --problems $pt \
            --sizes $sz \
            --num_samples $NUM_SAMPLES \
            --workers $WORKERS \
            --seed $SEED \
            --lkh_bin "$LKH_BIN" \
            --output "$OUTPUT_SOL"
        echo ""
    done
done

# ── Step 2: 构造思维链 ────────────────────────────────────────────────────────
echo "[Step 2/2] 构造思维链 SFT 数据..."
echo ""

# 收集所有 solutions 文件
INPUT_FILES=""
for pt in $PROBLEMS; do
    for sz in $SIZES; do
        SOL_FILE="data/solutions_${pt}${sz}.jsonl"
        if [ -f "$SOL_FILE" ]; then
            INPUT_FILES="$INPUT_FILES $SOL_FILE"
        fi
    done
done

if [ -z "$INPUT_FILES" ]; then
    echo "ERROR: 没有找到任何 solutions 文件"
    exit 1
fi

# 按问题类型分别生成（方便单独管理）
for pt in $PROBLEMS; do
    for sz in $SIZES; do
        SOL_FILE="data/solutions_${pt}${sz}.jsonl"
        OUT_CHAIN="data/chains_template_${pt}${sz}.jsonl"
        if [ ! -f "$SOL_FILE" ]; then
            echo "  跳过: $SOL_FILE 不存在"
            continue
        fi
        echo "  → $SOL_FILE → $OUT_CHAIN (reject_ratio=$REJECT_RATIO)"
        python build_think_chains.py \
            --input "$SOL_FILE" \
            --output "$OUT_CHAIN" \
            --reject_ratio $REJECT_RATIO
        echo ""
    done
done

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  全部完成！生成的文件："
ls -lh data/chains_template_*.jsonl 2>/dev/null || echo "  (无文件)"
echo "════════════════════════════════════════════════════════════"
