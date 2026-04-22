#!/bin/bash
# OpenRLHF 环境一键安装脚本
# 用法:
#   cd /Data04/yangzhihan/lzj/UniCOP-Reason/openrlhf
#   bash install.sh
#
# 前置条件:
#   - 服务器已有 conda
#   - /Data04/yangzhihan/envs/analog_env 里已有 nvcc (给 flash-attn 源码 fallback 用)
#   - 有外网访问权限(装 torch + flash-attn wheel)
#
# 总耗时: 10-15 分钟(走预编译 wheel)

set -euo pipefail

ENV_PATH=/Data04/yangzhihan/envs/openrlhf_env

echo "==============================================="
echo "OpenRLHF 环境安装"
echo "  目标 env: $ENV_PATH"
echo "==============================================="

# ── Step 1: 创建 conda env ───────────────────────────────────────────
if [ ! -d "$ENV_PATH" ]; then
    echo "[1/5] 创建 conda env (python 3.10)..."
    conda create -p "$ENV_PATH" python=3.10 -y
else
    echo "[1/5] env 已存在, 跳过创建"
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

python_ver=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "      python: $python_ver"

# ── Step 2: 装 PyTorch 2.5.1 + cu121 ───────────────────────────────
echo "[2/5] 装 PyTorch 2.5.1 + cu121..."
if ! python -c 'import torch' 2>/dev/null; then
    pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "      torch 已装: $(python -c 'import torch; print(torch.__version__)')"
fi

# ── Step 3: 装 flash-attn 2.8.3 (pre-built wheel, 免编译) ────────
echo "[3/5] 装 flash-attn 2.8.3 (wheel)..."
if ! python -c 'import flash_attn' 2>/dev/null; then
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    pip install "$WHEEL_URL"
else
    echo "      flash-attn 已装: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
fi

# ── Step 4: 装 OpenRLHF[vllm] (拉 vllm/deepspeed/ray/transformers) ─
echo "[4/5] 装 openrlhf[vllm]..."
pip install "openrlhf[vllm]"

# ── Step 5: 装本目录杂项依赖 ───────────────────────────────────────
echo "[5/5] 装 requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "==============================================="
echo "安装完成。下一步运行自检:"
echo "  conda activate $ENV_PATH"
echo "  python $SCRIPT_DIR/scripts/verify_env.py"
echo "==============================================="
