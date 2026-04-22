#!/bin/bash
# OpenRLHF 环境一键安装脚本
# 用法:
#   cd /Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason/openrlhf
#   bash install.sh
#
# 前置条件:
#   - 服务器已有 conda
#   - /Data04/yangzhihan/envs/analog_env 里已有 nvcc (flash-attn 源码编译需要)
#   - 有外网访问权限
#
# 总耗时: 约 40-60 分钟
#   - torch 等包下载: 5 分钟
#   - flash-attn 源码编译: 20-40 分钟 (单架构 sm_86, MAX_JOBS=4)
#
# 版本选择说明 (2026-04 调研):
#   OpenRLHF 0.10.2 锁 flash-attn==2.8.3 + vllm==0.19.1
#   vllm 0.19.1 硬 pin torch==2.10.0 + torchvision==0.25.0 + torchaudio==2.10.0
#   flash-attn 2.8.3 官方 pre-built wheel 最高只到 torch 2.9,且 torch 2.10
#     的社区 wheel 只有 cp312,没有 cp310
#   → 结论: torch 必须 2.10.0, flash-attn 必须源码编译

set -euo pipefail

ENV_PATH=/Data04/yangzhihan/envs/openrlhf_env
CUDA_HOME_SRC=/Data04/yangzhihan/envs/analog_env/targets/x86_64-linux

echo "==============================================="
echo "OpenRLHF 环境安装"
echo "  目标 env:  $ENV_PATH"
echo "  CUDA_HOME: $CUDA_HOME_SRC"
echo "==============================================="

# ── Step 1: 创建 conda env ───────────────────────────────────────────
if [ ! -d "$ENV_PATH" ]; then
    echo "[1/6] 创建 conda env (python 3.10)..."
    conda create -p "$ENV_PATH" python=3.10 -y
else
    echo "[1/6] env 已存在, 跳过创建"
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

python_ver=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "      python: $python_ver"
if [ "$python_ver" != "3.10" ]; then
    echo "      警告: 期望 python 3.10, 当前 $python_ver (影响 flash-attn wheel 匹配)"
fi

# ── Step 2: 装 torch 2.10.0 全家桶 + cu128 ──────────────────────────
# 版本必须与 vllm 0.19.1 的 requires_dist 完全一致, 否则 pip 会报冲突
echo "[2/6] 装 torch 2.10.0 + torchvision 0.25.0 + torchaudio 2.10.0 (cu128)..."
if ! python -c 'import torch; assert torch.__version__.startswith("2.10")' 2>/dev/null; then
    pip install --force-reinstall \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "      torch 2.10.x 已装: $(python -c 'import torch; print(torch.__version__)')"
fi

# ── Step 3: 配置编译环境 (flash-attn 源码编译用) ─────────────────────
echo "[3/6] 配置 CUDA 编译环境..."
export CUDA_HOME="$CUDA_HOME_SRC"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4                       # 并行编译数, 防 OOM
export TORCH_CUDA_ARCH_LIST="8.6"       # 只为 3090 (sm_86) 编, 省时间

if ! command -v nvcc >/dev/null 2>&1; then
    echo "      ❌ nvcc 不在 PATH, CUDA_HOME 设置有误. 当前 CUDA_HOME=$CUDA_HOME"
    exit 1
fi
nvcc_ver=$(nvcc --version | grep "release" | head -1)
echo "      nvcc: $nvcc_ver"

# ── Step 4: 源码编译 flash-attn 2.8.3 ────────────────────────────────
# 为什么编译:
#   flash-attn 2.8.3 + torch 2.10.0 + cp310 + cxx11abiFALSE 这个组合
#   官方没出 wheel, 社区 wheel 只到 cp312. 必须自己编.
# 常见错误:
#   - ModuleNotFoundError: No module named 'torch' → 缺 --no-build-isolation
#   - OSError: CUDA_HOME does not exist → Step 3 的 export 没生效
#   - killed (OOM) → 降低 MAX_JOBS 到 2
echo "[4/6] 源码编译 flash-attn 2.8.3 (预计 20-40 分钟)..."
if ! python -c 'import flash_attn; assert flash_attn.__version__ == "2.8.3"' 2>/dev/null; then
    pip install flash-attn==2.8.3 --no-build-isolation
else
    echo "      flash-attn 2.8.3 已装"
fi

# ── Step 5: 装 OpenRLHF[vllm] (拉 vllm/deepspeed/ray/transformers) ─
echo "[5/6] 装 openrlhf[vllm]..."
pip install "openrlhf[vllm]"

# ── Step 6: 装本目录杂项依赖 (fastapi/uvicorn/pydantic 等) ───────────
echo "[6/6] 装 openrlhf/requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "==============================================="
echo "安装完成。下一步:"
echo "  conda activate $ENV_PATH"
echo "  # CUDA_HOME 要手动 export (或配到 activate.d 里):"
echo "  export CUDA_HOME=$CUDA_HOME_SRC"
echo "  python $SCRIPT_DIR/scripts/verify_env.py"
echo "==============================================="
