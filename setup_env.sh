#!/bin/bash
# setup_env.sh — 在 CUDA driver 12.2 的服务器上建 UniCOP 训练环境
#
# 约束: driver 12.2 → 只能用 cu121 的 torch
# 版本栈: torch 2.5.1+cu121 → vLLM 0.6.6.post1 → TRL (待定)
#
# 用法: bash setup_env.sh

set -euo pipefail

ENV_NAME="unicop"
PYTHON_VER="3.12"

echo "=========================================="
echo "Step 1: 创建 conda 环境"
echo "=========================================="
conda create -n "$ENV_NAME" python=$PYTHON_VER -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Python: $(python --version)"
echo "Env: $CONDA_PREFIX"

echo ""
echo "=========================================="
echo "Step 2: 安装 torch 2.5.1+cu121"
echo "=========================================="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Step 3: 安装 vLLM 0.6.6.post1"
echo "=========================================="
# --no-deps 避免 vLLM 拉新 torch
pip install vllm==0.6.6.post1 --no-deps
# 补装 vLLM 的非 torch 依赖
pip install \
    ray>=2.10 \
    aiohttp \
    fastapi \
    uvicorn \
    pydantic \
    msgspec \
    lm-format-enforcer \
    outlines \
    xformers==0.0.28.post3 \
    triton \
    prometheus-client \
    sentencepiece \
    tiktoken
python -c "import vllm; print(f'vllm {vllm.__version__}')"

echo ""
echo "=========================================="
echo "Step 4: 安装 TRL (先试 0.16.0)"
echo "=========================================="
# TRL 0.16.0 是代码 sanity_versions.py 要求的最低版本
# 如果和 vLLM 0.6.x 不兼容,后面手动降级到 0.14/0.13
pip install trl==0.16.0
python -c "from trl import GRPOConfig, GRPOTrainer; print('TRL GRPOConfig+GRPOTrainer OK')"

echo ""
echo "=========================================="
echo "Step 5: 安装其余训练栈"
echo "=========================================="
pip install \
    transformers \
    accelerate \
    peft \
    datasets \
    numpy'<2.0' \
    scipy \
    wandb

# DeepSpeed: 预编译 cpu_adam
export CUDA_HOME="$CONDA_PREFIX"
DS_BUILD_CPU_ADAM=1 pip install deepspeed --no-cache-dir

echo ""
echo "=========================================="
echo "Step 6: 软链 nvidia pip 库到 env/lib"
echo "=========================================="
NVIDIA_DIR="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages/nvidia"
if [ -d "$NVIDIA_DIR" ]; then
    for lib in "$NVIDIA_DIR"/*/lib/lib*.so*; do
        [ -f "$lib" ] || continue
        name=$(basename "$lib")
        target="$CONDA_PREFIX/lib/$name"
        if [ ! -e "$target" ]; then
            ln -s "$lib" "$target"
        fi
    done
    echo "nvidia 库软链完成"
else
    echo "无 nvidia pip 包,跳过"
fi

echo ""
echo "=========================================="
echo "Step 7: 最终验证"
echo "=========================================="
python -c "
pkgs = ['torch','vllm','trl','transformers','deepspeed','peft','accelerate']
for p in pkgs:
    try:
        m = __import__(p)
        v = getattr(m,'__version__','?')
        print(f'[OK] {p:20s} {v}')
    except Exception as e:
        print(f'[FAIL] {p:20s} {e}')
import torch
print(f'CUDA: {torch.cuda.is_available()}  version: {torch.version.cuda}')
"

echo ""
echo "=========================================="
echo "完成！激活环境: conda activate $ENV_NAME"
echo "=========================================="
