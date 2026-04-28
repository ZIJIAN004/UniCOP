#!/bin/bash
# setup_env.sh — 在现有 zjh 环境中降级到 CUDA 12.2 驱动兼容的版本栈
#
# 约束: driver 12.2 → 只能用 cu121 的 torch
# 版本栈: torch 2.5.1+cu121 → vLLM 0.6.6.post1 → TRL 0.16.0
#
# 用法: conda activate zjh && bash setup_env.sh

set -euo pipefail

echo "当前环境: $CONDA_PREFIX"
echo "Python: $(python --version)"
echo ""

echo "=========================================="
echo "Step 1: 降级 torch → 2.5.1+cu121"
echo "=========================================="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Step 2: 降级 vLLM → 0.6.6.post1"
echo "=========================================="
pip install vllm==0.6.6.post1 --no-deps
python -c "import vllm; print(f'vllm {vllm.__version__}')"

echo ""
echo "=========================================="
echo "Step 3: 降级 TRL → 0.16.0"
echo "=========================================="
pip install trl==0.16.0 --no-deps
python -c "from trl import GRPOConfig, GRPOTrainer; print('TRL GRPOConfig+GRPOTrainer OK')"

echo ""
echo "=========================================="
echo "Step 4: 重新编译 DeepSpeed cpu_adam"
echo "=========================================="
export CUDA_HOME="$CONDA_PREFIX"
DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.18.9 --no-deps --force-reinstall --no-cache-dir

echo ""
echo "=========================================="
echo "Step 5: 修复 numpy (如被连锁升级)"
echo "=========================================="
pip install 'numpy>=1.26,<2.0' --no-deps

echo ""
echo "=========================================="
echo "Step 6: 软链 nvidia pip 库"
echo "=========================================="
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
count=0
if [ -d "$NVIDIA_DIR" ]; then
    for lib in "$NVIDIA_DIR"/*/lib/lib*.so*; do
        [ -f "$lib" ] || continue
        name=$(basename "$lib")
        target="$CONDA_PREFIX/lib/$name"
        if [ ! -e "$target" ]; then
            ln -s "$lib" "$target"
            count=$((count + 1))
        fi
    done
fi
echo "新建 $count 个软链"

echo ""
echo "=========================================="
echo "Step 7: 最终验证"
echo "=========================================="
CUDA_HOME="$CONDA_PREFIX" python -c "
pkgs = ['torch','vllm','trl','transformers','deepspeed','peft','accelerate','numpy']
for p in pkgs:
    try:
        m = __import__(p)
        v = getattr(m,'__version__','?')
        print(f'[OK] {p:20s} {v}')
    except Exception as e:
        print(f'[FAIL] {p:20s} {e}')
import torch
print(f'CUDA: {torch.cuda.is_available()}  version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
"

echo ""
echo "=========================================="
echo "完成！可以跑 bash auto_all.sh"
echo "=========================================="
