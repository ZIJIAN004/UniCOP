#!/bin/bash
# setup_env.sh — 重建 UniCOP 训练环境（CUDA driver 12.2 兼容）
#
# 用法: bash setup_env.sh
# 完成后: conda activate unicop && bash auto_all.sh

set -eo pipefail

ENV_NAME="unicop"

echo "=========================================="
echo "Step 1: 删除旧环境（如存在）并创建新环境"
echo "=========================================="
conda deactivate 2>/dev/null || true
conda remove -n "$ENV_NAME" --all -y 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Python: $(python --version)"
echo "Env: $CONDA_PREFIX"

echo ""
echo "=========================================="
echo "Step 2: 安装 CUDA nvcc（编译 DeepSpeed 需要）"
echo "=========================================="
conda install -c nvidia cuda-nvcc cuda-toolkit -y
echo "nvcc: $(nvcc --version | tail -1)"

echo ""
echo "=========================================="
echo "Step 3: 安装 torch 2.5.1+cu121"
echo "=========================================="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Step 4: 安装 vLLM 0.7.3 + 完整依赖"
echo "=========================================="
pip install vllm==0.7.3
python -c "import vllm; print(f'vllm {vllm.__version__}')"

echo ""
echo "=========================================="
echo "Step 5: 安装 TRL 0.16.0 + HuggingFace 生态"
echo "=========================================="
pip install 'trl==0.16.0' 'transformers>=4.46,<=4.51.3' 'huggingface-hub>=0.27,<1.0'
python -c "from trl import GRPOConfig, GRPOTrainer; print('TRL OK')"

echo ""
echo "=========================================="
echo "Step 6: 安装其余训练栈"
echo "=========================================="
pip install \
    'accelerate>=1.2,<2.0' \
    'peft>=0.14,<1.0' \
    'datasets>=3.0,<4.0' \
    'scipy' \
    'wandb'

echo ""
echo "=========================================="
echo "Step 7: 预编译 DeepSpeed cpu_adam"
echo "=========================================="
export CUDA_HOME="$CONDA_PREFIX"

# 软链 nvidia pip 库到 env/lib（编译链接需要）
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
echo "nvidia 库软链: $count 个"

DS_BUILD_CPU_ADAM=1 pip install deepspeed --no-cache-dir --no-build-isolation
python -c "
from deepspeed.ops.op_builder import CPUAdamBuilder
b = CPUAdamBuilder()
b.load()
print('DeepSpeed cpu_adam OK')
"

echo ""
echo "=========================================="
echo "Step 8: 最终验证"
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
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "=========================================="
echo "完成！"
echo "  conda activate $ENV_NAME"
echo "  bash auto_all.sh"
echo "=========================================="
