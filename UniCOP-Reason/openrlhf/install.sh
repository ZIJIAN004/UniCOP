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
#   - 最后一步配 conda activate hook, 以后进 env 自动 export CUDA_HOME
#
# 版本选择说明 (2026-04 调研):
#   OpenRLHF 0.10.2 锁 flash-attn==2.8.3 + vllm==0.19.1
#   vllm 0.19.1 硬 pin torch==2.10.0 + torchvision==0.25.0 + torchaudio==2.10.0
#   flash-attn 2.8.3 官方 pre-built wheel 最高只到 torch 2.9,且 torch 2.10
#     的社区 wheel 只有 cp312,没有 cp310
#   → 结论: torch 必须 2.10.0, flash-attn 必须源码编译
#
# 设计规则:
#   1. 所有 pip 调用用 `python -m pip` 形式, 保证走当前 env 的 python,
#      避免 PATH 里其他 pip (系统 pip / 其他 env pip) 干扰
#   2. CUDA_HOME 在所有可能触发 CUDA op 编译/检查的步骤前都已 export,
#      包括 deepspeed (via openrlhf[vllm]) 的 is_compatible() 检查

set -euo pipefail

ENV_PATH=/Data04/yangzhihan/envs/openrlhf_env
CUDA_HOME_SRC=/Data04/yangzhihan/envs/analog_env
# 注意: 必须用 env 根目录, 不是 targets/x86_64-linux
#   targets/x86_64-linux 路径对运行时 (libcudart) 够用,
#   但对编译时的 nvcc 不够: nvcc 编译时要调 $CUDA_HOME/bin/cudafe++,
#   而 cudafe++ 在 env 根目录的 bin/ 下, 不在 targets/.../bin/ 下

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
echo "      python: $python_ver  ($(which python))"
if [ "$python_ver" != "3.10" ]; then
    echo "      警告: 期望 python 3.10, 当前 $python_ver (影响 flash-attn wheel 匹配)"
fi

# ── Step 2: 统一配置 CUDA 编译环境 (提前到最早,后续所有步骤都要用) ───
# 前移原因:
#   - flash-attn 源码编译需要 (Step 4)
#   - openrlhf[vllm] 里的 deepspeed 在 import 时 is_compatible() 会读 CUDA_HOME,
#     没设就抛 MissingCUDAException
#   - 提前 export 一次,后续步骤全部受益
echo "[2/6] 配置 CUDA 编译环境..."
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
echo "      CUDA_HOME=$CUDA_HOME"

# ── Step 3: 装 torch 2.10.0 全家桶 + cu128 ──────────────────────────
# 版本必须与 vllm 0.19.1 的 requires_dist 完全一致, 否则 pip 会报冲突
echo "[3/6] 装 torch 2.10.0 + torchvision 0.25.0 + torchaudio 2.10.0 (cu128)..."
if ! python -c 'import torch; assert torch.__version__.startswith("2.10")' 2>/dev/null; then
    python -m pip install --force-reinstall \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "      torch 2.10.x 已装: $(python -c 'import torch; print(torch.__version__)')"
fi

# ── Step 4: 源码编译 flash-attn 2.8.3 ────────────────────────────────
# 为什么编译:
#   flash-attn 2.8.3 + torch 2.10.0 + cp310 + cxx11abiFALSE 这个组合
#   官方没出 wheel, 社区 wheel 只到 cp312. 必须自己编.
# 常见错误:
#   - ModuleNotFoundError: No module named 'torch' → 缺 --no-build-isolation
#   - OSError: CUDA_HOME does not exist → Step 2 的 export 没生效
#   - killed (OOM) → 降低 MAX_JOBS 到 2
echo "[4/6] 源码编译 flash-attn 2.8.3 (预计 20-40 分钟)..."
if ! python -c 'import flash_attn; assert flash_attn.__version__ == "2.8.3"' 2>/dev/null; then
    python -m pip install flash-attn==2.8.3 --no-build-isolation
else
    echo "      flash-attn 2.8.3 已装"
fi

# ── Step 5: 强制锁版本装 vllm + openrlhf ──────────────────────────
# 重要: 不能用 `pip install "openrlhf[vllm]"`, 因为 pip resolver 会被
# env 里已有的其他包约束拽跑, 选老版本 vllm (如 0.15.1) 匹配老 torch,
# 结果和我们 pinned 的 torch 2.10 打架.
# 解决: 按顺序 + 显式版本号, 不给 resolver 自由度.
# CUDA_HOME 已在 Step 2 export, deepspeed 的 post-install 检查能过.
echo "[5/6] 装 vllm 0.19.1 + openrlhf 0.10.2 (强制锁版本)..."
python -m pip install vllm==0.19.1
python -m pip install openrlhf==0.10.2

# ── Step 6: 装本目录杂项依赖 (fastapi/uvicorn/pydantic 等) ───────────
echo "[6/7] 装 openrlhf/requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

# ── Step 7: 配 conda activate/deactivate hook (根治 CUDA_HOME 问题) ─
# 一次性写 hook: 以后 conda activate openrlhf_env 自动 export CUDA_HOME,
# 不用再每次手动 export.
echo "[7/7] 配置 conda activate/deactivate hook..."
ACTIVATE_DIR="$ENV_PATH/etc/conda/activate.d"
DEACTIVATE_DIR="$ENV_PATH/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/cuda_home.sh" << EOF
#!/bin/bash
# 自动配置 CUDA_HOME, 每次 conda activate 时运行
# 指向 env 根目录 (不是 targets/x86_64-linux), 编译时 cudafe++ 才能被找到
export CUDA_HOME=$CUDA_HOME_SRC
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib:\${LD_LIBRARY_PATH:-}
EOF
chmod +x "$ACTIVATE_DIR/cuda_home.sh"

cat > "$DEACTIVATE_DIR/cuda_home.sh" << 'EOF'
#!/bin/bash
unset CUDA_HOME
# PATH 和 LD_LIBRARY_PATH 由 conda 自动恢复
EOF
chmod +x "$DEACTIVATE_DIR/cuda_home.sh"
echo "      activate hook: $ACTIVATE_DIR/cuda_home.sh"
echo "      deactivate hook: $DEACTIVATE_DIR/cuda_home.sh"

echo ""
echo "==============================================="
echo "安装完成。下一步:"
echo "  conda deactivate && conda activate $ENV_PATH   # 让 hook 生效"
echo "  echo \$CUDA_HOME                                # 验证: 应该有输出"
echo "  python $SCRIPT_DIR/scripts/verify_env.py"
echo "==============================================="
