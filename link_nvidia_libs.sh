#!/bin/bash
# 把 pip 安装的 nvidia-* 包的 .so 库软链到 conda env 的 lib/ 下
# 解决 DeepSpeed/flash-attn 等编译时找不到 -lcurand/-lcublas 等问题

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

ENV_LIB="$CUDA_HOME/lib"
NVIDIA_DIR="$CUDA_HOME/lib/python3.12/site-packages/nvidia"

count=0
for lib in "$NVIDIA_DIR"/*/lib/lib*.so*; do
    [ -f "$lib" ] || continue
    name=$(basename "$lib")
    target="$ENV_LIB/$name"
    if [ ! -e "$target" ]; then
        ln -s "$lib" "$target"
        echo "[linked] $name"
        count=$((count + 1))
    fi
done

echo ""
echo "新建 $count 个软链到 $ENV_LIB"
echo ""
echo "验证关键库:"
for check in libcurand libcublas libcusparse libcudart libcufft libnccl; do
    found=$(ls "$ENV_LIB"/${check}* 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "  [OK] $check"
    else
        echo "  [MISS] $check"
    fi
done
