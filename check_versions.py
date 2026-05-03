"""
check_versions.py — 检查当前环境的关键包版本 + CUDA / nvcc 信息

用法:
    python check_versions.py           # 检查所有关键包
    python check_versions.py --json    # 输出 JSON 格式（方便跨主机对比）
"""

import argparse
import importlib
import json
import os
import subprocess
import sys


PACKAGES = [
    "torch",
    "transformers",
    "trl",
    "vllm",
    "deepspeed",
    "peft",
    "accelerate",
    "datasets",
    "numpy",
    "scipy",
    "flash_attn",
    "openrlhf",
    "ray",
    "fastapi",
    "uvicorn",
    "wandb",
    "openai",
    "pyvrp",
    "tqdm",
]


def get_version(pkg_name: str) -> str | None:
    try:
        mod = importlib.import_module(pkg_name)
        return getattr(mod, "__version__", "installed (no __version__)")
    except ImportError:
        return None


def get_cuda_info() -> dict:
    info = {}
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda or "N/A"
        info["gpu_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        info["torch_cuda"] = "torch not available"

    info["CUDA_HOME"] = os.environ.get("CUDA_HOME", "not set")

    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                info["nvcc"] = line.strip()
                break
        else:
            info["nvcc"] = result.stdout.strip() or "nvcc found but no version line"
    except FileNotFoundError:
        info["nvcc"] = "not found"
    except Exception as e:
        info["nvcc"] = f"error: {e}"

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    versions = {}
    for pkg in PACKAGES:
        versions[pkg] = get_version(pkg)

    cuda = get_cuda_info()

    if args.json:
        print(json.dumps({"python": sys.version, "packages": versions, "cuda": cuda}, indent=2))
        return

    print(f"Python: {sys.version}")
    print(f"Env:    {sys.prefix}")
    print()

    print("=" * 50)
    print("Packages")
    print("=" * 50)
    for pkg in PACKAGES:
        ver = versions[pkg]
        if ver is None:
            print(f"  [ -- ] {pkg:<16s} not installed")
        else:
            print(f"  [ OK ] {pkg:<16s} {ver}")

    print()
    print("=" * 50)
    print("CUDA / GPU")
    print("=" * 50)
    print(f"  CUDA_HOME:  {cuda['CUDA_HOME']}")
    print(f"  nvcc:       {cuda['nvcc']}")
    print(f"  torch CUDA: {cuda.get('torch_cuda', 'N/A')}")
    print(f"  GPU avail:  {cuda.get('gpu_available', 'N/A')}")
    print(f"  GPU count:  {cuda.get('gpu_count', 'N/A')}")
    for gpu in cuda.get("gpus", []):
        print(f"              {gpu}")


if __name__ == "__main__":
    main()
