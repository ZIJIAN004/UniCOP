"""
OpenRLHF 环境验证脚本

用途: 装完 openrlhf[vllm] 后,一次性检查所有核心包版本 + 兼容性
用法:
    conda activate /Data04/yangzhihan/envs/openrlhf_env
    python verify_env.py
"""

import sys
import importlib


REQUIRED_PKGS = [
    ("torch", None),
    ("openrlhf", None),
    ("vllm", "0.19.0"),
    ("deepspeed", "0.18.2"),
    ("ray", "2.48.0"),
    ("flash_attn", "2.8.0"),
    ("transformers", "4.57.0"),
    ("peft", None),
    ("bitsandbytes", None),
]


def check_version(pkg_name, min_version):
    try:
        mod = importlib.import_module(pkg_name)
        version = getattr(mod, "__version__", "unknown")
        status = "OK"
        warn = ""
        if min_version and version != "unknown":
            from packaging.version import Version
            if Version(version) < Version(min_version):
                status = "WARN"
                warn = f" (< {min_version} min)"
        return version, status, warn
    except ImportError as e:
        return None, "MISSING", f" ({e})"


def main():
    print("=" * 60)
    print("OpenRLHF Environment Verification")
    print("=" * 60)

    # Python
    print(f"Python:        {sys.version.split()[0]}")
    print(f"Executable:    {sys.executable}")
    print()

    # Packages
    print(f"{'Package':<15} {'Version':<15} {'Status':<10}")
    print("-" * 60)
    all_ok = True
    for pkg, min_v in REQUIRED_PKGS:
        version, status, warn = check_version(pkg, min_v)
        version_str = version if version else "N/A"
        print(f"{pkg:<15} {version_str:<15} {status}{warn}")
        if status == "MISSING":
            all_ok = False

    print()

    # CUDA check
    print("-" * 60)
    print("CUDA Environment")
    print("-" * 60)
    try:
        import torch
        print(f"torch.version.cuda:     {torch.version.cuda}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"GPU count:              {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB, "
                      f"sm_{props.major}{props.minor})")
    except Exception as e:
        print(f"CUDA check FAILED: {e}")
        all_ok = False

    print()

    # Flash-attn smoke test (tiny tensor, fast)
    print("-" * 60)
    print("FlashAttention Smoke Test")
    print("-" * 60)
    try:
        import torch
        from flash_attn import flash_attn_func
        q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
        out = flash_attn_func(q, k, v)
        assert out.shape == (1, 128, 8, 64)
        print(f"flash_attn_func OK, output shape = {out.shape}")
    except Exception as e:
        print(f"flash_attn FAILED: {e}")
        all_ok = False

    print()
    print("=" * 60)
    print("RESULT:", "ALL OK" if all_ok else "SOME ISSUES (see above)")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
