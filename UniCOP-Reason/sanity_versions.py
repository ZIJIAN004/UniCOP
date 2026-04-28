"""
版本 + API 兼容性 sanity check（不加载模型、不训练）
用法：python sanity_versions.py

通过条件：全部检查打 [OK]，无 [FAIL]。
"""
import importlib
import inspect
import os
import subprocess
import sys


FAILED = []
WARNED = []


# DeepSpeed 在 import 时会尝试编译 CUDA ops，必须能找到 nvcc (CUDA_HOME)
if not os.environ.get("CUDA_HOME"):
    print("!" * 60)
    print("注意：环境变量 CUDA_HOME 未设置，DeepSpeed 将无法 import。")
    print("auto_train.sh 里已经 hardcode 设了，但直接跑本脚本需要你自己 export：")
    print("  export CUDA_HOME=/usr/local/cuda")
    print("（请先验证路径有效：ls $CUDA_HOME/bin/nvcc）")
    print("!" * 60)
    print()


def check(label, cond, detail=""):
    tag = "OK  " if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(label)


def warn(label, detail=""):
    print(f"  [WARN] {label}" + (f"  ({detail})" if detail else ""))
    WARNED.append(label)


print("=" * 60)
print("Step 1: 包导入 + 版本打印")
print("=" * 60)

PACKAGES = ["trl", "vllm", "transformers", "accelerate",
            "deepspeed", "peft", "torch"]
versions = {}
for pkg in PACKAGES:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        versions[pkg] = ver
        print(f"  [OK  ] {pkg:14s} {ver}")
    except Exception as e:
        print(f"  [FAIL] {pkg:14s} import 失败: {e}")
        FAILED.append(f"import {pkg}")

print()
print("=" * 60)
print("Step 2: 版本兼容性（对照 TRL 1.1.0 pyproject 硬约束）")
print("=" * 60)


def vtup(v):
    # "0.19.0" -> (0, 19, 0)
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


if "vllm" in versions:
    vllm_v = vtup(versions["vllm"])
    # TRL 1.1.0 pin: vllm>=0.11.0,<=0.17.1
    check("vllm >= 0.11.0", vllm_v >= (0, 11, 0), versions["vllm"])
    if vllm_v > (0, 17, 1):
        warn(f"vllm {versions['vllm']} 超过 TRL 1.1.0 官方上界 0.17.1",
             "降级建议 pip install 'vllm==0.17.1'")
    else:
        print(f"  [OK  ] vllm <= 0.17.1  ({versions['vllm']})")

if "transformers" in versions:
    tr_v = vtup(versions["transformers"])
    # PR #5541: GRPO+ZeRO-3 deque bug fixed in transformers 5.5.4
    if tr_v >= (5, 0, 0) and tr_v < (5, 5, 4):
        warn(f"transformers {versions['transformers']} 在 5.0~5.5.3 之间",
             "GRPO+ZeRO-3 有 IndexError: pop from empty deque (trl#4899)，升到 5.5.4+")
    else:
        check(f"transformers 不在 5.0~5.5.3 坑区",
              not (tr_v >= (5, 0, 0) and tr_v < (5, 5, 4)),
              versions["transformers"])

if "trl" in versions:
    trl_v = vtup(versions["trl"])
    check("trl >= 0.16.0（vllm-serve CLI 要求）", trl_v >= (0, 16, 0),
          versions["trl"])
    if trl_v == (0, 15, 0) or (trl_v >= (0, 15, 0) and trl_v < (0, 16, 0)):
        warn("trl 0.15.x 有 LoRA server 模式 bug (trl#2698)")

print()
print("=" * 60)
print("Step 3: TRL GRPOConfig 关键参数签名")
print("=" * 60)

try:
    from trl import GRPOConfig
    sig = inspect.signature(GRPOConfig.__init__)
    params = set(sig.parameters.keys())
    for p in ["use_vllm", "vllm_mode", "vllm_server_host",
             "vllm_server_port", "epsilon", "epsilon_high", "beta",
             "num_generations", "max_completion_length",
             "gradient_checkpointing"]:
        check(f"GRPOConfig 支持参数 '{p}'", p in params)
except Exception as e:
    print(f"  [FAIL] 导入 GRPOConfig: {e}")
    FAILED.append("GRPOConfig import")

print()
print("=" * 60)
print("Step 4: TRL 核心类 + vllm-serve CLI 存在")
print("=" * 60)

try:
    from trl import GRPOTrainer  # noqa: F401
    print("  [OK  ] trl.GRPOTrainer 可导入")
except Exception as e:
    print(f"  [FAIL] trl.GRPOTrainer: {e}")
    FAILED.append("GRPOTrainer import")

# 用 `python -m trl.cli` 而不是 `trl` 可执行文件，强制走当前 Python 环境
# 避免 /home/user/.local/bin/trl 的 shebang 指向别的 Python 导致 ImportError
try:
    out = subprocess.run([sys.executable, "-m", "trl.cli", "vllm-serve", "--help"],
                         capture_output=True, text=True, timeout=30)
    ok = out.returncode == 0 and "model" in (out.stdout + out.stderr).lower()
    check("'python -m trl.cli vllm-serve --help' 能执行", ok,
          f"returncode={out.returncode}")
    if not ok:
        print("  --- stdout ---")
        print(out.stdout[:500])
        print("  --- stderr ---")
        print(out.stderr[:500])
except Exception as e:
    check("'python -m trl.cli vllm-serve --help' 能执行", False, str(e))

print()
print("=" * 60)
print("Step 5: vLLM 核心类导入 + 连通 torch")
print("=" * 60)

try:
    from vllm import LLM, SamplingParams  # noqa: F401
    print("  [OK  ] vllm.LLM / SamplingParams 可导入")
except Exception as e:
    print(f"  [FAIL] vllm 导入: {e}")
    FAILED.append("vllm import")

print()
print("=" * 60)
print("Step 6: transformers + peft API")
print("=" * 60)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    print("  [OK  ] transformers.AutoModelForCausalLM 可导入")
except Exception as e:
    print(f"  [FAIL] transformers 5.x API: {e}")
    FAILED.append("transformers API")

try:
    from peft import LoraConfig, get_peft_model  # noqa: F401
    print("  [OK  ] peft.LoraConfig / get_peft_model 可导入")
except Exception as e:
    print(f"  [FAIL] peft API: {e}")
    FAILED.append("peft API")

print()
print("=" * 60)
print("Step 7: CUDA / GPU 可见性")
print("=" * 60)

try:
    import torch
    check("torch.cuda.is_available()", torch.cuda.is_available())
    n = torch.cuda.device_count()
    check("GPU 数量 >= 4", n >= 4, f"检测到 {n} 张")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [INFO] GPU {i}: {name} ({total_gb:.1f} GB)")
except Exception as e:
    print(f"  [FAIL] torch.cuda: {e}")
    FAILED.append("torch.cuda")

print()
print("=" * 60)
print("Step 8: DeepSpeed 可用")
print("=" * 60)

try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import Init  # noqa: F401
    print(f"  [OK  ] deepspeed.runtime.zero 可导入 ({deepspeed.__version__})")
except Exception as e:
    print(f"  [FAIL] deepspeed ZeRO: {e}")
    FAILED.append("deepspeed ZeRO")

print()
print("=" * 60)
print("总结")
print("=" * 60)
if WARNED:
    print(f"  {len(WARNED)} 条警告：")
    for w in WARNED:
        print(f"    - {w}")

if FAILED:
    print(f"  {len(FAILED)} 条失败：")
    for f in FAILED:
        print(f"    - {f}")
    print("\n[FAIL] sanity check 未通过，上面列出的问题要先解决。")
    sys.exit(1)
else:
    print("  [PASS] 所有强制检查通过。")
    if WARNED:
        print("  注意：仍有警告项，建议处理后再跑训练。")
    sys.exit(0)
