"""环境检查脚本：验证训练栈核心包是否正常"""
import sys

PACKAGES = [
    "torch", "vllm", "trl", "transformers",
    "deepspeed", "peft", "accelerate", "numpy",
]

fail = 0
for p in PACKAGES:
    try:
        m = __import__(p)
        v = getattr(m, "__version__", "?")
        print(f"[OK  ] {p:20s} {v}")
    except Exception as e:
        print(f"[FAIL] {p:20s} {e}")
        fail += 1

import torch
print()
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version:   {torch.version.cuda}")
print(f"GPU count:      {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:          {torch.cuda.get_device_name(i)}")

if fail:
    print(f"\n{fail} 个包导入失败")
    sys.exit(1)
else:
    print("\n全部通过")
