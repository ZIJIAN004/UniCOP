"""检查 DeepSpeed cpu_adam 是否编译成功可用"""
import sys

print("=" * 50)
print("DeepSpeed cpu_adam 检查")
print("=" * 50)

try:
    import torch
    print(f"[OK  ] torch {torch.__version__} (CUDA {torch.version.cuda})")
except Exception as e:
    print(f"[FAIL] torch 导入失败: {e}")
    sys.exit(1)

try:
    import deepspeed
    print(f"[OK  ] deepspeed {deepspeed.__version__}")
except Exception as e:
    print(f"[FAIL] deepspeed 导入失败: {e}")
    sys.exit(1)

try:
    from deepspeed.ops.op_builder import CPUAdamBuilder
    builder = CPUAdamBuilder()
    print(f"[OK  ] CPUAdamBuilder 实例化")
except Exception as e:
    print(f"[FAIL] CPUAdamBuilder 导入失败: {e}")
    sys.exit(1)

# 检查是否有预编译的 .so
if builder.is_compatible():
    print(f"[OK  ] cpu_adam 兼容当前环境")
else:
    print(f"[WARN] cpu_adam 兼容性检查未通过")

# 实际加载 .so
try:
    op = builder.load()
    print(f"[OK  ] cpu_adam .so 加载成功")
except Exception as e:
    print(f"[FAIL] cpu_adam .so 加载失败: {e}")
    sys.exit(1)

# 跑一次 CPUAdam 验证能用
try:
    param = torch.randn(64, requires_grad=True, dtype=torch.float32)
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam([param], lr=1e-3)
    loss = param.sum()
    loss.backward()
    optimizer.step()
    print(f"[OK  ] CPUAdam 优化器实际运行成功")
except Exception as e:
    print(f"[FAIL] CPUAdam 运行失败: {e}")
    sys.exit(1)

print("\n全部通过，cpu_adam 可用")
