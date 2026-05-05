"""
诊断 SFT 训练显存问题。单卡运行，不需要 accelerate。

用法:
    source ../paths.sh
    python diagnose_memory.py

检查项:
  1. GPU 初始显存（是否有僵尸进程）
  2. 模型加载后显存（bf16 实际占用）
  3. 模型 + LoRA 显存
  4. 数据实际 token 长度分布
  5. 不同 seq_len 的前向 pass 峰值显存
"""

import gc
import json
import os
import subprocess
import sys

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_PATH = os.environ.get("BASE_MODEL", "")
if not MODEL_PATH:
    print("ERROR: BASE_MODEL 未设置, 请先 source ../paths.sh")
    sys.exit(1)

DATA_PATH = "data/chains_template_cvrp20.jsonl"
LORA_RANK = 64
LORA_ALPHA = 128
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def gpu_mem():
    """返回 (已分配GB, 已保留GB, 总量GB)"""
    if not torch.cuda.is_available():
        return 0, 0, 0
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    t = torch.cuda.get_device_properties(0).total_mem / 1024**3
    return a, r, t


def print_gpu(tag):
    a, r, t = gpu_mem()
    print(f"  [{tag}] allocated={a:.2f}GB  reserved={r:.2f}GB  total={t:.2f}GB  free≈{t-r:.2f}GB")


# ═══════════════════════════════════════════════════════════════
# 1. GPU 初始状态
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. GPU 初始状态")
print("=" * 60)

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,memory.free",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.strip().split("\n"):
        print(f"  {line.strip()}")
except Exception as e:
    print(f"  nvidia-smi 失败: {e}")

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,name",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    procs = result.stdout.strip()
    if procs:
        print(f"\n  当前占用 GPU 的进程:")
        for line in procs.split("\n"):
            print(f"    {line.strip()}")
    else:
        print(f"\n  ✓ 无进程占用 GPU")
except Exception:
    pass

print_gpu("PyTorch 初始")

# ═══════════════════════════════════════════════════════════════
# 2. 模型参数量与显存
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. 加载模型 (bf16)")
print("=" * 60)

print(f"  模型路径: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

total_params = sum(p.numel() for p in model.parameters())
total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"  参数量: {total_params/1e9:.3f}B")
print(f"  参数体积: {total_bytes/1024**3:.2f}GB (dtype={model.dtype})")
print_gpu("模型加载后")

# ═══════════════════════════════════════════════════════════════
# 3. 加 LoRA 后显存
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"3. 应用 LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
print("=" * 60)

lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
print(f"  可训练参数: {trainable/1e6:.1f}M ({trainable_bytes/1024**2:.1f}MB)")
print_gpu("LoRA 后")

# ═══════════════════════════════════════════════════════════════
# 4. 数据 token 长度分布
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"4. 数据 token 长度分布: {DATA_PATH}")
print("=" * 60)

if os.path.isfile(DATA_PATH):
    prompt_lens = []
    completion_lens = []
    total_lens = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            system_text = r["prompt"]["system"]
            user_text = r["prompt"]["user"]
            output_text = r["output"]

            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_text},
                 {"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True,
            )
            p_len = len(tokenizer.encode(prompt))
            c_len = len(tokenizer.encode(output_text))
            prompt_lens.append(p_len)
            completion_lens.append(c_len)
            total_lens.append(p_len + c_len)

    prompt_lens = np.array(prompt_lens)
    completion_lens = np.array(completion_lens)
    total_lens = np.array(total_lens)

    print(f"  样本数: {len(total_lens)}")
    print(f"  {'':15s} {'min':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}")
    for name, arr in [("prompt", prompt_lens), ("completion", completion_lens), ("total", total_lens)]:
        print(f"  {name:15s} {int(np.min(arr)):8d} {int(np.percentile(arr,25)):8d} "
              f"{int(np.percentile(arr,50)):8d} {int(np.percentile(arr,75)):8d} "
              f"{int(np.percentile(arr,90)):8d} {int(np.percentile(arr,95)):8d} "
              f"{int(np.percentile(arr,99)):8d} {int(np.max(arr)):8d}")

    for threshold in [2048, 2560, 3072, 3584, 4096, 4992, 6144, 8192]:
        n = np.sum(total_lens <= threshold)
        print(f"    total <= {threshold}: {n}/{len(total_lens)} ({100*n/len(total_lens):.1f}%)")
else:
    print(f"  文件不存在: {DATA_PATH}")

# ═══════════════════════════════════════════════════════════════
# 5. 不同 seq_len 的前向 pass 峰值显存
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. 前向 pass 显存测试 (gradient_checkpointing=True)")
print("=" * 60)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

test_configs = [
    (1, 1024), (1, 2048), (1, 3072), (1, 3584),
    (1, 4096), (1, 4992), (1, 6144),
    (2, 1024), (2, 2048), (2, 3072), (2, 3584),
    (2, 4096), (2, 4992),
]

for bs, seq_len in test_configs:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    try:
        dummy_ids = torch.randint(0, 1000, (bs, seq_len), device="cuda:0")
        dummy_labels = dummy_ids.clone()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=dummy_ids, labels=dummy_labels)
            loss = outputs.loss
            loss.backward()

        peak = torch.cuda.max_memory_allocated() / 1024**3
        _, _, total = gpu_mem()
        print(f"  batch={bs} seq={seq_len:5d} → peak={peak:.2f}GB  "
              f"headroom={total-peak:.2f}GB  ✓")
    except torch.cuda.OutOfMemoryError:
        print(f"  batch={bs} seq={seq_len:5d} → OOM  ✗")
    except Exception as e:
        print(f"  batch={bs} seq={seq_len:5d} → ERROR: {e}")
    finally:
        model.zero_grad(set_to_none=True)
        if 'dummy_ids' in dir():
            del dummy_ids, dummy_labels
        if 'outputs' in dir():
            del outputs
        if 'loss' in dir():
            del loss
        torch.cuda.empty_cache()
        gc.collect()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
