"""独立进程合并 LoRA adapter 到基座模型（不能在 DeepSpeed 环境下运行）。

用法:
    # 自动从 adapter_config.json 读取基座模型路径:
    python merge_adapter.py --adapter_path ./output/final_model

    # 手动指定基座模型路径（覆盖 adapter_config.json）:
    python merge_adapter.py --adapter_path ./output/final_model --base_model /path/to/base

    # 指定输出路径（默认覆盖 adapter_path）:
    python merge_adapter.py --adapter_path ./output/final_model --output_path ./output/merged
"""

import argparse
import os
import sys

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型路径，不传则从 adapter_config.json 自动读取")
    parser.add_argument("--output_path", type=str, default=None,
                        help="合并后模型保存路径，默认覆盖 adapter_path")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.adapter_path

    # 检查 adapter 文件是否存在且有效
    adapter_file = os.path.join(args.adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(args.adapter_path, "adapter_model.bin")
    if not os.path.exists(adapter_file):
        print(f"ERROR: 未找到 adapter 文件: {args.adapter_path}")
        sys.exit(1)

    size_mb = os.path.getsize(adapter_file) / (1024 * 1024)
    print(f"Adapter 文件大小: {size_mb:.1f} MB")
    if size_mb < 1.0:
        print("ERROR: adapter 文件过小 (<1MB)，可能是 ZeRO-3 保存的空文件，中止合并")
        sys.exit(1)

    # 确定基座模型路径
    base_model_path = args.base_model
    if base_model_path is None:
        peft_config = PeftConfig.from_pretrained(args.adapter_path)
        base_model_path = peft_config.base_model_name_or_path
        print(f"从 adapter_config.json 读取基座模型: {base_model_path}")

    print(f"加载基座模型: {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"加载 adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("合并 LoRA 权重...")
    merged = model.merge_and_unload()

    print(f"保存合并后模型到: {args.output_path}")
    merged.save_pretrained(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    for leftover in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        p = os.path.join(args.output_path, leftover)
        if os.path.exists(p):
            os.remove(p)
            print(f"  已删除残留 adapter 文件: {leftover}")

    print("合并完成")


if __name__ == "__main__":
    main()
