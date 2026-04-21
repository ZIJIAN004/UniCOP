"""Merge LoRA adapter into base model, save merged model + tokenizer.

用法:
    python tools/merge_lora.py \
        --adapter /path/to/adapter_dir \
        --output  /path/to/merged_model

不填 --base 时,自动从 adapter 的 adapter_config.json 读 base_model_name_or_path。
"""
import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base, save merged model")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory (含 adapter_config.json)")
    parser.add_argument("--base", default=None,
                        help="Base model 路径,不填则从 adapter_config.json 读取")
    parser.add_argument("--output", required=True, help="合并后模型保存目录")
    parser.add_argument("--device", default="cpu",
                        help="load 设备: cpu (安全,慢) / cuda (快,占 GPU 显存)")
    args = parser.parse_args()

    if not os.path.isdir(args.adapter):
        print(f"[FAIL] adapter 目录不存在: {args.adapter}", file=sys.stderr)
        sys.exit(1)

    adapter_cfg_path = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.isfile(adapter_cfg_path):
        print(f"[FAIL] 不是 LoRA adapter: {adapter_cfg_path} 缺失", file=sys.stderr)
        sys.exit(1)

    base_path = args.base
    if base_path is None:
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            base_path = json.load(f).get("base_model_name_or_path")
    if not base_path or not os.path.isdir(base_path):
        print(f"[FAIL] 无效 base 路径: {base_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Loading base model: {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device,
    )

    print(f"[2/3] Loading adapter: {args.adapter}")
    lora = PeftModel.from_pretrained(base, args.adapter)

    print("[3/3] Merging + saving...")
    merged = lora.merge_and_unload()
    os.makedirs(args.output, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)

    # tokenizer 优先从 adapter 目录读(可能含训练时追加的 pad_token),
    # 兜底从 base
    tok_src = args.adapter if os.path.isfile(
        os.path.join(args.adapter, "tokenizer_config.json")
    ) else base_path
    AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True).save_pretrained(args.output)

    print(f"[DONE] 合并完成: {args.output}")


if __name__ == "__main__":
    main()
