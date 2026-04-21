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

    # 先读 adapter 旁的 tokenizer, 判断 SFT 训练时是否新加了 pad_token
    # (train_sft.py 在 candidate 全 miss 时会 add_special_tokens + resize_token_embeddings,
    # 但 LoRA target_modules 不含 embed_tokens, adapter 不保存这个 resize。
    # merge 时必须先把 base 也 resize, 否则 merged model vocab_size 比 tokenizer 小,
    # pad_token_id 越界, 下游 from_pretrained 就崩。)
    tok_src = args.adapter if os.path.isfile(
        os.path.join(args.adapter, "tokenizer_config.json")
    ) else base_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    print(f"[1/4] Loading base model: {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device,
    )

    target_vocab = len(tokenizer)
    base_vocab = base.get_input_embeddings().num_embeddings
    if target_vocab > base_vocab:
        print(f"[2/4] resize base embedding {base_vocab} → {target_vocab} "
              f"(tokenizer 比 base 多 {target_vocab - base_vocab} 个 token, 对齐)")
        base.resize_token_embeddings(target_vocab)
    elif target_vocab < base_vocab:
        print(f"[2/4] WARN tokenizer ({target_vocab}) < base ({base_vocab}), 保留 base 大小")
    else:
        print(f"[2/4] tokenizer 与 base vocab 一致 ({base_vocab}), 无需 resize")

    print(f"[3/4] Loading adapter: {args.adapter}")
    lora = PeftModel.from_pretrained(base, args.adapter)

    print("[4/4] Merging + saving...")
    merged = lora.merge_and_unload()
    os.makedirs(args.output, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    print(f"[DONE] 合并完成: {args.output}")


if __name__ == "__main__":
    main()
