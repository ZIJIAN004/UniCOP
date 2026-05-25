"""
远程兼容性自检：用 trainer **真实的** load_sft_dataset + **真 tokenizer** 跑一遍
reverse 数据，确认 token 长度过滤 / chat_template / response_template 都正常，
无样本因超长被静默丢弃。这是训练前的权威兼容性测试（本地无模型/tokenizer 测不了）。

须在 zhuoyi（有模型+unicop 环境）上运行：
    conda activate unicop
    python UniCOP-Distill/check_reverse_compat.py \
        --model /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_qwen3_template_cvrp20/final_model \
        --data  UniCOP-Distill/data/chains_reverse_cvrp20_10k.jsonl

判读：脚本末尾打印"最终训练样本数"。等于输入条数(10000) = 完全兼容；
少于则说明有样本 token 超 max_length(8192)/max_output_length(4096) 被过滤。
"""

import argparse
import os
import sys

from transformers import AutoTokenizer

# 让 import train_sft_stage2 生效（它在 stage2_reasoning/ 下）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "stage2_reasoning"))
from train_sft_stage2 import load_sft_dataset, _detect_response_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="reverse 数据 × trainer 加载器兼容性自检")
    ap.add_argument("--model", required=True, help="基座模型/ tokenizer 路径")
    ap.add_argument("--data", required=True, help="reverse chains jsonl")
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--max_output_length", type=int, default=4096)
    ap.add_argument("--expect", type=int, default=0,
                    help="期望样本数（如 10000）；>0 时据此判定是否有过滤")
    args = ap.parse_args()

    print(f"加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if getattr(tok, "add_bos_token", False):
        tok.add_bos_token = False

    # response_template 探测（与 trainer 同逻辑，验证 completion-only loss 切点可识别）
    rt = _detect_response_template(tok)
    rt_ids = tok.encode(rt, add_special_tokens=False)
    print(f"response_template: {rt!r}")
    print(f"response_template ids: {rt_ids}")
    if not rt_ids:
        print("⚠️ response_template 为空，completion-only loss 可能无法切分！")

    # 用 trainer 真实加载器跑（含 chat_template 渲染 + token 长度过滤 + 首条样本验证）
    print("\n>>> 调用 trainer 真实 load_sft_dataset ...")
    ds = load_sft_dataset(
        args.data, tok, args.max_length, args.max_output_length,
        filter_problems=["cvrp"], filter_sizes=[20],
    )

    print(f"\n>>> 最终训练样本数: {len(ds)}")
    if args.expect > 0:
        dropped = args.expect - len(ds)
        if dropped <= 0:
            print(f"✓ 完全兼容：无样本被过滤（期望 {args.expect}）")
        else:
            print(f"⚠️ 有 {dropped} 条被过滤（token 超 max_length={args.max_length} / "
                  f"max_output_length={args.max_output_length}）。"
                  f"如比例小可接受；要全留可调高 --max_length。")


if __name__ == "__main__":
    main()
