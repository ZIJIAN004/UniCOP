#!/usr/bin/env python
"""
diag_parse_rate.py — 诊断 RL rollout parse 率腰斩 (eval~1.0 但 RL~0.5)。

假设要分开验证的三个变量:
  (A) 截断:    max_tokens=3584 (RL 训练值) vs 6144 (eval 值) → instruct think 太长被截
  (B) 采样:    T=0.7/top_p=0.8/top_k=20 (RL) vs greedy (eval bo1)
  (C) 并发:    本脚本单序列离线生成, 天然排除 vLLM prefix-caching 抢占 (踩坑 #32)

做法: 用 seed=42 生成跟训练**完全相同**的 CVRP20 实例, 用 train.py 一模一样的
      prompt 构造 (build_prompt + apply_chat_template(add_generation_prompt=True)),
      离线 vLLM 在多组 (max_tokens, 采样) 配置下各跑一遍, 对每条统计:
        - 实际 completion token 数 / 是否触 max_tokens (finish_reason=='length')
        - 是否含 </think>
        - parse_multi_route 是否成功 (跟 terminal_reward / COMPLETION_DUMP 同一函数)
      最后打印每组的 truncation_rate / has_close_rate / parse_rate, 对号入座。

用法 (在 UniCOP-Reason-Mask 目录下, conda activate unicop 后):
  source ../paths.sh   # 取 MODEL/采样参数(可选, 下面也有默认)
  CUDA_VISIBLE_DEVICES=<空闲卡> python diag_parse_rate.py \
      --model "$DISTILL_DIR/output_sft_qwen3_instruct_template_cvrp20/final_model" \
      --num 32

  # 想跟 thinking 模型对照, 把 --model 换成 output_sft_qwen3_template_cvrp20/final_model
"""
import argparse
import numpy as np

from problems.cvrp import CVRP
from utils.parse import parse_multi_route


def build_prompts(num: int, n: int, tokenizer):
    """seed=42 复现训练实例, 用 train.py 同一套构造 chat_text。"""
    prob = CVRP()
    rng = np.random.default_rng(seed=42)          # 跟 build_dataset(seed=config.data_seed=42) 一致
    insts, prompts = [], []
    for _ in range(num):
        inst = prob.generate_instance(n, rng)
        msgs = prob.build_prompt(inst)
        chat_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        insts.append(inst)
        prompts.append(chat_text)
    return insts, prompts


def tally(label, outputs, insts, max_tokens):
    n_total = len(outputs)
    n_trunc = n_has_close = n_parse = 0
    comp_lens = []
    for out, inst in zip(outputs, insts):
        o = out.outputs[0]
        txt = o.text
        clen = len(o.token_ids)
        comp_lens.append(clen)
        is_trunc = (o.finish_reason == "length") or (clen >= max_tokens - 1)
        has_close = "</think>" in txt
        parse_ok = parse_multi_route(txt, inst["n"]) is not None
        n_trunc += is_trunc
        n_has_close += has_close
        n_parse += parse_ok
    comp_lens = np.array(comp_lens)
    print(f"\n[{label}]  max_tokens={max_tokens}")
    print(f"  样本数            : {n_total}")
    print(f"  completion len    : mean={comp_lens.mean():.0f} "
          f"p95={np.percentile(comp_lens, 95):.0f} max={comp_lens.max()}")
    print(f"  truncation_rate   : {n_trunc / n_total:.3f}   "
          f"({n_trunc}/{n_total} 触 max_tokens)")
    print(f"  has</think>_rate  : {n_has_close / n_total:.3f}")
    print(f"  parse_rate        : {n_parse / n_total:.3f}   "
          f"({n_parse}/{n_total} parse_multi_route 成功)")
    return n_parse / n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num", type=int, default=32)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    # RL 训练采样 (paths.sh instruct 分支)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    insts, prompts = build_prompts(args.num, args.n, tokenizer)
    p_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    print(f"prompt token: mean={np.mean(p_lens):.0f} max={max(p_lens)}  (max_prompt_length=1280)")

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    # 四组对照: 拆开 截断 × 采样
    configs = [
        ("RL 复刻: 采样 + 3584",  3584, args.temperature, args.top_p, args.top_k),
        ("放宽长度: 采样 + 6144", 6144, args.temperature, args.top_p, args.top_k),
        ("eval 复刻: greedy + 6144", 6144, 0.0, 1.0, -1),
        ("greedy + 3584",          3584, 0.0, 1.0, -1),
    ]
    results = {}
    for label, mt, temp, tp, tk in configs:
        sp = SamplingParams(
            n=1, temperature=temp, top_p=tp,
            top_k=tk, max_tokens=mt,
        )
        outs = llm.generate(prompts, sp)
        results[label] = tally(label, outs, insts, mt)

    print("\n" + "=" * 60)
    print("  结论速读")
    print("=" * 60)
    print("  若 [采样+3584] parse 低、[采样+6144] parse 高  → 截断 (扩 max_completion_length)")
    print("  若 6144 下采样仍低、greedy 才高               → instruct 采样鲁棒性 (降温/多采样)")
    print("  若 3584 下离线 parse 就高(跟 RL 内 ~0.5 矛盾) → 是 RL 内并发抢占 (踩坑 #32)")
    for k, v in results.items():
        print(f"    {k:28s}: parse_rate={v:.3f}")


if __name__ == "__main__":
    main()
