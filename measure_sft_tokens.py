#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""measure_sft_tokens.py — 服务器端单独测 SFT 样本 token 长度 (真 tokenizer + 真数据)。

精确复刻两臂的 prompt+completion 拼接, 与训练脚本逐行一致:
  - UniCOP 思维臂: train_sft_stage2.load_sft_dataset
      prompt = chat_template(system,user,add_generation_prompt) [+ "<think>\\n" 若模板不带]
      completion = output 去开头<think> + eos
  - FOARL 无推理臂: train_sft_foarl.load_foarl_dataset
      prompt = chat_template(FOARL_PREAMBLE, instruction+"\\n\\n"+input, add_generation_prompt)
      completion = output + eos
长度统计口径与训练脚本一致 (len(tokenizer.encode(text)), add_bos_token=False)。

用法 (zhihan, conda activate unicop 后):
  source paths.sh   # 取 $BASE_MODEL (=Qwen3-4B-Instruct-2507)
  python measure_sft_tokens.py \
      --model "$BASE_MODEL" \
      --unicop_data UniCOP-Distill/data/chains_template_cvrp100.jsonl \
      --foarl_data  FOARL/data/foarl_cvrp100.jsonl \
      --max_length 8192 --max_output_length 4096    # 传 cvrp20 旧默认, 看会丢多少

  # 只测一个臂就只传对应 --xxx_data; 默认只测前 1000 条 (--limit 0 测全量)
"""
import argparse
import json
import numpy as np
from transformers import AutoTokenizer

# 必须与 FOARL/train_sft_foarl.py 的 FOARL_PREAMBLE 完全一致
FOARL_PREAMBLE = (
    "Below is an instruction describing a combinatorial optimization problem. "
    "It is paired with an input that provides the data of the instance. "
    "Your task is to produce a feasible solution that optimizes (minimizes or maximizes) "
    "the given objective."
)
# 与 train_sft_stage2.py 一致的 posthoc 标记 (template 链通常不含, strip 为 no-op)
_POSTHOC_SYS = "\n\nYour output MUST start with <think>"
_POSTHOC_USR = "\n\nTarget solution ("


def _stats(arr):
    a = np.array(arr)
    return (f"n={len(a)}  mean={a.mean():.0f}  p50={np.percentile(a,50):.0f}  "
            f"p95={np.percentile(a,95):.0f}  p99={np.percentile(a,99):.0f}  max={a.max():.0f}")


def _recommend(comp, total):
    c = np.percentile(comp, 99) * 1.05
    t = np.percentile(total, 99) * 1.05
    ceil128 = lambda x: int(np.ceil(x / 128) * 128)
    return ceil128(c), ceil128(t)


def _report(name, P, C, T, max_length, max_output_length):
    C, T = np.array(C), np.array(T)
    print(f"\n{'='*64}\n  [{name}]  样本 {len(P)} 条\n{'='*64}")
    print(f"  prompt     : {_stats(P)}")
    print(f"  completion : {_stats(C)}")
    print(f"  total      : {_stats(T)}")
    drop_out = (C > max_output_length).mean() * 100
    drop_tot = (T > max_length).mean() * 100
    print(f"  --max_output_length={max_output_length}: completion 超长被丢 {drop_out:.1f}%")
    print(f"  --max_length={max_length}: 整条被丢 {drop_tot:.1f}%")
    rec_o, rec_t = _recommend(C, T)
    print(f"  建议 (p99×1.05 取整128): --max_output_length {rec_o}  --max_length {rec_t}")


def measure_unicop(path, tok, max_length, max_output_length, limit):
    probe = tok.apply_chat_template(
        [{"role": "system", "content": "p"}, {"role": "user", "content": "p"}],
        tokenize=False, add_generation_prompt=True)
    ends_think = probe.rstrip().endswith("<think>")
    print(f"  [UniCOP] chat_template 末尾带<think>: {ends_think} "
          f"({'用模板自带' if ends_think else '手动补<think>\\n'})")
    P, C, T = [], [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            system = r["prompt"]["system"].split(_POSTHOC_SYS)[0]
            user = r["prompt"]["user"].split(_POSTHOC_USR)[0]
            output = r["output"]
            if not output.strip():
                continue
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
            if not ends_think:
                prompt += "<think>\n"
            out = output.lstrip()
            if out.startswith("<think>"):
                out = out[len("<think>"):].lstrip("\n")
            completion = out + tok.eos_token
            pl = len(tok.encode(prompt))
            cl = len(tok.encode(completion))
            P.append(pl); C.append(cl); T.append(pl + cl)
    _report("UniCOP 思维臂 (stride=5)", P, C, T, max_length, max_output_length)


def measure_foarl(path, tok, max_length, max_output_length, limit):
    P, C, T = [], [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            instruction = r.get("instruction", "")
            inp = r.get("input", "")
            output = r.get("output", "")
            if not output.strip():
                continue
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": FOARL_PREAMBLE},
                 {"role": "user", "content": f"{instruction}\n\n{inp}"}],
                tokenize=False, add_generation_prompt=True)
            completion = output + tok.eos_token
            pl = len(tok.encode(prompt))
            cl = len(tok.encode(completion))
            P.append(pl); C.append(cl); T.append(pl + cl)
    _report("FOARL 无推理臂", P, C, T, max_length, max_output_length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tokenizer 路径 (=Qwen3-4B-Instruct-2507)")
    ap.add_argument("--unicop_data", default=None, help="chains_template_cvrp100.jsonl")
    ap.add_argument("--foarl_data", default=None, help="foarl_cvrp100.jsonl")
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--max_output_length", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=1000, help="只测前 N 条 (默认 1000; 0=全量)")
    args = ap.parse_args()

    if not args.unicop_data and not args.foarl_data:
        ap.error("至少传一个 --unicop_data 或 --foarl_data")

    print(f"加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if getattr(tok, "add_bos_token", False):
        tok.add_bos_token = False  # 与训练脚本一致, 防双 BOS 影响计数
    print(f"  eos={tok.eos_token!r} (id={tok.eos_token_id})")

    if args.unicop_data:
        measure_unicop(args.unicop_data, tok, args.max_length, args.max_output_length, args.limit)
    if args.foarl_data:
        measure_foarl(args.foarl_data, tok, args.max_length, args.max_output_length, args.limit)


if __name__ == "__main__":
    main()
