#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_no_repeat_ngram.py — 实跑 vLLM, 确定性验证 evaluate.py 里的 NoRepeatNGramLogitsProcessor 真生效。

原理 (与模型是否"循环"无关的强对照):
  - 任何正常文本都含重复的短 n-gram。所以用 **小 n=3**:
      关处理器 → 输出里有一堆重复 3-gram;
      开 n=3   → 处理器保证输出里"重复 3-gram 数 = 0"(这是它的不变量)。
  - 若开 n=3 后仍有重复 3-gram → 处理器没被 vLLM 调用 / -inf 没拦住 → FAIL。
  - 再跑生产档 n=20 确认不报错且同样满足"无重复 20-gram"。

zhuoyi SLURM 用法: sbatch submit_smoke_nrn.sh   (本脚本由该 submit 调起, 勿在登录节点直跑)
"""
import os
import sys


def count_repeated_ngrams(token_ids, n):
    """返回输出中"重复出现(>=2 次)的 n-gram 实例数"(出现 k 次计 k-1)。"""
    L = len(token_ids)
    if L < n:
        return 0
    seen = set()
    dup = 0
    for i in range(L - n + 1):
        g = tuple(token_ids[i:i + n])
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "")
    if not model_path:
        print("用法: python smoke_no_repeat_ngram.py <model_path>  (或 export MODEL=...)")
        sys.exit(2)
    gpu_mem = float(os.environ.get("GPU_MEM", "0.8"))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate import NoRepeatNGramLogitsProcessor   # ← 用真处理器, 不复制逻辑
    from vllm import LLM, SamplingParams

    print(f"加载 vLLM: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=gpu_mem, max_model_len=2048,
              trust_remote_code=True, enforce_eager=True)
    tok = llm.get_tokenizer()

    # 一个让模型多写、容易出现重复短 n-gram 的 prompt (greedy 解码 → 可复现)
    msgs = [{"role": "user", "content":
             "List the numbers from 1 to 10. Then write several vehicle routes "
             "in the form 'Route k: 0 -> a -> b -> 0'. Make the answer long."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    base = dict(max_tokens=400, temperature=0.0, n=1, skip_special_tokens=False)

    print("\n[1] 关闭 no-repeat-ngram, greedy 生成...")
    off = llm.generate([prompt], SamplingParams(**base))[0].outputs[0]
    off_ids = list(off.token_ids)

    print("[2] 开 no_repeat_ngram_size=3 (小 n 强对照)...")
    on3 = llm.generate([prompt], SamplingParams(
        **base, logits_processors=[NoRepeatNGramLogitsProcessor(3)]))[0].outputs[0]
    on3_ids = list(on3.token_ids)

    print("[3] 开 no_repeat_ngram_size=20 (生产档, 确认不报错)...")
    on20 = llm.generate([prompt], SamplingParams(
        **base, logits_processors=[NoRepeatNGramLogitsProcessor(20)]))[0].outputs[0]
    on20_ids = list(on20.token_ids)

    r_off3 = count_repeated_ngrams(off_ids, 3)
    r_on3 = count_repeated_ngrams(on3_ids, 3)
    r_on20 = count_repeated_ngrams(on20_ids, 20)

    print("\n" + "=" * 64)
    print(f"OFF       : len={len(off_ids):4d}  重复 3-gram 数 = {r_off3}")
    print(f"ON  n=3   : len={len(on3_ids):4d}  重复 3-gram 数 = {r_on3}    (必须 = 0)")
    print(f"ON  n=20  : len={len(on20_ids):4d}  重复 20-gram 数 = {r_on20}   (必须 = 0)")
    print("-" * 64)
    ok = True
    if r_on3 != 0:
        print("❌ FAIL: 开 n=3 后仍有重复 3-gram → 处理器未生效(vLLM 没调用 / -inf 没拦住)")
        ok = False
    if r_on20 != 0:
        print("❌ FAIL: 开 n=20 后仍有重复 20-gram → 处理器未生效")
        ok = False
    if r_off3 == 0:
        print("⚠️ 对照偏弱: OFF 未出现重复 3-gram(此 prompt/模型没触发), 但 ON=0 仍证明不变量成立")
    elif ok:
        print(f"✅ 对照成立: OFF 有 {r_off3} 个重复 3-gram, 开 n=3 后归零 → 处理器确实生效且正确")
    print("=" * 64)
    print("✅✅ PASS" if ok else "❌❌ FAIL")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
