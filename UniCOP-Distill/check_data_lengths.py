"""检查 chains_template_cvrp20.jsonl 的 prompt / completion / total token 长度分布"""

import json
import numpy as np
from transformers import AutoTokenizer

DATA_PATH = "data/chains_template_cvrp20.jsonl"
MODEL_PATH = "/homes/zhuoyi/zijianliu/models/DeepSeek-R1-Distill-Qwen-7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

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
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(output_text))
        total = prompt_tokens + completion_tokens

        prompt_lens.append(prompt_tokens)
        completion_lens.append(completion_tokens)
        total_lens.append(total)

prompt_lens = np.array(prompt_lens)
completion_lens = np.array(completion_lens)
total_lens = np.array(total_lens)

print(f"样本总数: {len(total_lens)}")
print()
print(f"{'':15s} {'min':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p90':>8s} {'p95':>8s} {'max':>8s} {'mean':>8s}")
for name, arr in [("prompt", prompt_lens), ("completion", completion_lens), ("total", total_lens)]:
    print(f"{name:15s} {int(np.min(arr)):8d} {int(np.percentile(arr,25)):8d} {int(np.percentile(arr,50)):8d} "
          f"{int(np.percentile(arr,75)):8d} {int(np.percentile(arr,90)):8d} {int(np.percentile(arr,95)):8d} "
          f"{int(np.max(arr)):8d} {np.mean(arr):8.0f}")

for threshold in [2048, 3072, 4096, 6144, 8192]:
    n = np.sum(total_lens <= threshold)
    print(f"  total <= {threshold}: {n}/{len(total_lens)} ({100*n/len(total_lens):.1f}%)")
