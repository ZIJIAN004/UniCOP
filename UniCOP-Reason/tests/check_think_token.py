"""快速诊断：看看 DeepSeek-R1-Distill-Qwen 的 </think> 到底是怎么编码的。"""

import sys
from transformers import AutoTokenizer

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/Data04/yangzhihan/lzj/UniCOP-Reason/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"Model: {MODEL_PATH}")
print(f"Vocab size: {tok.vocab_size}")
print(f"Special tokens: {tok.special_tokens_map}")
print()

# 方式 1：独立 encode
ids1 = tok.encode("</think>", add_special_tokens=False)
print(f"encode('</think>', add_special=False) = {ids1}")
print(f"  decoded back: {tok.decode(ids1)!r}")
print(f"  num tokens: {len(ids1)}")
print()

# 方式 2：上下文内 encode
probe = "Some reasoning here.\n</think>\n\nRoute: 0 -> 1 -> 0"
ids2 = tok.encode(probe, add_special_tokens=False)
print(f"encode 带上下文的完整串:")
print(f"  ids = {ids2}")
for i, t in enumerate(ids2):
    print(f"    [{i}] {t:>7} = {tok.decode([t])!r}")
print()

# 方式 3：检查 </think> 是否在特殊 token 列表里
print("Added / special tokens 检查:")
if hasattr(tok, "added_tokens_decoder"):
    for tid, tk in tok.added_tokens_decoder.items():
        if "think" in str(tk).lower():
            print(f"  → {tid}: {tk}")
print()

# 结论
if len(ids1) == 1:
    print(f"结论：</think> 是单 token (id={ids1[0]})，"
          "ThinkOnlyDRYProcessor 的 _contains_subseq 能正常工作 ✓")
else:
    print(f"结论：</think> 被切成 {len(ids1)} 个 token，需要验证模型真实生成时"
          f"是否真的产出这组 token 序列，否则门控失效 ✗")
