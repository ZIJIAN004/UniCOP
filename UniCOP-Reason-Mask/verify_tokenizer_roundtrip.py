"""verify tokenizer decode-reencode 一致性 (risk #5 排查)

在远程跑:
    cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
    source /homes/zhuoyi/.bashrc && conda activate unicop
    python verify_tokenizer_roundtrip.py

或者把生成 token 序列从 vLLM log 抓出来本地跑.

测试 5 种 input:
  1. 简单 CVRP-like 文本
  2. 实际 SFT 生成 sample (从 distill 数据集拿)
  3. 含 [R1,2] 等 bracket 标记
  4. 含数字密集 "0.563" "select 13" 等
  5. 含特殊 token (EOS) 后再 skip_special

输出:
  - len(原 ids) vs len(re-tokenize ids)
  - 前 5 个不一致位置
  - PRM segment 边界 (典型 char range) 映射准确率
"""
import sys
from pathlib import Path

# 让 utils.* 能 import
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer

MODEL = "/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/output_sft_hybrid_cvrp20/final_model"

# 测试样本: 模拟实际 completion 的关键片段
SAMPLES = [
    # 1. 简单 ASCII
    "Route1:0->1->2->3->0",

    # 2. 典型 think segment 段 (来自实际 7490 log step 10)
    "3(d=0.563,dem=0.07,cap0.93),2(d=0.590,dem=0.13,cap0.87),7(d=0.700,dem=0.23,cap0.77),...select13[R4,2]cap=1.00-0.07=0.93|feasible:7(d=0.142,dem=0.23,cap0.70),14(d=0.220,dem=0.23,cap0.70),2(d=0.567,dem=0.13,cap0.80)select14[R4,3]",

    # 3. 含 <think> </think> 特殊片段
    "<think>Step 1: select 5</think>Route1:0->5->0",

    # 4. 含 unicode 数字密集
    "[R1,1] cap=1.00 → select 10\n[R1,2] cap=0.85",

    # 5. 长 ASCII 段 (typical 段长度 ~80-200 chars)
    "[R3,1]cap=1.00|feasible:1(d=0.234,dem=0.13,cap0.87),9(d=0.456,dem=0.20,cap0.80),11(d=0.678,dem=0.17,cap0.83),...→select1[R3,2]cap=0.87-0.13=0.74|feasible:9(d=0.301,dem=0.20,cap0.54),11(d=0.487,dem=0.17,cap0.57)→select9",
]


def test_roundtrip(tokenizer, text: str, idx: int):
    """对一段文本测试 encode → decode → re-encode 的一致性."""
    print(f"\n{'='*70}")
    print(f"Sample #{idx}: len={len(text)} chars")
    print(f"text preview: {text[:120]}...")
    print(f"{'='*70}")

    # 1. encode 原始文本 → token ids
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    orig_ids = enc.input_ids
    orig_offsets = enc.offset_mapping

    # 2. decode → 文本 → re-encode
    decoded = tokenizer.decode(orig_ids, skip_special_tokens=True)
    re_enc = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    re_ids = re_enc.input_ids
    re_offsets = re_enc.offset_mapping

    # 3. 对比
    len_match = len(orig_ids) == len(re_ids)
    text_match = text == decoded
    ids_match = orig_ids == re_ids

    print(f"  原 ids 数:    {len(orig_ids)}")
    print(f"  re-encode 数: {len(re_ids)}")
    print(f"  长度一致:     {'[OK]' if len_match else '[FAIL]'}")
    print(f"  decode == 原: {'[OK]' if text_match else '[FAIL]'}")
    print(f"  ids 全部一致: {'[OK]' if ids_match else '[FAIL]'}")

    if not text_match:
        # 找前 3 个差异字符
        for i in range(min(len(text), len(decoded))):
            if text[i] != decoded[i]:
                print(f"    char diff @ {i}: orig {text[i]!r} vs decoded {decoded[i]!r}")
                if i > 5:
                    print(f"    context orig:    ...{text[max(0,i-5):i+10]!r}")
                    print(f"    context decoded: ...{decoded[max(0,i-5):i+10]!r}")
                break
        if len(text) != len(decoded):
            print(f"    长度不同: orig {len(text)} vs decoded {len(decoded)}")

    if not ids_match:
        # 找前 3 个差异位置
        n = 0
        for i in range(min(len(orig_ids), len(re_ids))):
            if orig_ids[i] != re_ids[i]:
                orig_tok = tokenizer.decode([orig_ids[i]])
                re_tok = tokenizer.decode([re_ids[i]])
                print(f"    diff @ token {i}: orig {orig_ids[i]} ({orig_tok!r}) vs re {re_ids[i]} ({re_tok!r})")
                n += 1
                if n >= 3:
                    break

    # 4. PRM segment 模拟: 随机选一段 char range, 看 char→token 映射在原 vs re 是否一致
    if len(text) > 50:
        # 模拟 PRM segment (类似 "select14[R4,3]cap=...")
        seg_starts = [0, len(text) // 4, len(text) // 2, 3 * len(text) // 4]
        seg_len = 30
        print(f"\n  模拟 PRM segment 边界映射 (seg_len={seg_len}):")
        for cs in seg_starts:
            ce = min(cs + seg_len, len(text))
            tok_s_orig, tok_e_orig = char_to_token_range(cs, ce, orig_offsets)
            tok_s_re, tok_e_re = char_to_token_range(cs, ce, re_offsets)
            match = tok_s_orig == tok_s_re and tok_e_orig == tok_e_re
            print(f"    char[{cs}:{ce}] → orig tok[{tok_s_orig}:{tok_e_orig}] vs "
                  f"re tok[{tok_s_re}:{tok_e_re}]  {'[OK]' if match else '[FAIL]'}")


def char_to_token_range(char_start, char_end, offset_mapping):
    """复刻 grpo_prm_trainer._char_to_token_range."""
    tok_start = None
    tok_end = None
    for t_idx, (cs, ce) in enumerate(offset_mapping):
        if ce <= char_start:
            continue
        if cs >= char_end:
            break
        if tok_start is None:
            tok_start = t_idx
        tok_end = t_idx + 1
    return tok_start, tok_end


def main():
    print(f"加载 tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"  tokenizer 类型: {type(tokenizer).__name__}")
    print(f"  is_fast:        {tokenizer.is_fast}")
    print(f"  vocab_size:     {tokenizer.vocab_size}")
    print(f"  eos_token:      {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token:      {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    if not tokenizer.is_fast:
        print("[WARN] 不是 fast tokenizer, offset_mapping 不可用!")
        return

    for i, sample in enumerate(SAMPLES, 1):
        test_roundtrip(tokenizer, sample, i)

    # 真实场景: 模拟 vLLM 生成完整 completion (含 EOS 末尾)
    print(f"\n\n{'#'*70}")
    print("# 真实 vLLM 场景模拟: encode → 加 EOS → decode(skip_special) → re-encode")
    print(f"{'#'*70}")
    text = SAMPLES[1]  # 典型 think segment
    enc = tokenizer(text, add_special_tokens=False)
    ids_with_eos = enc.input_ids + [tokenizer.eos_token_id]  # 模拟 vLLM 末尾加 EOS
    decoded = tokenizer.decode(ids_with_eos, skip_special_tokens=True)
    re_enc = tokenizer(decoded, add_special_tokens=False)
    print(f"  原 ids (含 EOS): {len(ids_with_eos)}")
    print(f"  re-encode:       {len(re_enc.input_ids)}")
    print(f"  长度差:          {len(ids_with_eos) - len(re_enc.input_ids)}  (期望 = 1 因为 EOS 被 skip)")

if __name__ == "__main__":
    main()
