"""
验证 DeepSeek-R1-Distill-Qwen-7B tokenizer 的 BOS 行为和 response_template 检测。
用于排查双 BOS 问题和 DataCollatorForCompletionOnlyLM 的 response_template 匹配。

运行: python verify_tokenizer_bos.py --model <model_path>
"""

import argparse
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/home/ntu/lzj/Model/model/DeepSeek-R1-Distill-Qwen-7B")
    args = parser.parse_args()

    print(f"加载 tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\n{'='*70}")
    print(f"1. Tokenizer 基本配置")
    print(f"{'='*70}")
    print(f"  add_bos_token:  {getattr(tokenizer, 'add_bos_token', 'N/A')}")
    print(f"  add_eos_token:  {getattr(tokenizer, 'add_eos_token', 'N/A')}")
    print(f"  bos_token:      {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    print(f"  eos_token:      {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token:      {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # ── 2. chat_template 输出 ──
    print(f"\n{'='*70}")
    print(f"2. chat_template 渲染结果")
    print(f"{'='*70}")
    messages = [
        {"role": "system", "content": "You are an expert."},
        {"role": "user",   "content": "Solve this problem."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    print(f"  渲染文本:\n    {rendered!r}")
    print(f"  文本开头是 BOS: {rendered.startswith(tokenizer.bos_token)}")

    # ── 3. 双 BOS 验证 ──
    print(f"\n{'='*70}")
    print(f"3. 双 BOS 验证")
    print(f"{'='*70}")

    # 默认 encode（add_special_tokens=True）
    ids_default = tokenizer.encode(rendered)
    # 不加特殊 token
    ids_no_special = tokenizer.encode(rendered, add_special_tokens=False)
    # 直接 tokenizer() 调用（TRL 内部用的方式）
    ids_call = tokenizer(rendered)["input_ids"]

    bos_id = tokenizer.bos_token_id
    print(f"  BOS token id: {bos_id}")
    print()
    print(f"  encode(text) 默认 (add_special_tokens=True):")
    print(f"    前 5 个 token: {ids_default[:5]}")
    print(f"    开头连续 BOS 数: {sum(1 for t in ids_default if t == bos_id) - sum(1 for i, t in enumerate(ids_default) if t == bos_id and i > 0 and ids_default[i-1] != bos_id)}")
    count_bos_default = 0
    for t in ids_default:
        if t == bos_id:
            count_bos_default += 1
        else:
            break
    print(f"    开头连续 BOS 数: {count_bos_default}")
    decoded_default = tokenizer.decode(ids_default[:10])
    print(f"    decode 前 10 token: {decoded_default!r}")
    print()

    print(f"  encode(text, add_special_tokens=False):")
    print(f"    前 5 个 token: {ids_no_special[:5]}")
    count_bos_no_special = 0
    for t in ids_no_special:
        if t == bos_id:
            count_bos_no_special += 1
        else:
            break
    print(f"    开头连续 BOS 数: {count_bos_no_special}")
    decoded_no_special = tokenizer.decode(ids_no_special[:10])
    print(f"    decode 前 10 token: {decoded_no_special!r}")
    print()

    print(f"  tokenizer(text) 即 __call__:")
    print(f"    前 5 个 token: {ids_call[:5]}")
    count_bos_call = 0
    for t in ids_call:
        if t == bos_id:
            count_bos_call += 1
        else:
            break
    print(f"    开头连续 BOS 数: {count_bos_call}")
    print()

    if count_bos_default > 1 or count_bos_call > 1:
        print(f"  ⚠️  双 BOS 确认! 原因: chat_template 已加 BOS + tokenizer add_bos_token=True 又加一个")
    elif count_bos_no_special > 1:
        print(f"  ⚠️  即使 add_special_tokens=False 也有双 BOS (chat_template 自身的问题)")
    else:
        print(f"  ✓  无双 BOS 问题")

    # ── 4. response_template 检测 ──
    print(f"\n{'='*70}")
    print(f"4. response_template 检测")
    print(f"{'='*70}")
    msgs = [{"role": "user", "content": "Hi"}]
    without_gp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    with_gp    = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gen_prompt = with_gp[len(without_gp):]
    print(f"  without add_generation_prompt: ...{without_gp[-30:]!r}")
    print(f"  with    add_generation_prompt: ...{with_gp[-30:]!r}")
    print(f"  差异 (generation_prompt):       {gen_prompt!r}")

    think_idx = gen_prompt.find("<think>")
    response_template = gen_prompt[:think_idx] if think_idx > 0 else gen_prompt
    print(f"  response_template (去 <think>): {response_template!r}")

    rt_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"  response_template token IDs:    {rt_ids}")
    print(f"  decode 验证:                    {tokenizer.decode(rt_ids)!r}")

    # 验证 response_template 能否在 tokenized text 中找到
    text_ids = ids_no_special  # 用不加特殊 token 的版本
    prompt_with_completion = rendered + "This is the completion." + tokenizer.eos_token
    full_ids = tokenizer.encode(prompt_with_completion, add_special_tokens=False)
    found = False
    for i in range(len(full_ids) - len(rt_ids) + 1):
        if full_ids[i:i+len(rt_ids)] == rt_ids:
            found = True
            print(f"  ✓  response_template 在完整序列中找到 (位置 {i})")
            break
    if not found:
        print(f"  ✗  response_template 在完整序列中未找到!")

    # 对比: 旧的硬编码 response_template
    old_rt = "<|im_start|>assistant\n"
    old_rt_ids = tokenizer.encode(old_rt, add_special_tokens=False)
    print(f"\n  旧 (硬编码) response_template: {old_rt!r}")
    print(f"  旧 token IDs:                  {old_rt_ids}")
    old_found = False
    for i in range(len(full_ids) - len(old_rt_ids) + 1):
        if full_ids[i:i+len(old_rt_ids)] == old_rt_ids:
            old_found = True
            break
    print(f"  旧 template 能找到:            {old_found}")

    # ── 5. pad_token 候选检查 ──
    print(f"\n{'='*70}")
    print(f"5. pad_token 候选检查")
    print(f"{'='*70}")
    candidates = ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>"]
    for cand in candidates:
        tid = tokenizer.convert_tokens_to_ids(cand)
        in_vocab = isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id
        # 也检查 unk_token_id
        is_unk = (tid == getattr(tokenizer, 'unk_token_id', None))
        print(f"  {cand!r:25s} → id={tid!r:10s}  在词表中={in_vocab}  是UNK={is_unk}")

    # ── 6. 总结 ──
    print(f"\n{'='*70}")
    print(f"总结")
    print(f"{'='*70}")
    print(f"  双 BOS 问题:       {'是' if count_bos_default > 1 or count_bos_call > 1 else '否'}")
    print(f"  新 response_template 可用: {'是' if found else '否'}")
    print(f"  旧 response_template 可用: {'是' if old_found else '否'}")
    if count_bos_default > 1:
        print(f"\n  修复建议: 在 SFTTrainer 之前设置 tokenizer.add_bos_token = False")
        print(f"           (因为 apply_chat_template 已经在文本开头加了 BOS)")


if __name__ == "__main__":
    main()
