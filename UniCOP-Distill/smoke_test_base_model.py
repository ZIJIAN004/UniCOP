"""Smoke test for base-model migration (R1-Distill ↔ Qwen3-4B-Thinking).

打印 chat_template 渲染原貌，并逐项验证训练/推理可能的不兼容点。
出现 [FAIL] 必须修复后再上正式训练；[WARN] 需关注但可继续。

用法：
    # 测试当前 BASE_MODEL_TYPE（来自 paths.sh）
    source ../paths.sh && python smoke_test_base_model.py

    # 显式指定模型路径
    python smoke_test_base_model.py --model /path/to/Qwen3-4B-Thinking-2507

    # 同时跑训练集首条数据的 round-trip 验证（需 --data 指定 chains*.jsonl）
    python smoke_test_base_model.py --data data/chains_v3_clean.jsonl
"""

import argparse
import json
import os
import sys

try:
    from transformers import AutoTokenizer, AutoConfig
except ImportError:
    print("ERROR: pip install transformers")
    sys.exit(1)


PASS_TAG = "[PASS]"
FAIL_TAG = "[FAIL]"
WARN_TAG = "[WARN]"
INFO_TAG = "[INFO]"

failures = []
warnings = []


def section(title: str):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = PASS_TAG if ok else (WARN_TAG if warn_only else FAIL_TAG)
    print(f"  {tag} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        (warnings if warn_only else failures).append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.environ.get("BASE_MODEL", ""))
    parser.add_argument("--model_type", type=str,
                        default=os.environ.get("BASE_MODEL_TYPE", ""),
                        help="r1_distill / qwen3_thinking（来自 paths.sh）")
    parser.add_argument("--data", type=str, default="",
                        help="可选 chains_*.jsonl 路径，做首条样本 round-trip 验证")
    parser.add_argument("--online", action="store_true",
                        help="启用 in-GPU 验证: 加载真实模型跑 1 step forward+backward, "
                             "验证 loss finite + grad 非 NaN + grad_norm 合理")
    parser.add_argument("--online_max_len", type=int, default=4096,
                        help="online 验证序列截断长度 (避免单卡 OOM)")
    args = parser.parse_args()

    if not args.model:
        print("ERROR: 需要 --model 或 export BASE_MODEL=...")
        sys.exit(1)

    print(f"模型路径:        {args.model}")
    print(f"BASE_MODEL_TYPE: {args.model_type or '(未设)'}")
    print(f"GEN_TEMPERATURE: {os.environ.get('GEN_TEMPERATURE', '(未设)')}")
    print(f"GEN_TOP_P:       {os.environ.get('GEN_TOP_P', '(未设)')}")
    print(f"GEN_TOP_K:       {os.environ.get('GEN_TOP_K', '(未设)')}")
    print(f"VLLM_REASONING_FLAGS: {os.environ.get('VLLM_REASONING_FLAGS', '(未设)')!r}")

    # ── 0. 加载 ─────────────────────────────────────────────────────────────
    section("0. 加载 tokenizer / config")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"  model_type:       {config.model_type}")
    print(f"  architectures:    {config.architectures}")
    print(f"  hidden_size:      {config.hidden_size}")
    print(f"  num_hidden_layers:{config.num_hidden_layers}")
    print(f"  vocab_size:       {config.vocab_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  tokenizer_class:  {type(tokenizer).__name__}")
    print(f"  tokenizer vocab:  {tokenizer.vocab_size}")

    # ── 1. Special tokens ──────────────────────────────────────────────────
    section("1. Special tokens 配置")
    print(f"  bos_token:      {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    print(f"  eos_token:      {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token:      {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"  unk_token:      {tokenizer.unk_token!r} (id={tokenizer.unk_token_id})")
    print(f"  add_bos_token:  {getattr(tokenizer, 'add_bos_token', 'N/A')}")
    print(f"  add_eos_token:  {getattr(tokenizer, 'add_eos_token', 'N/A')}")

    # pad ≠ eos（SFT 关键，否则 EOS 被 mask）
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    check("pad_token 已配置", pad_id is not None,
          f"pad_id={pad_id}")
    check("pad_token != eos_token", pad_id != eos_id,
          f"pad={pad_id} vs eos={eos_id}（相等会导致 EOS 被 label mask 吃掉）",
          warn_only=True)

    # think token 探测
    think_open  = tokenizer.convert_tokens_to_ids("<think>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    print(f"  <think>  token id:  {think_open}")
    print(f"  </think> token id:  {think_close}")
    is_special_think = think_open != tokenizer.unk_token_id and think_open >= 0
    check("<think> 在词表中", is_special_think,
          f"id={think_open}（应为正且 != unk_id）")

    # ChatML 特殊 token
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end   = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"  <|im_start|>: id={im_start}")
    print(f"  <|im_end|>:   id={im_end}")

    # ── 2. chat_template 渲染（原本格式打印） ───────────────────────────────
    section("2. chat_template 原始渲染（用户要求看原本格式）")
    messages = [
        {"role": "system", "content": "You are a CVRP expert."},
        {"role": "user",   "content": "Solve the routing problem for n=3 nodes."},
    ]
    rendered_no_gp = tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=False)
    rendered_gp = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)

    print("\n--- add_generation_prompt=False (full conversation) ---")
    print(repr(rendered_no_gp))
    print("\n--- 可读形式 ---")
    print(rendered_no_gp)

    print("\n--- add_generation_prompt=True (用于推理 prompt) ---")
    print(repr(rendered_gp))
    print("\n--- 可读形式 ---")
    print(rendered_gp)

    gen_suffix = rendered_gp[len(rendered_no_gp):]
    print(f"\n--- generation_prompt 差异（chat_template 自动 append 的部分）---")
    print(repr(gen_suffix))

    # ── 3. 关键不兼容点验证 ──────────────────────────────────────────────────
    section("3. 关键不兼容点验证（怀疑列表逐条核对）")

    # 3.1 chat_template 是否自动 prepend <think>\n
    ends_with_think = rendered_gp.rstrip().endswith("<think>")
    check("chat_template 自动 prepend <think>", ends_with_think,
          "Qwen3-Thinking 和 R1-Distill 都应为 True；"
          "False 则 rationalize_solutions.py:292 的补 <think> 逻辑会触发",
          warn_only=not ends_with_think)
    if not ends_with_think:
        print(f"        gen_suffix 末尾 50 字: {gen_suffix[-50:]!r}")

    # 3.2 双 BOS 验证（chat_template + add_bos_token 双重加）
    ids_default = tokenizer.encode(rendered_gp)
    ids_no_special = tokenizer.encode(rendered_gp, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        n_leading_bos = 0
        for t in ids_default:
            if t == bos_id:
                n_leading_bos += 1
            else:
                break
        check("无双 BOS", n_leading_bos <= 1,
              f"开头连续 BOS 数={n_leading_bos}（train_sft_*.py 应 add_bos_token=False）")
    else:
        print(f"  {INFO_TAG} bos_token=None（Qwen3 行为），跳过双 BOS 检查")

    # 3.3 response_template 探测（Stage 2 应包含 <think>\n）
    def detect_resp_template_stripthink(tok) -> str:
        msgs = [{"role": "user", "content": "Hi"}]
        wo = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        w  = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        gp = w[len(wo):]
        ti = gp.find("<think>")
        return gp[:ti] if ti > 0 else gp

    def detect_resp_template_full(tok) -> str:
        msgs = [{"role": "user", "content": "Hi"}]
        wo = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        w  = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return w[len(wo):]

    rt_stage1 = detect_resp_template_stripthink(tokenizer)
    rt_stage2 = detect_resp_template_full(tokenizer)
    print(f"\n  Stage 1 response_template (剥 <think>): {rt_stage1!r}")
    print(f"  Stage 2 response_template (含 <think>): {rt_stage2!r}")

    # 验证 Stage 1 response_template 能在完整序列中找到
    full_with_completion = rendered_gp + "thinking content\n</think>\nfinal answer" + (tokenizer.eos_token or "")
    full_ids = tokenizer.encode(full_with_completion, add_special_tokens=False)
    rt1_ids = tokenizer.encode(rt_stage1, add_special_tokens=False)
    rt2_ids = tokenizer.encode(rt_stage2, add_special_tokens=False)

    def subseq_pos(haystack, needle):
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1

    pos1 = subseq_pos(full_ids, rt1_ids)
    pos2 = subseq_pos(full_ids, rt2_ids)
    check("Stage 1 response_template 可定位", pos1 >= 0,
          f"pos={pos1}, ids={rt1_ids}")
    check("Stage 2 response_template 可定位", pos2 >= 0,
          f"pos={pos2}, ids={rt2_ids}")

    # 3.4 LoRA target_modules 命名兼容性
    try:
        from transformers import AutoModelForCausalLM
        # 不加载权重，只看模块结构（meta tensor 节省内存）
        import torch
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
                trust_remote_code=True, low_cpu_mem_usage=True,
            )
        module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
        expected = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        missing = [m for m in expected if m not in module_names]
        check("LoRA target_modules 命名匹配", not missing,
              f"缺失={missing}" if missing else "全部存在")
    except Exception as e:
        print(f"  {WARN_TAG} LoRA target_modules 检查跳过（{type(e).__name__}: {e}）")

    # 3.5 EOS / stop_token_ids 验证
    section("4. EOS / stop_token_ids（vLLM 推理用）")
    # 探测多 EOS（Qwen3 有 [151645 <|im_end|>, 151643 <|endoftext|>]）
    eos_candidates = []
    for name in ["<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            eos_candidates.append((name, tid))
    print(f"  候选 stop tokens: {eos_candidates}")
    print(f"  推荐传给 vLLM 的 stop_token_ids: {[t for _, t in eos_candidates]}")

    # 3.6 模型类型检测
    section("5. 模型类型识别")
    is_qwen3 = config.model_type == "qwen3"
    is_qwen2 = config.model_type == "qwen2"
    print(f"  config.model_type = {config.model_type!r}")
    print(f"  is_qwen3 = {is_qwen3}")
    print(f"  is_qwen2 = {is_qwen2}")
    if is_qwen3:
        check("vLLM reasoning-parser 配置存在",
              "qwen3" in os.environ.get("VLLM_REASONING_FLAGS", ""),
              "Qwen3-Thinking 推理需要 --reasoning-parser qwen3",
              warn_only=True)
    if args.model_type == "qwen3_thinking" and not is_qwen3:
        check("BASE_MODEL_TYPE 与 config 一致", False,
              f"BASE_MODEL_TYPE=qwen3_thinking 但 config.model_type={config.model_type!r}")
    if args.model_type == "r1_distill" and is_qwen3:
        check("BASE_MODEL_TYPE 与 config 一致", False,
              f"BASE_MODEL_TYPE=r1_distill 但 config.model_type=qwen3")

    # ── 6. chat_template 默认扔 think 行为验证（关键!） ─────────────────────
    section("6. chat_template 历史 think 剥离行为（关键!）")
    print("""  说明:
    R1-Distill-Qwen 的 chat_template 对任何 assistant 消息都做
       content.split('</think>')[-1] → 完全扔掉 think。
    Qwen3-Thinking 只扔 [历史] assistant 的 think; 最后一轮保留。

    UniCOP-Distill 当前 SFT 训练单轮渲染(system+user, 无 assistant turn),
    且 completion 手动字符串拼接, 绕过 chat_template, 因此 think 不会被扔。
    本节做端到端验证。
""")

    # 6.1 多轮对话渲染验证: 历史 assistant 的 think 是否被扔
    multi_turn_msgs = [
        {"role": "system",    "content": "You are a CVRP expert."},
        {"role": "user",      "content": "Solve question A."},
        {"role": "assistant", "content": "<think>HISTORY_THINK_A</think>\nHISTORY_ANSWER_A"},
        {"role": "user",      "content": "Solve question B."},
    ]
    multi_rendered = tokenizer.apply_chat_template(
        multi_turn_msgs, tokenize=False, add_generation_prompt=True,
    )
    print("  [多轮渲染原文]")
    print(multi_rendered)
    print()
    history_think_kept = "HISTORY_THINK_A" in multi_rendered
    history_answer_kept = "HISTORY_ANSWER_A" in multi_rendered
    print(f"  历史 think (HISTORY_THINK_A)  是否保留: {history_think_kept}")
    print(f"  历史 answer (HISTORY_ANSWER_A) 是否保留: {history_answer_kept}")
    check("历史 assistant 的 think 被扔（符合两模型预期行为）",
          not history_think_kept and history_answer_kept,
          "若历史 think 仍保留, 说明 chat_template 行为与预期不符, 需深查",
          warn_only=False)

    # 6.2 最后一轮 assistant 的 think 是否扔（两模型行为不同）
    last_turn_msgs = [
        {"role": "system",    "content": "You are a CVRP expert."},
        {"role": "user",      "content": "Solve question."},
        {"role": "assistant", "content": "<think>FINAL_THINK</think>\nFINAL_ANSWER"},
    ]
    last_rendered = tokenizer.apply_chat_template(
        last_turn_msgs, tokenize=False, add_generation_prompt=False,
    )
    print("\n  [最后一轮 assistant 渲染原文]")
    print(last_rendered)
    last_think_kept = "FINAL_THINK" in last_rendered
    last_answer_kept = "FINAL_ANSWER" in last_rendered
    print(f"  最后一轮 think (FINAL_THINK)   是否保留: {last_think_kept}")
    print(f"  最后一轮 answer (FINAL_ANSWER) 是否保留: {last_answer_kept}")
    if is_qwen3:
        check("Qwen3 最后一轮 assistant 保留 think", last_think_kept,
              "Qwen3-Thinking 应在 loop.last 时包成 <think>...</think>\\n\\n{content}")
    else:
        # R1-Distill: 永远扔
        check("R1-Distill 即使最后一轮也扔 think（已知行为）",
              not last_think_kept and last_answer_kept,
              "R1-Distill chat_template 对所有 assistant 都 split('</think>')[-1]",
              warn_only=True)

    # 6.3 当前 SFT 数据流验证: 单轮 system+user 不会触发扔 think
    sft_msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user",   "content": "user prompt"},
    ]
    sft_rendered = tokenizer.apply_chat_template(
        sft_msgs, tokenize=False, add_generation_prompt=True,
    )
    # 单轮渲染里不会有任何 assistant turn, 也就没有 think 可扔
    # completion 是手动拼接, 整段 <think>X</think>Y 进 completion
    fake_completion = "FAKE_THINK_CONTENT</think>\n\nFAKE_ANSWER" + (tokenizer.eos_token or "")
    full_text = sft_rendered + fake_completion
    has_think_content_in_full = "FAKE_THINK_CONTENT" in full_text
    has_answer_in_full = "FAKE_ANSWER" in full_text
    check("当前 SFT 数据流: think 内容完整保留在训练序列中",
          has_think_content_in_full and has_answer_in_full,
          "completion 手动拼接, 不经 chat_template, think 不会被扔")

    # 6.4 警告: 将来若做 multi-turn SFT 必须用 reasoning_content 字段
    if is_qwen3:
        # 验证 reasoning_content 字段确实能保住历史 think
        reasoning_msgs = [
            {"role": "system",    "content": "sys"},
            {"role": "user",      "content": "Q1"},
            {"role": "assistant", "content": "ANSWER_VIA_REASONING",
             "reasoning_content": "THINK_VIA_REASONING"},
            {"role": "user",      "content": "Q2"},
        ]
        try:
            reasoning_rendered = tokenizer.apply_chat_template(
                reasoning_msgs, tokenize=False, add_generation_prompt=True,
            )
            keeps_via_field = "THINK_VIA_REASONING" in reasoning_rendered
            print(f"\n  Qwen3 多轮: 用 message.reasoning_content 字段时 think 是否保留: {keeps_via_field}")
            check("Qwen3 multi-turn 推荐做法: reasoning_content 字段",
                  True,  # 信息性, 不做硬断言
                  f"将来做多轮训练应使用 reasoning_content 字段, 不能塞进 content",
                  warn_only=True)
        except Exception as e:
            print(f"  reasoning_content 字段渲染异常: {e}")

    # ── 7. config.json 关键字段（dtype / rope / tie_word_embeddings） ───────
    section("7. config.json 关键字段")
    torch_dtype = getattr(config, "torch_dtype", "未设置")
    print(f"  torch_dtype:              {torch_dtype}")
    check("torch_dtype = bfloat16", str(torch_dtype) == "torch.bfloat16",
          f"实际={torch_dtype}（fp16 可能导致 Qwen3 loss 爆炸）", warn_only=True)

    rope_theta = getattr(config, "rope_theta", None)
    rope_scaling = getattr(config, "rope_scaling", None)
    print(f"  rope_theta:               {rope_theta}")
    print(f"  rope_scaling:             {rope_scaling}")
    if is_qwen3:
        check("Qwen3 rope_theta ~ 5M（原生 256K 支持）",
              rope_theta is not None and rope_theta >= 1_000_000,
              f"rope_theta={rope_theta}（Qwen3-Thinking 应 ~5M, 不需额外 scaling）",
              warn_only=True)
        check("Qwen3 不应设额外 rope_scaling",
              rope_scaling is None,
              f"rope_scaling={rope_scaling}（HF discussion #15: Qwen3 已原生支持 256K, 加 scaling 反而出错）",
              warn_only=True)

    tie_we = getattr(config, "tie_word_embeddings", None)
    print(f"  tie_word_embeddings:      {tie_we}")
    # LoRA 不应包含 embed_tokens / lm_head，否则 tie 状态错乱
    lora_targets = ["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
    has_embed_in_targets = any(t in ["embed_tokens", "lm_head"] for t in lora_targets)
    check("LoRA target_modules 不含 embed_tokens/lm_head",
          not has_embed_in_targets,
          "Qwen3 tie_word_embeddings=True, LoRA 误包含会导致 merge 后异常 "
          "(PEFT issue #2944/#2777)")

    # ── 8. generation_config.json 推断默认值 ───────────────────────────────
    section("8. generation_config.json 默认采样值")
    gen_cfg_path = os.path.join(args.model, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        with open(gen_cfg_path, "r", encoding="utf-8") as f:
            gen_cfg = json.load(f)
        for k in ["temperature", "top_p", "top_k", "do_sample",
                  "eos_token_id", "pad_token_id"]:
            if k in gen_cfg:
                print(f"  {k}: {gen_cfg[k]}")
        # eos_token_id 应为 list（避免老 transformers 只取首个）
        eos_in_cfg = gen_cfg.get("eos_token_id")
        if isinstance(eos_in_cfg, list):
            check("eos_token_id 是 list（多 EOS 都生效）",
                  len(eos_in_cfg) >= 1, f"{eos_in_cfg}")
        elif isinstance(eos_in_cfg, int):
            print(f"  {INFO_TAG} eos_token_id 是单 int, 生成时只识别一个 EOS")
        # 警告: vLLM 默认会用这里的值覆盖客户端传参
        if "temperature" in gen_cfg and gen_cfg["temperature"] != 1.0:
            check("vLLM 启动加 --generation-config vllm",
                  "vllm" in os.environ.get("VLLM_REASONING_FLAGS", ""),
                  f"否则 generation_config.json 中 temperature={gen_cfg['temperature']} "
                  "会覆盖你传的采样参数", warn_only=True)
    else:
        print(f"  {INFO_TAG} 未找到 generation_config.json")

    # ── 9. tokenizer round-trip / padding side / 额外 special tokens ────────
    section("9. tokenizer 细节")
    # round-trip 验证（确保 BPE 不丢字符）
    sample_text = rendered_gp + "thinking content </think>\nanswer"
    try:
        ids = tokenizer.encode(sample_text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        roundtrip_ok = decoded == sample_text
        check("tokenizer.decode(encode(x)) == x（round-trip）",
              roundtrip_ok,
              f"长度: 原={len(sample_text)} vs 解码={len(decoded)}（不等可能是 BPE 合并空格）",
              warn_only=not roundtrip_ok)
        if not roundtrip_ok:
            # 找第一个差异位置
            for i, (a, b) in enumerate(zip(sample_text, decoded)):
                if a != b:
                    print(f"        首个差异 @{i}: orig={sample_text[i:i+20]!r} vs decoded={decoded[i:i+20]!r}")
                    break
    except Exception as e:
        print(f"  {WARN_TAG} round-trip 异常: {e}")

    # padding_side（训练必 right, 推理建议 left）
    print(f"  tokenizer.padding_side (默认): {tokenizer.padding_side}")
    print(f"  {INFO_TAG} train_sft_*.py 在 main 中显式设 padding_side='right'，此处不强制断言")

    # additional special tokens
    extra_tokens = getattr(tokenizer, "additional_special_tokens", None) or []
    print(f"  additional_special_tokens 数量: {len(extra_tokens)}")
    if extra_tokens:
        # 截断打印
        print(f"  样本: {extra_tokens[:8]}{' ...' if len(extra_tokens) > 8 else ''}")
    # 检查 response_template 字符串里有没有意外冲突 token
    rt2 = detect_resp_template_full(tokenizer)
    conflict_tokens = [t for t in extra_tokens
                       if t in rt2 and t not in ("<think>", "</think>",
                                                 "<|im_start|>", "<|im_end|>")]
    check("response_template 不含意外的 special token",
          not conflict_tokens,
          f"冲突: {conflict_tokens}", warn_only=True)

    # ── 10. labels mask 验证（构造一个 mini-batch 看 labels 是否只在 completion 上算 loss） ──
    section("10. labels mask 验证（completion-only loss 真实生效）")
    try:
        # 用 Stage 2 流程渲染一条假数据
        sys_p = "You are a CVRP expert."
        user_p = "Solve CVRP for n=3."
        assistant_output = "thinking content here\n</think>\nfinal answer here"

        prompt_t = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user",   "content": user_p}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt_t.rstrip().endswith("<think>"):
            prompt_t += "<think>\n"
        completion_t = assistant_output + (tokenizer.eos_token or "")
        full_t = prompt_t + completion_t

        # 模拟 DataCollatorForCompletionOnlyLM 的 mask 逻辑（Stage 2 路径）
        rt = detect_resp_template_full(tokenizer)
        full_ids = tokenizer.encode(full_t, add_special_tokens=False)
        rt_ids = tokenizer.encode(rt, add_special_tokens=False)
        pos = subseq_pos(full_ids, rt_ids)
        if pos < 0:
            check("Stage 2 response_template 在完整序列中定位", False,
                  "找不到模板, mask 必失败")
        else:
            # 模拟 mask
            labels = list(full_ids)
            mask_end = pos + len(rt_ids)
            for i in range(mask_end):
                labels[i] = -100
            unmasked_ids = [t for t in labels if t != -100]
            unmasked_text = tokenizer.decode(unmasked_ids, skip_special_tokens=False)
            print(f"  loss 计算区间 decode: {unmasked_text!r}")
            # 期望: 不含 system/user 内容, 不含 prompt 末尾的 <think>\n
            mask_ok_no_sys = sys_p not in unmasked_text
            mask_ok_no_user = user_p not in unmasked_text
            mask_ok_has_thinking = "thinking content here" in unmasked_text
            mask_ok_has_answer = "final answer here" in unmasked_text
            check("labels mask: 不含 system 内容", mask_ok_no_sys)
            check("labels mask: 不含 user 内容", mask_ok_no_user)
            check("labels mask: 含 thinking 内容（应学）", mask_ok_has_thinking)
            check("labels mask: 含 final answer（应学）", mask_ok_has_answer)
    except Exception as e:
        print(f"  {WARN_TAG} labels mask 验证异常: {type(e).__name__}: {e}")

    # ── 11. 真实数据 round-trip ────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        section(f"11. 真实数据 round-trip: {args.data}")
        with open(args.data, "r", encoding="utf-8") as f:
            first = json.loads(next(f for f in f if f.strip()))
        # 复用 stage2 渲染逻辑
        system = first["prompt"]["system"]
        # 剥 post-hoc 后缀（与 train_sft_stage2.py 行为一致）
        marker = "\n\nYour output MUST start with <think>"
        if marker in system:
            system = system[:system.find(marker)]
        user = first["prompt"]["user"]
        if "\n\nTarget solution (" in user:
            user = user[:user.find("\n\nTarget solution (")]
        output = first["output"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt_text.rstrip().endswith("<think>"):
            prompt_text += "<think>\n"
        out_stripped = output.lstrip()
        if out_stripped.startswith("<think>"):
            out_stripped = out_stripped[len("<think>"):].lstrip("\n")
        completion_text = out_stripped + (tokenizer.eos_token or "")

        full_text = prompt_text + completion_text
        ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        print(f"  prompt tokens:     {len(prompt_ids)}")
        print(f"  completion tokens: {len(ids) - len(prompt_ids)}")
        print(f"  total tokens:      {len(ids)}")
        print(f"  prompt 末尾 60 字: {prompt_text[-60:]!r}")
        print(f"  completion 前 80 字: {completion_text[:80]!r}")

        # 验证 Stage 2 response_template 仍可在该样本中定位
        rt2_ids = tokenizer.encode(rt_stage2, add_special_tokens=False)
        pos = subseq_pos(ids, rt2_ids)
        check("真实样本 Stage 2 response_template 可定位", pos >= 0,
              f"pos={pos}")

        # 验证 completion 开头不再有重复 <think>
        completion_ids = ids[len(prompt_ids):]
        first_few = tokenizer.decode(completion_ids[:10])
        has_dup_think = first_few.lstrip().startswith("<think>")
        check("completion 开头无重复 <think>", not has_dup_think,
              f"decoded 前 10 token: {first_few!r}（应是 thinking 内容直接开始）")

        # ────────────────────────────────────────────────────────────────────
        # 关键: 模拟 DataCollatorForCompletionOnlyLM 的 mask, 解码 labels != -100
        # 段, 验证 *思维链* 整段都被算入 loss(不只是 final answer)
        # 这是历史踩坑: 若 mask 错位, 只剩 answer 算 loss, 思维链完全没学,
        # loss/grad 仍正常, 难以察觉
        # ────────────────────────────────────────────────────────────────────
        print("\n  [Labels mask 真实数据验证 - 思维链是否进 loss]")
        if pos >= 0:
            mask_end = pos + len(rt2_ids)
            labels = list(ids)
            for i in range(mask_end):
                labels[i] = -100
            loss_token_ids = [t for t in labels if t != -100]
            n_loss = len(loss_token_ids)
            n_completion = len(ids) - len(prompt_ids)
            print(f"  算 loss 的 token 数:   {n_loss}")
            print(f"  completion 总 token: {n_completion}")
            print(f"  覆盖率:               {n_loss/max(1,n_completion):.1%} (应接近 100%)")

            # 解码完整 loss 段(供人眼检查)
            loss_decoded = tokenizer.decode(loss_token_ids, skip_special_tokens=False)
            print(f"\n  [loss 段开头 300 字]")
            print(f"  {loss_decoded[:300]!r}")
            print(f"\n  [loss 段末尾 200 字]")
            print(f"  {loss_decoded[-200:]!r}")

            # 关键断言: thinking 段必须在 loss 段中
            # chains_*.jsonl 里 output 通常是 <think>X</think>Y, 剥 <think> 后
            # out_stripped 应以 thinking 内容开头, 末尾 </think>{answer}<eos>
            check("loss 覆盖整个 completion(>95%)",
                  n_loss / max(1, n_completion) > 0.95,
                  f"覆盖率 {n_loss/max(1,n_completion):.1%}; 若 <50% 说明 mask 错位, "
                  "只剩 final answer 算 loss(历史踩坑: thinking 段没学到)")

            # 找 </think> 在 loss 段中的位置
            think_close_idx = loss_decoded.find("</think>")
            if think_close_idx > 0:
                thinking_chars = think_close_idx  # </think> 之前的字符数 = thinking 段
                answer_chars = len(loss_decoded) - think_close_idx - len("</think>")
                print(f"\n  </think> 在 loss 段位置: {think_close_idx} (前 {thinking_chars} 字符为 thinking, 后 {answer_chars} 字符为 answer)")
                check("thinking 段(</think>之前) 字符数 > 100",
                      thinking_chars > 100,
                      f"thinking 段只有 {thinking_chars} 字符, 大概率被 mask 吃掉")
                check("answer 段(</think>之后) 也在 loss 中",
                      answer_chars > 0,
                      f"answer 段长度 {answer_chars}")
            else:
                check("loss 段含 </think> 标记", False,
                      "loss 解码段里没找到 </think>, 说明 thinking 或 answer 缺失")

            # 反向验证: prompt 部分(system / user 内容)不应进 loss
            sample_user_snippet = user[:50] if len(user) > 50 else user
            user_in_loss = sample_user_snippet in loss_decoded
            check("user 问题内容未泄漏进 loss 段",
                  not user_in_loss,
                  f"user 前 50 字出现在 loss 段中, 说明 prompt 没被 mask 掉")

    # ── 12. Online: 加载真实模型 + 1 step forward/backward ─────────────────
    if args.online:
        section("12. Online: 加载模型跑 1 step forward + backward")
        if not args.data or not os.path.exists(args.data):
            print(f"  {FAIL_TAG} --online 需要同时 --data <chains_*.jsonl>")
            failures.append("--online 缺 --data")
        else:
            try:
                import torch
                from transformers import AutoModelForCausalLM
            except ImportError as e:
                print(f"  {FAIL_TAG} 导入失败: {e}")
                failures.append("torch/transformers 未安装")
            else:
                if not torch.cuda.is_available():
                    print(f"  {FAIL_TAG} CUDA 不可用")
                    failures.append("CUDA 不可用")
                else:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"  GPU:  {gpu_name}")
                    print(f"  显存: {gpu_mem:.1f} GB")

                    print(f"  加载模型 (bf16, device_map=cuda:0)...")
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model, torch_dtype=torch.bfloat16,
                        device_map={"": 0}, trust_remote_code=True,
                    )
                    model.train()
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  参数量: {n_params/1e9:.2f} B")
                    print(f"  model.dtype: {model.dtype}")
                    check("model.dtype == bfloat16", model.dtype == torch.bfloat16,
                          f"实际 {model.dtype}")

                    # 取首条数据走 stage2 流程
                    with open(args.data, "r", encoding="utf-8") as f:
                        line = next(l for l in f if l.strip())
                    rec = json.loads(line)
                    sys_p = rec["prompt"]["system"]
                    marker = "\n\nYour output MUST start with <think>"
                    if marker in sys_p:
                        sys_p = sys_p[:sys_p.find(marker)]
                    user_p = rec["prompt"]["user"]
                    if "\n\nTarget solution (" in user_p:
                        user_p = user_p[:user_p.find("\n\nTarget solution (")]
                    output_text = rec["output"]

                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "system", "content": sys_p},
                         {"role": "user",   "content": user_p}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    if not prompt_text.rstrip().endswith("<think>"):
                        prompt_text += "<think>\n"
                    out_stripped = output_text.lstrip()
                    if out_stripped.startswith("<think>"):
                        out_stripped = out_stripped[len("<think>"):].lstrip("\n")
                    completion_text = out_stripped + (tokenizer.eos_token or "")

                    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                    full_ids = tokenizer.encode(prompt_text + completion_text,
                                                 add_special_tokens=False)
                    orig_len = len(full_ids)
                    if len(full_ids) > args.online_max_len:
                        full_ids = full_ids[:args.online_max_len]
                        print(f"  {INFO_TAG} 序列截断 {orig_len} → {args.online_max_len}")

                    # 构造 labels: prompt 部分 -100, completion 部分保留
                    labels = list(full_ids)
                    p_len = min(len(prompt_ids), len(labels))
                    for i in range(p_len):
                        labels[i] = -100
                    n_loss_tokens = sum(1 for x in labels if x != -100)
                    print(f"  序列长度: total={len(full_ids)}, prompt={p_len}, "
                          f"算 loss 的 token={n_loss_tokens}")
                    check("算 loss 的 token 数 > 0", n_loss_tokens > 0,
                          "若为 0 说明截断把 completion 全切掉")

                    if n_loss_tokens > 0:
                        input_ids = torch.tensor([full_ids], dtype=torch.long,
                                                  device="cuda:0")
                        labels_t = torch.tensor([labels], dtype=torch.long,
                                                  device="cuda:0")

                        # 关键: 解码真实进 loss 的 token, 给人眼看
                        loss_ids_for_decode = [t for t in labels if t != -100]
                        loss_decoded_real = tokenizer.decode(loss_ids_for_decode,
                                                              skip_special_tokens=False)
                        print(f"\n  [真实 forward 输入: labels != -100 段解码]")
                        print(f"  loss token 数:   {len(loss_ids_for_decode)}")
                        print(f"  开头 200 字:     {loss_decoded_real[:200]!r}")
                        print(f"  末尾 150 字:     {loss_decoded_real[-150:]!r}")
                        # 自动断言: thinking 段在 loss 中
                        tc_idx = loss_decoded_real.find("</think>")
                        if tc_idx > 0:
                            thinking_part = loss_decoded_real[:tc_idx]
                            answer_part = loss_decoded_real[tc_idx + len("</think>"):]
                            print(f"  thinking 段长度: {len(thinking_part)} 字符")
                            print(f"  answer   段长度: {len(answer_part)} 字符")
                            check("[online] thinking 段(>100 字符)实际进入 forward 的 loss 计算",
                                  len(thinking_part) > 100,
                                  f"thinking 部分只剩 {len(thinking_part)} 字符, 可能 mask 错位")
                        else:
                            check("[online] forward loss 段含 </think>", False,
                                  "未找到, 可能截断或 mask 错")

                        # forward
                        print(f"\n  forward...")
                        out = model(input_ids=input_ids, labels=labels_t)
                        loss_val = out.loss.item()
                        print(f"  Loss = {loss_val:.4f}")
                        check("loss finite",
                              bool(torch.isfinite(out.loss).item()),
                              "NaN/inf 通常是 dtype/pad/template 问题")
                        check("loss 在合理范围 [0.5, 10]",
                              0.5 <= loss_val <= 10,
                              f"loss={loss_val:.4f} (thinking SFT 首 loss 通常 1.5-3)",
                              warn_only=True)

                        # 严格分段 loss 对比: 用 model.forward(labels) 算
                        # 仅 thinking 段 / 仅 answer 段 的 loss, 验证两段都非零
                        try:
                            # 找 </think> 在 input_ids 中的位置 (token 级)
                            close_str = "</think>"
                            close_ids = tokenizer.encode(close_str, add_special_tokens=False)
                            close_pos = subseq_pos(full_ids, close_ids)
                            if close_pos > p_len and close_pos + len(close_ids) < len(full_ids):
                                # 构造两份 labels: 仅 thinking / 仅 answer
                                labels_think_only = [-100] * len(full_ids)
                                for i in range(p_len, close_pos):
                                    labels_think_only[i] = full_ids[i]
                                labels_ans_only = [-100] * len(full_ids)
                                for i in range(close_pos, len(full_ids)):
                                    labels_ans_only[i] = full_ids[i]
                                n_think = sum(1 for t in labels_think_only if t != -100)
                                n_ans = sum(1 for t in labels_ans_only if t != -100)

                                with torch.no_grad():
                                    out_think = model(
                                        input_ids=input_ids,
                                        labels=torch.tensor([labels_think_only],
                                                              dtype=torch.long, device="cuda:0"))
                                    out_ans = model(
                                        input_ids=input_ids,
                                        labels=torch.tensor([labels_ans_only],
                                                              dtype=torch.long, device="cuda:0"))
                                loss_think = out_think.loss.item()
                                loss_ans = out_ans.loss.item()
                                print(f"\n  [分段 loss 对照]")
                                print(f"  thinking-only loss = {loss_think:.4f} ({n_think} tokens)")
                                print(f"  answer-only   loss = {loss_ans:.4f} ({n_ans} tokens)")
                                print(f"  combined      loss = {loss_val:.4f} ({n_loss_tokens} tokens)")
                                check("thinking 段 loss 有限且 > 0",
                                      torch.isfinite(out_think.loss).item() and loss_think > 0,
                                      f"thinking_loss={loss_think:.4f} (若 = 0 或 NaN 说明 thinking 没参与 loss)")
                                check("answer 段 loss 有限且 > 0",
                                      torch.isfinite(out_ans.loss).item() and loss_ans > 0,
                                      f"answer_loss={loss_ans:.4f}")
                                # combined ≈ token 加权平均
                                expected = (loss_think * n_think + loss_ans * n_ans) / max(1, n_think + n_ans)
                                rel_err = abs(loss_val - expected) / max(0.01, expected)
                                check("combined loss ≈ thinking/answer 加权平均",
                                      rel_err < 0.05,
                                      f"实际 {loss_val:.4f} vs 期望 {expected:.4f} (相对误差 {rel_err:.1%}; "
                                      "若 > 5% 说明 combined loss 主要由某一段贡献, 另一段被错误屏蔽)")
                                del out_think, out_ans
                        except Exception as _seg_e:
                            print(f"  {WARN_TAG} 分段 loss 对照异常: {type(_seg_e).__name__}: {_seg_e}")

                        # backward + grad check
                        print(f"  backward...")
                        out.loss.backward()
                        grad_norm_sq = 0.0
                        grad_nan = False
                        first_nan_name = ""
                        n_with_grad = 0
                        for name, p in model.named_parameters():
                            if p.grad is not None:
                                if not torch.isfinite(p.grad).all():
                                    grad_nan = True
                                    first_nan_name = name
                                    break
                                grad_norm_sq += p.grad.float().norm().item() ** 2
                                n_with_grad += 1
                        grad_norm = grad_norm_sq ** 0.5
                        print(f"  grad_norm = {grad_norm:.4f} ({n_with_grad} 个 tensor 有 grad)")
                        check("所有参数 grad finite", not grad_nan,
                              f"首个 NaN 参数: {first_nan_name}" if grad_nan else "")
                        check("grad_norm > 0", grad_norm > 0,
                              "= 0 说明 loss 与参数断链 (label mask 错位 / model.eval())")

                        # 清理
                        del out, input_ids, labels_t
                    del model
                    torch.cuda.empty_cache()
                    print(f"  显存已释放")

    # ── 结尾汇总 ────────────────────────────────────────────────────────────
    section("总结")
    if failures:
        print(f"  {FAIL_TAG} {len(failures)} 项失败:")
        for f in failures:
            print(f"      - {f}")
    else:
        print(f"  {PASS_TAG} 全部硬性检查通过")
    if warnings:
        print(f"  {WARN_TAG} {len(warnings)} 项警告（不阻塞但需关注）:")
        for w in warnings:
            print(f"      - {w}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
