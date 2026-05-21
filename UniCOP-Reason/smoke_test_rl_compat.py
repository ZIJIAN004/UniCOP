"""Smoke test for RL/GRPO migration (R1-Distill ↔ Qwen3-4B-Thinking).

跑前必通过, 验证 GRPO 训练栈在 Qwen3 上的兼容性。
出现 [FAIL] 必须修复, [WARN] 关注但可继续, [PASS] 通过。

覆盖 10 个 section:
  0. 环境 + tokenizer / config 加载
  1. Special tokens (pad ≠ eos / `<think>` `</think>` token id)
  2. tokenizer.decode 行为: skip_special_tokens=False 是否保住 </think>
  3. chat_template 自动 prepend `<think>` 验证 (R1 + Qwen3 都这样)
  4. config.py: gen_temperature/top_p/top_k 是否正确从 env 读
  5. GRPOConfig 是否真的接受 temperature/top_p/top_k (trl 0.16 验证)
  6. _strip_chat_specials 单元测试 (保 think / 抹 chat 控制 token)
  7. reward_fn.py 的 _extract_completion 在 Qwen3 marker 上的行为
  8. completion `</think>` 边界检测 (parse 逻辑依赖)
  9. End-to-end mini round-trip (build_prompt → chat_template → encode/decode)

用法:
    # 默认从 BASE_MODEL_TYPE 环境变量推导模型路径 (建议先 source paths.sh)
    source ../paths.sh && python smoke_test_rl_compat.py

    # 显式指定模型路径
    python smoke_test_rl_compat.py --model /path/to/Qwen3-4B-Thinking-2507
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoConfig
except ImportError:
    print("ERROR: pip install transformers")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

PASS_TAG = "[PASS]"
FAIL_TAG = "[FAIL]"
WARN_TAG = "[WARN]"
INFO_TAG = "[INFO]"

failures = []
warnings = []


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = PASS_TAG if ok else (WARN_TAG if warn_only else FAIL_TAG)
    print(f"  {tag} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        (warnings if warn_only else failures).append(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.environ.get("BASE_MODEL", ""))
    parser.add_argument("--model_type", type=str,
                        default=os.environ.get("BASE_MODEL_TYPE", ""))
    args = parser.parse_args()

    if not args.model:
        print("ERROR: 需要 --model 或先 source paths.sh export BASE_MODEL")
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
    is_qwen3 = "qwen3" in config.model_type.lower() or "Qwen3" in str(config.architectures)
    is_r1 = "qwen2" in config.model_type.lower() and not is_qwen3
    print(f"  model_type:       {config.model_type}")
    print(f"  architectures:    {config.architectures}")
    print(f"  is_qwen3 = {is_qwen3}  is_r1_distill = {is_r1}")

    # ── 1. Special tokens ──────────────────────────────────────────────────
    section("1. Special tokens")
    print(f"  bos_token:      {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    print(f"  eos_token:      {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token:      {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # GRPO 用 attention_mask 算 completion_mask, pad == eos 不会触发 SFT 那种
    # "EOS 被 mask 不学"的 bug, 但仍然警告 (训练 log 更难看)
    check("pad ≠ eos (GRPO 不致命但推荐)",
          tokenizer.pad_token_id != tokenizer.eos_token_id,
          f"pad_id={tokenizer.pad_token_id} eos_id={tokenizer.eos_token_id}",
          warn_only=True)

    # <think> / </think> token id 探测
    think_open_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    think_close_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    print(f"  <think>:        ids={think_open_ids}  len={len(think_open_ids)}")
    print(f"  </think>:       ids={think_close_ids}  len={len(think_close_ids)}")

    check("</think> 单 token (mask processor R2 think 闭合规则需要)",
          len(think_close_ids) == 1,
          f"ids={think_close_ids}",
          warn_only=True)  # R1-Distill 上多 token 是已知 fallback, 不阻塞

    # ── 2. skip_special_tokens 行为 ─────────────────────────────────────────
    section("2. tokenizer.decode 行为: skip_special_tokens 是否抹掉 </think>")
    sample_text = "Reasoning here.</think>\n\nAnswer: 1 2 3"
    sample_ids = tokenizer(sample_text, add_special_tokens=False).input_ids
    dec_true = tokenizer.decode(sample_ids, skip_special_tokens=True)
    dec_false = tokenizer.decode(sample_ids, skip_special_tokens=False)
    print(f"  原文:                       {sample_text!r}")
    print(f"  decode(skip=True):  {dec_true!r}")
    print(f"  decode(skip=False): {dec_false!r}")
    check("decode(skip=False) 保住 </think>",
          "</think>" in dec_false,
          "训练 + eval 后续 parse 依赖此边界")
    # decode(skip=True) 在 Qwen3 上是否抹 </think>? 实测 Qwen3 tokenizer_config.json
    # 里 </think> "special": false, 所以 skip=True 也保留. 这里只是 sanity print, 不强 check.
    print(f"  {INFO_TAG} decode(skip=True) 保 </think>? "
          f"{'是' if '</think>' in dec_true else '否'}  "
          f"(Qwen3 应为 是 / R1 也是; 我们仍坚持用 skip=False 更明确)")

    # ── 3. chat_template 自动 prepend <think> 验证 ─────────────────────────
    section("3. chat_template 自动注入 `<think>` 验证")
    msgs = [
        {"role": "system", "content": "You are a planner."},
        {"role": "user", "content": "Solve this CVRP."},
    ]
    chat_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"  apply_chat_template(add_generation_prompt=True) 末尾 80 char:")
    print(f"    {chat_text[-80:]!r}")
    check("chat_template 在 assistant header 后 prepend <think>",
          chat_text.rstrip().endswith("<think>") or "<think>\n" in chat_text[-30:],
          "两个模型都这样, generation 实际从 <think> 后开始")

    # ── 4. config.py: gen_* 是否从 env 正确读 ──────────────────────────────
    section("4. config.py 字段验证 (从 env 正确读取)")
    try:
        from config import config as _config_module
        # 重新 import 模块强制读最新 env (Config 是 dataclass, 字段在 import 时求值)
        import importlib
        import config as _cm
        importlib.reload(_cm)
        c = _cm.config

        env_t = float(os.environ.get("GEN_TEMPERATURE", "1.0"))
        env_tp = float(os.environ.get("GEN_TOP_P", "1.0"))
        env_tk = int(os.environ.get("GEN_TOP_K", "-1"))

        check("config.gen_temperature == GEN_TEMPERATURE",
              abs(c.gen_temperature - env_t) < 1e-6,
              f"config={c.gen_temperature} env={env_t}")
        check("config.gen_top_p == GEN_TOP_P",
              abs(c.gen_top_p - env_tp) < 1e-6,
              f"config={c.gen_top_p} env={env_tp}")
        check("config.gen_top_k == GEN_TOP_K",
              c.gen_top_k == env_tk,
              f"config={c.gen_top_k} env={env_tk}")
        print(f"  {INFO_TAG} BASE_MODEL_TYPE={args.model_type} → 期望: "
              f"r1_distill T=1.0/p=1.0/k=-1, qwen3_thinking T=0.6/p=0.95/k=20")
    except Exception as e:
        check("config.py import / 字段读取", False, f"异常: {type(e).__name__}: {e}")

    # ── 5. GRPOConfig 接受 temperature/top_p/top_k ─────────────────────────
    section("5. GRPOConfig 字段验证 (trl 0.16 接受这三个参数)")
    try:
        from trl import GRPOConfig
        import trl
        print(f"  trl version: {trl.__version__}")
        # 用最小参数实例化
        gc = GRPOConfig(
            output_dir="/tmp/smoke_grpo",
            num_generations=2,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
        check("GRPOConfig 接受 temperature 参数", hasattr(gc, "temperature"),
              f"实际值={getattr(gc, 'temperature', 'N/A')}")
        check("GRPOConfig 接受 top_p 参数", hasattr(gc, "top_p"),
              f"实际值={getattr(gc, 'top_p', 'N/A')}")
        check("GRPOConfig 接受 top_k 参数", hasattr(gc, "top_k"),
              f"实际值={getattr(gc, 'top_k', 'N/A')}")
    except Exception as e:
        check("GRPOConfig 实例化", False,
              f"异常: {type(e).__name__}: {e} → 若 'unexpected keyword', "
              f"trl 版本太老需升级, train.py 改动失效")

    # ── 6. _strip_chat_specials 单元测试 ───────────────────────────────────
    section("6. _strip_chat_specials helper (保 think / 抹 chat 控制 token)")
    try:
        from grpo_prm_trainer import _strip_chat_specials, _CHAT_SPECIAL_TOKENS
        # case A: 含 chat 控制 token 应被抹
        text_a = "Hi.<|im_end|>\n<|im_start|>assistant\n<think>Reason</think>\nAnswer"
        out_a = _strip_chat_specials(text_a)
        check("_strip_chat_specials 抹掉 <|im_end|>",
              "<|im_end|>" not in out_a, f"输出: {out_a!r}")
        check("_strip_chat_specials 抹掉 <|im_start|>",
              "<|im_start|>" not in out_a, f"输出: {out_a!r}")
        check("_strip_chat_specials 保留 <think>",
              "<think>" in out_a, f"输出: {out_a!r}")
        check("_strip_chat_specials 保留 </think>",
              "</think>" in out_a, f"输出: {out_a!r}")
        print(f"  {INFO_TAG} _CHAT_SPECIAL_TOKENS 共 {len(_CHAT_SPECIAL_TOKENS)} 项")
    except Exception as e:
        check("_strip_chat_specials import / 调用", False,
              f"异常: {type(e).__name__}: {e}")

    # ── 7. reward_fn.py: _extract_completion + assistant marker ────────────
    section("7. reward_fn._extract_completion 在两种模型 marker 上的行为")
    try:
        import importlib.util
        rwf_path = SCRIPT_DIR / "openrlhf" / "reward" / "reward_fn.py"
        if rwf_path.exists():
            spec = importlib.util.spec_from_file_location("reward_fn", rwf_path)
            mod = importlib.util.module_from_spec(spec)
            # _ensure_loaded 会去找 instances 文件, 用 mock 避开
            sys.modules["reward_fn"] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass  # 可能 import 失败但 _extract_completion 仍可用
            extract = getattr(mod, "_extract_completion", None)
            markers = getattr(mod, "_ASSISTANT_MARKERS", [])

            check("_ASSISTANT_MARKERS 包含 Qwen3 marker `<|im_start|>assistant`",
                  "<|im_start|>assistant" in markers,
                  f"实际: {markers}")
            check("_ASSISTANT_MARKERS 包含 R1 marker `<｜Assistant｜>`",
                  "<｜Assistant｜>" in markers,
                  f"实际: {markers}")

            if extract is not None:
                # Qwen3 marker 路径
                prompt_q = "<|im_start|>user\nHi<|im_end|>\n"
                query_q = prompt_q + "<|im_start|>assistant\n<think>\nthink content\n</think>\nfinal"
                out_q = extract(query_q, prompt="THIS-WILL-NOT-MATCH")  # 强迫走 marker 路径
                check("Qwen3 marker 路径: extract 后含 think 内容",
                      "think content" in out_q,
                      f"输出: {out_q!r}")
                check("Qwen3 marker 路径: extract 后已去掉 `<think>` 起始",
                      not out_q.lstrip().startswith("<think>"),
                      f"输出: {out_q!r}")
        else:
            check("reward_fn.py 存在", False, f"未找到 {rwf_path}")
    except Exception as e:
        check("reward_fn.py 加载", False,
              f"异常: {type(e).__name__}: {e}", warn_only=True)

    # ── 8. completion `</think>` 边界检测 ──────────────────────────────────
    section("8. completion `</think>` 边界检测 (parse 逻辑)")
    fake_completion = "Reasoning...</think>\n\nRoute 1: [R1,0] 1 2 3 [R1,4] 0"
    te = fake_completion.find("</think>")
    check("fake completion 能用 find('</think>') 定位答案区",
          te > 0, f"位置: {te}")
    if te > 0:
        answer_part = fake_completion[te + len("</think>"):]
        check("答案区非空", len(answer_part.strip()) > 0,
              f"答案区前 50 char: {answer_part[:50]!r}")

    # ── 9. End-to-end mini round-trip ──────────────────────────────────────
    section("9. End-to-end: build_prompt → chat_template → encode/decode")
    try:
        from problems.cvrp import CVRP
        import numpy as np
        prob = CVRP()
        rng = np.random.default_rng(seed=42)
        inst = prob.generate_instance(n=10, rng=rng)
        prompt_msgs = prob.build_prompt(inst)
        chat_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids
        n_prompt_tokens = prompt_ids.shape[1]
        # 解码回去
        decoded = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
        # 控制 token 残留情况
        ctrl_tokens_present = [t for t in
                                ("<|im_start|>", "<|im_end|>", "<｜begin▁of▁sentence｜>")
                                if t in decoded]
        print(f"  CVRP-10 prompt: {n_prompt_tokens} tokens")
        print(f"  含 chat 控制 token: {ctrl_tokens_present}")
        check("prompt token 数 < max_prompt_length=768",
              n_prompt_tokens < 768,
              f"实际: {n_prompt_tokens}")
        check("encode → decode round-trip 含 user message",
              "CVRP" in decoded or "Customer" in decoded or "depot" in decoded.lower(),
              "decoded 前 200 char: " + decoded[:200].replace('\n', ' '))
    except Exception as e:
        check("End-to-end CVRP-10 prompt build", False,
              f"异常: {type(e).__name__}: {e}")

    # ── 总结 ───────────────────────────────────────────────────────────────
    section("总结")
    if failures:
        print(f"  ❌ {len(failures)} 项 FAIL:")
        for f in failures:
            print(f"     - {f}")
    if warnings:
        print(f"  ⚠️  {len(warnings)} 项 WARN:")
        for w in warnings:
            print(f"     - {w}")
    if not failures and not warnings:
        print(f"  ✅ 所有 check 通过, RL 训练栈在 {args.model_type} 下兼容")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
