"""chat_template 补 <think> 补丁 — 对齐 SFT stage2，修 instruct 基座 GRPO/eval 分布断裂。

背景 (2026-06-23 定位):
  Qwen3-4B-Instruct-2507 的 chat_template 在 add_generation_prompt 时末尾**不带** <think>，
  而 SFT stage2 (train_sft_stage2.py:199/243-244) 探针检测到这点后**手动补了** <think>\n
  教模型续写 think。但 GRPO 训练 (data/generate.py → TRL apply_chat_template) 和
  eval (evaluate.py) 都没补 → GRPO rollout / eval 的 prompt 分布与 SFT 训练断裂。
  thinking 基座 (Qwen3-4B-Thinking-2507) 的 chat_template 自动加 <think>，三处天然一致，
  不受影响 (这解释了为何 thinking 基座 V6 不塌缩)。

本补丁: 探针检测，若 add_generation_prompt 末尾无 <think>，patch chat_template 使其
  自动追加 <think>\n，让 GRPO rollout / eval 与 SFT 对齐。一处 patch，GRPO (trainer
  主进程用 processing_class 文本化 prompt 再发 vLLM server) 与 eval (apply_chat_template)
  全部生效。thinking 基座探针为 True，自动跳过、返回 False。
"""

# 追加到 chat_template 末尾的 jinja 块: add_generation_prompt 时补 <think>\n。
# {%- 去 if 块前空白; <think>\n 输出含真实换行; {% endif %} 不带 - 以保留该换行。
_THINK_INJECT = "{%- if add_generation_prompt %}<think>\n{% endif %}"


def patch_chat_template_for_think(tokenizer, *, verbose: bool = True) -> bool:
    """探针 + patch。返回 True=已 patch(instruct 基座); False=无需(thinking 基座/无 template)。"""
    if not getattr(tokenizer, "chat_template", None):
        if verbose:
            print("[think-patch] tokenizer 无 chat_template, 跳过")
        return False

    def _probe() -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": "probe"}],
            tokenize=False, add_generation_prompt=True,
        )

    if _probe().rstrip().endswith("<think>"):
        if verbose:
            print("[think-patch] chat_template 已自动加 <think> (thinking 基座口径), 无需 patch")
        return False

    tokenizer.chat_template = tokenizer.chat_template + _THINK_INJECT

    # 验证 patch 生效 (jinja whitespace 行为各版本可能不同 → 不静默, patch 失败直接报错)
    if not _probe().rstrip().endswith("<think>"):
        raise RuntimeError(
            "[think-patch] patch 后 chat_template 末尾仍无 <think>, patch 未生效。"
            "请人工检查该 tokenizer 的 chat_template jinja 结构。"
        )
    if verbose:
        print("[think-patch] instruct 基座: chat_template 末尾无 <think>, 已 patch 自动补 "
              "<think>\\n (对齐 SFT stage2, 修 GRPO/eval 分布断裂)")
    return True
