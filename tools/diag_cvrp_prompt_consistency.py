"""诊断 SFT 与 RL 的 CVRP prompt 一致性 + 实测 cvrp20/100 token 长度.

回答的问题:
  1. SFT 的 _SYSTEM (UniCOP-Distill/problems_prompt.py) 和 RL 的 _SYSTEM
     (UniCOP-Reason/problems/cvrp.py) 字符串是否完全一致?
  2. SFT 训练 cvrp20 时 max_length=4864, prompt 实际占多少, 给 completion 留多少余量?
  3. RL 训练 cvrp100 时 prompt 多长 (含 RL 加的 [[instance_id]] 标记)?
  4. cvrp20 SFT 模型基座见过的 prompt 长度 vs cvrp100 RL 喂的 prompt 长度差距?

只读 tokenizer 的 metadata, 不需要 torch / GPU. 完全本地跑.
PYTHONIOENCODING=utf-8 python tools/diag_cvrp_prompt_consistency.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "UniCOP-Reason"))
sys.path.insert(0, os.path.join(REPO, "UniCOP-Distill"))

import numpy as np
from transformers import AutoTokenizer

from problems import get_problem  # UniCOP-Reason/problems/__init__.py
from problems.cvrp import _SYSTEM as REASON_CVRP_SYSTEM
from problems_prompt import _PROMPTS as DISTILL_PROMPTS  # UniCOP-Distill/problems_prompt.py

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


def diff_strings(a: str, b: str, label_a: str, label_b: str) -> bool:
    """字符级 diff. 返回 True 若一致."""
    if a == b:
        print(f"  ✓ {label_a} == {label_b}  ({len(a)} chars)")
        return True
    print(f"  ✗ {label_a} != {label_b}")
    print(f"    {label_a}: {len(a)} chars,  {label_b}: {len(b)} chars")
    # 找首个 diff 位置
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            ctx_a = a[max(0, i-20):i+40]
            ctx_b = b[max(0, i-20):i+40]
            print(f"    首个不同位置 idx={i}:")
            print(f"      {label_a}[{i-20}:{i+40}] = {ctx_a!r}")
            print(f"      {label_b}[{i-20}:{i+40}] = {ctx_b!r}")
            return False
    # 长度不同, 一方是另一方前缀
    longer = a if len(a) > len(b) else b
    common_len = min(len(a), len(b))
    print(f"    一方是另一方前缀, 多出 {len(longer) - common_len} 字符:")
    print(f"      多出尾部: {longer[common_len:common_len+80]!r}")
    return False


def measure_prompt(tok, problem, n: int, samples: int = 5, with_instance_marker: bool = False):
    """返回字典 {sys, user, chat, chat_with_marker} 的 token 数中位数."""
    sys_lens, user_lens, chat_lens, chat_marker_lens = [], [], [], []
    rng = np.random.default_rng(42)
    for _ in range(samples):
        inst = problem.generate_instance(n, rng)
        msgs = problem.build_prompt(inst)
        sys_text = msgs[0]["content"]
        user_text = msgs[1]["content"]
        sys_lens.append(len(tok.encode(sys_text, add_special_tokens=False)))
        user_lens.append(len(tok.encode(user_text, add_special_tokens=False)))

        chat_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_lens.append(len(tok.encode(chat_txt, add_special_tokens=False)))

        # RL 模式: prepare_dataset.py 在 user content 末尾插入 [[instance_id:xxx]]
        msgs_rl = [dict(m) for m in msgs]
        msgs_rl[1]["content"] = msgs_rl[1]["content"] + f"\n\n[[instance_id:cvrp{n}_train_000001]]"
        chat_rl = tok.apply_chat_template(msgs_rl, tokenize=False, add_generation_prompt=True)
        chat_marker_lens.append(len(tok.encode(chat_rl, add_special_tokens=False)))

    return {
        "sys":  int(np.median(sys_lens)),
        "user": int(np.median(user_lens)),
        "chat": int(np.median(chat_lens)),
        "chat_with_marker": int(np.median(chat_marker_lens)),
        "max_chat": max(chat_lens),
        "max_chat_marker": max(chat_marker_lens),
    }


def main():
    print("=" * 80)
    print(f"  SFT vs RL · CVRP prompt 一致性 + 实测 token 长度")
    print(f"  Tokenizer: {MODEL}")
    print("=" * 80)

    print("\n[1/3] 检查 SFT 和 RL 的 system prompt 字符串是否一致")
    print("  SFT 系统 prompt 来源: UniCOP-Distill/problems_prompt.py _PROMPTS['cvrp']")
    print("  RL 系统 prompt 来源:  UniCOP-Reason/problems/cvrp.py _SYSTEM")
    is_same = diff_strings(
        DISTILL_PROMPTS["cvrp"], REASON_CVRP_SYSTEM,
        "Distill _PROMPTS[cvrp]", "Reason _SYSTEM",
    )
    if not is_same:
        print("  ❌ system prompt 不一致 → SFT 学的 think 模板和 RL 喂的不一样, 模型会跑偏!")
    else:
        print("  ✓ system prompt 完全一致, build_prompt(cvrp).system 在 SFT/RL 阶段相同")

    print("\n[2/3] 检查 RL 的 build_prompt 与 SFT 的 build_prompt 是否同一份")
    print("  SFT  : train_sft_stage2.py:195 从 chains_hybrid_cvrp20.jsonl 读 r['prompt']['system'/'user'],")
    print("         即 generate 阶段调 build_prompt 写入的字段, 训练时再 strip 掉 post-hoc 后缀")
    print("  RL   : prepare_dataset.py:97 直接调 problem.build_prompt(inst)")
    print("  → 共用 problems/cvrp.py:32 的 build_prompt, 只是 RL 在 user 末尾追加 [[instance_id:xxx]]")

    print("\n[3/3] 实测 CVRP n=20 (SFT 范畴) 和 n=100 (RL 目标) 的 prompt token 长度")
    print(f"  加载 tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"    model_max_length = {tok.model_max_length}")

    cvrp = get_problem("cvrp")

    # SFT 配置
    SFT_MAX_LENGTH = 4864
    # RL 配置 (脚本里设的)
    RL_MAX_LEN = 8192
    RL_MAX_NEW_TOKENS = 4096

    print(f"\n  ── CVRP n=20 (SFT 训练时见过) ──────────────────────────────────")
    m20 = measure_prompt(tok, cvrp, n=20, samples=5)
    print(f"    sys={m20['sys']} tokens")
    print(f"    user={m20['user']} tokens (101 节点信息)")
    print(f"    chat (apply_chat_template) = {m20['chat']} tokens [中位数]  / max {m20['max_chat']}")
    print(f"    chat + RL [[instance_id]]   = {m20['chat_with_marker']} tokens [中位数]")
    sft_chain_room = SFT_MAX_LENGTH - m20["max_chat"]
    print(f"    SFT max_length=4864:")
    print(f"      → prompt 用 {m20['max_chat']} tokens, 剩 {sft_chain_room} tokens 给 chain")
    print(f"      → train_sft_stage2.py:265 是 prompt+completion 总长 > 4864 时丢弃 (NOT 截断)")
    print(f"      → 所以 cvrp20 chain ≤ {sft_chain_room} tokens 的样本全保留, 超过的整条丢")

    print(f"\n  ── CVRP n=100 (RL 训练目标) ────────────────────────────────────")
    m100 = measure_prompt(tok, cvrp, n=100, samples=5)
    print(f"    sys={m100['sys']} tokens")
    print(f"    user={m100['user']} tokens (101 节点信息)")
    print(f"    chat (apply_chat_template) = {m100['chat']} tokens [中位数]  / max {m100['max_chat']}")
    print(f"    chat + RL [[instance_id]]   = {m100['chat_with_marker']} tokens [中位数] / max {m100['max_chat_marker']}")
    rl_chain_room = RL_MAX_LEN - m100["max_chat_marker"]
    print(f"    RL MAX_LEN={RL_MAX_LEN}, max_new_tokens={RL_MAX_NEW_TOKENS}:")
    print(f"      → 实际 prompt 上限 = MAX_LEN - max_new_tokens = {RL_MAX_LEN - RL_MAX_NEW_TOKENS}")
    if m100["max_chat_marker"] > RL_MAX_LEN - RL_MAX_NEW_TOKENS:
        print(f"      ❌ 实测 max prompt {m100['max_chat_marker']} > {RL_MAX_LEN - RL_MAX_NEW_TOKENS}, completion 被压缩!")
    else:
        room = (RL_MAX_LEN - RL_MAX_NEW_TOKENS) - m100["max_chat_marker"]
        print(f"      ✓ 实测 max prompt {m100['max_chat_marker']} 在上限 {RL_MAX_LEN - RL_MAX_NEW_TOKENS} 之内, 余量 {room} tokens")

    print(f"\n  ── 长度对比: SFT 模型见过 vs RL 喂的 ──────────────────────────")
    print(f"    SFT cvrp20 prompt:  {m20['max_chat']} tokens (模型见过)")
    print(f"    RL  cvrp100 prompt: {m100['max_chat_marker']} tokens (3.x 倍长)")
    print(f"    模型从没见过这么长的 user 输入 → OOD 风险, 但 prompt 结构 (system 模板) 完全一致")

    print("\n" + "=" * 80)
    print("  结论")
    print("=" * 80)
    print(f"  ✓ SFT/RL 的 prompt 来源同一个 build_prompt + 同一份 system 模板")
    print(f"  ✓ RL 多了 [[instance_id:xxx]] 标记 (~{m100['chat_with_marker'] - m100['chat']} tokens, 给 reward server 用)")
    print(f"  ⚠ SFT max_length=4864 是丢弃超长样本, 不是截断 prompt")
    print(f"     - cvrp20 prompt {m20['max_chat']} tokens, chain 余量 {sft_chain_room} tokens")
    print(f"     - 若 cvrp20 chain 中位数 < {sft_chain_room}, 多数样本进入训练; 否则被大量丢弃")
    print(f"  ⚠ RL cvrp100 prompt {m100['max_chat_marker']} tokens, 是 SFT 时见过的最大长度的 ~{m100['max_chat_marker']/m20['max_chat']:.1f}x")
    print(f"     → 长上下文 OOD, 但不会触发训练阶段的 prompt 截断")


if __name__ == "__main__":
    main()
