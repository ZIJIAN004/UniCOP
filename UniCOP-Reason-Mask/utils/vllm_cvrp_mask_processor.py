"""
vLLM logits_processor: CVRP constrained-decoding mask.

桥接层: 把 cvrp_mask_state.py 的语义级 MaskDecision 转成 vLLM token-level
logits 操作 (logits[token_id] = -inf).

设计要点 (基于调研结论):
  1. Stateless: 每次 __call__ 从 prompt_ids + output_ids 全量重建 state
     (vLLM 0.7.3 worker process + output_ids 是新 tuple, 不能 cache state)
  2. Per-call 局部缓存优化: 用 (id(output_ids), len(output_ids)) 作 key
     避免完全冗余重建. tuple 每次新建虽然 id 不稳, 但单次 __call__ 内 id 稳定.
  3. tokenizer 单 token 假设 + multi-token 检测告警
     (Qwen2.5 一般 0..99 数字单 token, 但用 sanity_check 严格验证)
  4. enabled 总开关: 默认 False, 跟现有 hardgate run 共存不冲突

集成方式 (Phase 3 做):
  vllm_serve_logprobs.py 内, /generate/ handler 构造 SamplingParams 后:
    sampling_params.logits_processors = [CVRPMaskProcessor(n=20, tokenizer=...)]
  (实例字段赋值, 不是 monkey-patch __init__, 因为 SamplingParams 是 msgspec.Struct)

参考:
  - utils/cvrp_mask_state.py — state machine 实现
  - CVRP-LLM-Mask-完整规则.md — 完整设计文档
"""
from __future__ import annotations

import sys
from typing import Optional

import torch

from utils.cvrp_mask_state import (
    MaskConfig,
    MaskDecision,
    build_state,
    compute_mask,
)


class CVRPMaskProcessor:
    """vLLM V0 3-arg logits_processor: (prompt_ids, output_ids, logits) -> logits.

    Stateless 每次重建. 多 sequence 共享同一实例无副作用 (无 self.* 累加 state).
    """

    def __init__(
        self,
        n: int,
        tokenizer,
        cfg: Optional[MaskConfig] = None,
        debug_log: bool = False,
    ):
        self.n = n
        self.tokenizer = tokenizer
        self.cfg = cfg or MaskConfig(n=n)
        self.debug_log = debug_log or self.cfg.debug_log

        # 预计算关键 token id (启动时一次完成, 不在 hot path 重算)
        self._build_token_maps()

        if self.debug_log:
            self._log_token_maps()

    # ── 初始化: token 映射 ──────────────────────────────────────────

    def _encode_first(self, text: str) -> Optional[int]:
        """编码 text, 返回第一个 token id. 多 token 时返回首 token + 告警."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) == 0:
            return None
        if len(ids) > 1:
            print(
                f"⚠️ [CVRPMask] '{text}' is multi-token ({len(ids)} tokens, ids={ids}), "
                f"using first token only ({ids[0]}). Mask may be incomplete.",
                file=sys.stderr, flush=True,
            )
        return ids[0]

    def _build_token_maps(self):
        """预计算 5 条规则需要的 token id."""
        # Customer 1..n 的 first token (带空格前缀, 跟 CVRP chain 里的 "→ select N" 一致)
        # 如果 N 是 multi-token (如 " 13" = " 1" + "3"), 只 mask " 1"
        # 这会有 prefix tree 共享问题 (mask 1 也 ban 10-19), 但只在 multi-token 时
        self.cust_first_token: dict[int, Optional[int]] = {}
        self.multi_token_customers: list[int] = []
        for c in range(1, self.n + 1):
            ids = self.tokenizer(f" {c}", add_special_tokens=False).input_ids
            if len(ids) == 1:
                self.cust_first_token[c] = ids[0]
            else:
                # multi-token: first token 跟其他客户可能共享 (如 " 13" 和 " 1" 都以 " 1" 开头)
                self.cust_first_token[c] = ids[0]
                self.multi_token_customers.append(c)
        if self.multi_token_customers:
            print(
                f"⚠️ [CVRPMask] customers {self.multi_token_customers} are multi-token; "
                f"mask 可能在这些 customer 上不精确 (prefix 共享). "
                f"考虑实施 prefix tree 或重新 SFT.",
                file=sys.stderr, flush=True,
            )

        # Depot " 0" (理论 select 位置不该选 depot, 但 cap=1.00 等场景 0 是出现的)
        self.depot_first_token = self._encode_first(" 0")

        # 关键单词的 first token (mask 用)
        # 注意: 检测时是字符 anchor; 应用 mask 时按 token id
        self.pipe_token = self._encode_first(" |")
        self.all_token = self._encode_first(" all")          # " all customers served"
        self.all_capital_token = self._encode_first(" All")  # 防 RL 漂移大小写

        # Verification: " Verification" 或 " **Verification" 形式
        self.verification_token = self._encode_first("Verification")
        self.verif_starred_token = self._encode_first("**Verification")

        # </think> (一般单 token, special token)
        think_close_ids = self.tokenizer("</think>", add_special_tokens=False).input_ids
        if len(think_close_ids) == 1:
            self.think_close_token = think_close_ids[0]
        else:
            self.think_close_token = None
            print(
                f"⚠️ [CVRPMask] </think> is multi-token ({think_close_ids}), "
                f"R2 think 闭合 mask 失效 (这种情况 tokenizer 不友好).",
                file=sys.stderr, flush=True,
            )

        # EOS token
        self.eos_token = self.tokenizer.eos_token_id

    def _log_token_maps(self):
        """debug: print 所有 token mapping."""
        print(f"[CVRPMask] token maps (n={self.n}):", file=sys.stderr)
        print(f"  customers (first token): "
              f"{[(c, self.cust_first_token[c]) for c in range(1, self.n + 1)]}",
              file=sys.stderr)
        print(f"  depot ' 0': {self.depot_first_token}", file=sys.stderr)
        print(f"  ' |': {self.pipe_token}", file=sys.stderr)
        print(f"  ' all': {self.all_token}", file=sys.stderr)
        print(f"  'Verification': {self.verification_token}", file=sys.stderr)
        print(f"  '</think>': {self.think_close_token}", file=sys.stderr)
        print(f"  EOS: {self.eos_token}", file=sys.stderr, flush=True)

    # ── vLLM logits_processor 接口 ──────────────────────────────────

    def __call__(
        self,
        prompt_token_ids,        # list[int] or tuple[int, ...]
        output_token_ids,        # list[int] or tuple[int, ...]
        logits: torch.Tensor,    # (vocab_size,) 1D tensor
    ) -> torch.Tensor:
        """vLLM 调用入口. 在 sample 第 t+1 token 之前修改 logits."""
        if not self.cfg.enabled:
            return logits

        # Decode 整个 output 文本 (stateless 设计, 每次重建 state)
        # 注意: prompt_ids 不需要 decode (state machine 只看 output_text)
        # 如果将来要 state 跨 prompt+output, 也要 decode prompt
        if len(output_token_ids) == 0:
            past_text = ""
        else:
            past_text = self.tokenizer.decode(
                list(output_token_ids), skip_special_tokens=False
            )

        # 重建 state + 计算 mask 决策
        state = build_state(past_text, n=self.n)
        decision = compute_mask(state, self.cfg)

        if self.debug_log and decision.mask_hit:
            print(
                f"[CVRPMask] olen={len(output_token_ids)}, section={state.section}, "
                f"visited={len(state.visited)}, decision={self._summarize_decision(decision)}",
                file=sys.stderr, flush=True,
            )

        return self._apply_decision(decision, logits)

    def _summarize_decision(self, dec: MaskDecision) -> str:
        flags = []
        if dec.select_strict:
            flags.append(f"select_strict({len(dec.select_allowed)})")
        if dec.forbid_pipe:
            flags.append("forbid_pipe")
        if dec.forbid_all_word:
            flags.append("forbid_all")
        if dec.forbid_think_close:
            flags.append("forbid_think_close")
        if dec.forbid_eos:
            flags.append("forbid_eos")
        if dec.forbid_verification_word:
            flags.append("forbid_verif")
        return "|".join(flags) if flags else "none"

    # ── Mask 应用 ──────────────────────────────────────────────────

    def _apply_decision(
        self, dec: MaskDecision, logits: torch.Tensor
    ) -> torch.Tensor:
        """把语义级 MaskDecision 应用到 token-level logits."""
        # 规则 1 (select_strict) 是 destructive 全 mask, 优先处理
        if dec.select_strict:
            allowed_token_ids = []
            for c in dec.select_allowed:
                tok = self.cust_first_token.get(c)
                if tok is not None:
                    allowed_token_ids.append(tok)

            if not allowed_token_ids:
                # Fallback: 没有任何 customer first token 可用 (异常或所有 multi-token)
                # 保守不 mask, 让 reward 兜底
                return logits

            # 全 -inf 除 allowed
            new_logits = torch.full_like(logits, float("-inf"))
            for tok in allowed_token_ids:
                new_logits[tok] = logits[tok]
            # 在 select 位置 depot " 0" 不允许 (避免 select depot hack)
            # (上面 new_logits 默认 -inf 已经 cover)
            return new_logits

        # 规则 2-5: 局部 -inf
        if dec.forbid_think_close and self.think_close_token is not None:
            logits[self.think_close_token] = float("-inf")
        if dec.forbid_eos and self.eos_token is not None:
            logits[self.eos_token] = float("-inf")
        if dec.forbid_pipe and self.pipe_token is not None:
            logits[self.pipe_token] = float("-inf")
        if dec.forbid_all_word:
            if self.all_token is not None:
                logits[self.all_token] = float("-inf")
            if self.all_capital_token is not None:
                logits[self.all_capital_token] = float("-inf")
        if dec.forbid_verification_word:
            if self.verification_token is not None:
                logits[self.verification_token] = float("-inf")
            if self.verif_starred_token is not None:
                logits[self.verif_starred_token] = float("-inf")

        return logits


# ── 公开 API ────────────────────────────────────────────────────────
__all__ = [
    "CVRPMaskProcessor",
]
