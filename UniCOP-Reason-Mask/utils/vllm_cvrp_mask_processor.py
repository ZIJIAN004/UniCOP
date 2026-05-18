"""
vLLM logits_processor: CVRP constrained-decoding mask.

桥接层: 把 cvrp_mask_state.py 的语义级 MaskDecision 转成 vLLM token-level
logits 操作 (logits[token_id] = -inf).

设计要点 (基于调研结论):
  1. Stateless: 每次 __call__ 从 prompt_ids + output_ids 全量重建 state
     (vLLM 0.7.3 worker process + output_ids 是新 tuple, 不能 cache state)
  2. Per-call 局部缓存优化: 用 (id(output_ids), len(output_ids)) 作 key
     避免完全冗余重建. tuple 每次新建虽然 id 不稳, 但单次 __call__ 内 id 稳定.
  3. Prefix tree mask 处理 multi-token customer (Qwen tokenizer 把 10-20
     拆成 [" 1", "0"-"9"], 跟 customer 1 的 first token 共享):
     - 规则 1 第 1 token mask: 按 unvisited 的 first tokens 允许
     - Prefix tree 第 2-N token mask: 检查 output_ids 末尾是否是某 customer
       的不完整 prefix; 是的话只允许"延续到 unvisited customer"的 token
     - 这样可以严格 ban 模型在 visited customer 上 "逃逸" (例如 visited={11},
       模型写 " 1" 后第 2 token "1" 被 ban → 强制选 0 或 2-9 或结束符)
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

import re
import sys
from typing import Optional

import torch

from utils.cvrp_mask_state import (
    MaskConfig,
    MaskDecision,
    build_state,
    compute_mask,
)


# 末尾匹配 "→ select 数字+" (没有空格结尾) → 正在 multi-token select 中
# 跟 SELECT_TRIGGER_RE (末尾 "→ select " 空格结尾, 触发规则 1) 互补
_PARTIAL_SELECT_TAIL_RE = re.compile(r"→\s*[Ss]elect\s+\d+$")


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
        """预计算 5 条规则 + prefix tree 需要的 token id."""
        # Customer 1..n 的完整 token 序列 (带空格前缀, 跟 CVRP chain 里 " select N" 一致)
        # multi-token (如 " 13" = [" 1", "3"]) 由 prefix tree 处理 (_compute_partial_select)
        self.cust_tokens: dict[int, list[int]] = {}
        self.cust_first_token: dict[int, Optional[int]] = {}  # 保留兼容性 (规则 1 用)
        self.multi_token_customers: list[int] = []
        # first_token → customer 列表 (反向索引, 调试用)
        self.first_to_customers: dict[int, list[int]] = {}
        # 最长 customer token 数 (prefix tree lookback 上限)
        self.max_cust_tok_len: int = 1
        for c in range(1, self.n + 1):
            ids = self.tokenizer(f" {c}", add_special_tokens=False).input_ids
            if not ids:
                self.cust_first_token[c] = None
                continue
            self.cust_tokens[c] = list(ids)
            self.cust_first_token[c] = ids[0]
            self.first_to_customers.setdefault(ids[0], []).append(c)
            if len(ids) > 1:
                self.multi_token_customers.append(c)
            self.max_cust_tok_len = max(self.max_cust_tok_len, len(ids))

        # select 后的合法结束符 token (空格/换行/管道符等), 用于 partial_select
        # "→ select 1" 写完 1 后, 下个 token 可能是 "\n", "[", " ", " |", " |\n" 等
        _end_strs = [" ", "\n", "[", " |", "|", "\t", " \n", ".", ",", ";"]
        self.select_end_tokens: list[int] = []
        for s in _end_strs:
            ids = self.tokenizer(s, add_special_tokens=False).input_ids
            self.select_end_tokens.extend(ids)
        self.select_end_tokens = list(set(self.select_end_tokens))

        if self.multi_token_customers:
            print(
                f"[CVRPMask] customers {self.multi_token_customers} are multi-token; "
                f"prefix tree mask 启用 (第 2-N token 按 unvisited 限制延续)",
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

        # 第一次 __call__ 强制 print 确认 vLLM 真 invoke processor (诊断 attach 是否成功)
        if not getattr(self, "_first_call_logged", False):
            self._first_call_logged = True
            print(
                f"!!! [CVRPMask] FIRST __call__ INVOKED: "
                f"olen={len(output_token_ids)}, otype={type(output_token_ids).__name__}, "
                f"plen={len(prompt_token_ids)}, "
                f"logits.shape={tuple(logits.shape) if hasattr(logits, 'shape') else 'N/A'}, "
                f"multi_token_cust={self.multi_token_customers}",
                file=sys.stderr, flush=True,
            )
        # 关键诊断: 第一次 decode call (olen>=1) 时 print, 验证 vLLM 是否在 decode 时也 call
        # 如果 vLLM 只在 prefill (olen=0) 时 call, 这行永远不出现 → mask 只影响第 1 token,
        # 后续 token vanilla → 解释 dup=66
        if len(output_token_ids) >= 1 and not getattr(self, "_decode_call_logged", False):
            self._decode_call_logged = True
            print(
                f"!!! [CVRPMask] FIRST DECODE __call__ (olen>=1) INVOKED: "
                f"olen={len(output_token_ids)}, last_tok={output_token_ids[-1] if output_token_ids else None}, "
                f"instance_id={id(self)}",
                file=sys.stderr, flush=True,
            )
        # 周期性 call counter (每 1000 call print 一次, 验证 instance 是否真持续被 call)
        self._call_count = getattr(self, "_call_count", 0) + 1
        if self._call_count in (10, 100, 1000, 5000):
            print(
                f"!!! [CVRPMask] call_count={self._call_count}, "
                f"current olen={len(output_token_ids)}, instance_id={id(self)}",
                file=sys.stderr, flush=True,
            )

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

        # 第 1 次进入 SECTION_2 时强制 print (验证模型有没有写 anchor)
        if (state.section == "SECTION_2"
                and not getattr(self, "_section2_logged", False)):
            self._section2_logged = True
            print(
                f"!!! [CVRPMask] FIRST SECTION_2 INVOKED: "
                f"olen={len(output_token_ids)}, visited={len(state.visited)}, "
                f"select_trigger={state.select_trigger_now}, "
                f"past_text_tail={past_text[-80:]!r}",
                file=sys.stderr, flush=True,
            )
        # 第 1 次 select_strict 触发时 print (验证 mask 真在 SECTION_2 限制 select)
        if decision.select_strict and not getattr(self, "_select_strict_logged", False):
            self._select_strict_logged = True
            print(
                f"!!! [CVRPMask] FIRST select_strict INVOKED: "
                f"olen={len(output_token_ids)}, visited={sorted(state.visited)}, "
                f"allowed_customers={decision.select_allowed}",
                file=sys.stderr, flush=True,
            )

        # ── Prefix tree: 检查是否在 multi-token customer select 中间 ──
        # 优先于 select_strict (select_strict 是第 1 token, partial_select 是第 2-N token,
        # 两者在不同 call 触发, 但 partial_select 时 select_strict 不会 fire 因为末尾不是 "→ select ")
        partial_info = None
        if state.section == "SECTION_2" and self.cfg.apply_select:
            partial_info = self._compute_partial_select(
                past_text, output_token_ids, state.visited,
            )

        if self.debug_log and (decision.mask_hit or partial_info is not None):
            extra = f" partial_select={partial_info[0]}" if partial_info else ""
            print(
                f"[CVRPMask] olen={len(output_token_ids)}, section={state.section}, "
                f"visited={len(state.visited)}, decision={self._summarize_decision(decision)}{extra}",
                file=sys.stderr, flush=True,
            )

        if partial_info is not None:
            return self._apply_partial_select(partial_info, logits)
        return self._apply_decision(decision, logits)

    # ── Prefix tree: 多 token customer select 中间状态处理 ─────────

    def _compute_partial_select(
        self, past_text: str, output_token_ids, visited: set,
    ):
        """检查 output_ids 末尾是否是某 multi-token customer 的不完整 prefix.

        Returns:
            None — 不在 partial_select 状态 (规则 1 / 其他 mask 走原逻辑)
            (n_written, candidates, unvisited) — 在 partial_select 中:
                n_written: 已写 select 进去的 token 数
                candidates: 末尾 token 序列匹配 prefix 的 customer 列表
                unvisited: 当前 unvisited 集合
        """
        # 文本快速过滤: 末尾必须匹配 "→ select 数字+"
        if not _PARTIAL_SELECT_TAIL_RE.search(past_text):
            return None

        output_list = list(output_token_ids)
        if not output_list:
            return None

        # 从最长 customer token 长度反向, 找最长 n 使 output_list[-n:] 是某 customer 的前 n 个 token
        # 注: 末尾匹配 "→ select 数字+" 已经保证 output_list 末尾是数字 token,
        #     一般 n_written = 1 (cust 10-19 的 " 1") 或 2 (后续 multi-token customer)
        max_n = min(self.max_cust_tok_len, len(output_list))
        for n_written in range(max_n, 0, -1):
            suffix = tuple(output_list[-n_written:])
            matching_customers = []
            for c, toks in self.cust_tokens.items():
                if len(toks) >= n_written and tuple(toks[:n_written]) == suffix:
                    matching_customers.append(c)
            if matching_customers:
                # 找到匹配 prefix; 但只有当存在"未完成 multi-token"候选时才需 mask
                # (即至少 1 个 candidate len(toks) > n_written, 待延续)
                has_incomplete = any(
                    len(self.cust_tokens[c]) > n_written for c in matching_customers
                )
                if has_incomplete:
                    unvisited = set(range(1, self.n + 1)) - visited
                    return (n_written, matching_customers, unvisited)
                # 全是 len == n_written (完整 customer), 不需 partial mask
                return None
        return None

    def _apply_partial_select(
        self, partial_info, logits: torch.Tensor,
    ) -> torch.Tensor:
        """根据 partial_select 状态, mask 第 2-N token 只允许延续到 unvisited."""
        n_written, candidates, unvisited = partial_info
        allowed_token_ids: set[int] = set()
        has_complete_unvisited_customer = False

        for c in candidates:
            toks = self.cust_tokens[c]
            if c in unvisited:
                if len(toks) == n_written:
                    # 已写完整 unvisited customer → 允许 select 结束符
                    has_complete_unvisited_customer = True
                elif len(toks) > n_written:
                    # 允许延续到这个 unvisited customer
                    allowed_token_ids.add(toks[n_written])
            # c in visited 时: 不允许它的延续 token
            # 这就是 prefix tree 修复 multi-token 失控的核心

        if has_complete_unvisited_customer:
            # 允许 select 结束符 (空格/换行/管道符等)
            allowed_token_ids.update(self.select_end_tokens)

        if not allowed_token_ids:
            # 所有候选都 visited → 无延续可走且当前 prefix 不是完整 customer
            # 例如 visited={1, 10-19}, partial=" 1" → 既不能结束 (1 visited)
            # 也不能延续 (10-19 都 visited). 此时 fallback 不 mask, 由 reward 罚.
            # 理论上 select_strict 不该 allow 这个 first_tok, 这里是冗余保护.
            return logits

        new_logits = torch.full_like(logits, float("-inf"))
        for tok in allowed_token_ids:
            new_logits[tok] = logits[tok]
        return new_logits

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
