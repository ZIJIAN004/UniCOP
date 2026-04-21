"""Think-only repetition_penalty LogitsProcessor。

仅对 `<think>...</think>` 区间内的 token 施加 HF 风格 rep_penalty；
`</think>` 出现之后不再干预，Route 输出段保持原始 logits。

额外：支持 exempt 白名单——列入白名单的 token id 即使在 think 内也不扣分。
典型场景：depot token `0`（以及其带空格变体 ` 0`），避免模型"不想回到 0"。
"""

from __future__ import annotations

from typing import Iterable

import torch
from transformers import LogitsProcessor


class ThinkOnlyRepPenaltyProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        penalty: float = 1.2,
        exempt_tokens: Iterable[str] | None = None,
        think_end: str = "</think>",
    ):
        self.penalty = float(penalty)
        if self.penalty <= 0:
            raise ValueError("penalty must be > 0")

        self.think_end_ids = self._resolve_think_end_ids(tokenizer, think_end)
        if not self.think_end_ids:
            raise ValueError(f"tokenizer 无法编码 think_end 标记 {think_end!r}")

        exempt_ids: set[int] = set()
        exempt_strs: list[str] = []
        for s in (exempt_tokens or ()):
            if not s:
                continue
            for tid in tokenizer.encode(s, add_special_tokens=False):
                exempt_ids.add(int(tid))
            exempt_strs.append(s)
        self.exempt_ids = exempt_ids

        print(
            f"[ThinkRep] init: penalty={penalty}, think_end_ids={self.think_end_ids} "
            f"({len(self.think_end_ids)} tokens), "
            f"exempt_strings={exempt_strs}, exempt_token_ids={sorted(self.exempt_ids)}",
            flush=True,
        )
        self._call_count = 0
        self._penalize_count = 0
        self._skip_count = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_think_end_ids(tokenizer, think_end: str) -> list[int]:
        """R1 蒸馏模型里 </think> 是单 special token；普通 encode 会被 BPE 拆开导致门控失效。"""
        added = getattr(tokenizer, "added_tokens_decoder", None)
        if added:
            for tid, tok_obj in added.items():
                if str(tok_obj) == think_end:
                    return [int(tid)]
        sp_map = getattr(tokenizer, "special_tokens_map", {}) or {}
        for _, val in sp_map.items():
            if val == think_end:
                tid = tokenizer.convert_tokens_to_ids(val)
                if isinstance(tid, int) and tid >= 0:
                    return [tid]
        return tokenizer.encode(think_end, add_special_tokens=False)

    def _contains_subseq(self, ids: list[int], target: list[int]) -> bool:
        n, m = len(ids), len(target)
        if m == 0 or n < m:
            return False
        for i in range(n - m + 1):
            if ids[i : i + m] == target:
                return True
        return False

    # ------------------------------------------------------------------
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self._call_count += 1
        think_end = list(self.think_end_ids)
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            if self._contains_subseq(ids, think_end):
                self._skip_count += 1
                continue

            seen = set(ids)
            seen.difference_update(self.exempt_ids)
            if not seen:
                continue

            self._penalize_count += 1
            idx = torch.tensor(list(seen), device=scores.device, dtype=torch.long)
            row = scores[b]
            vals = row.index_select(0, idx)
            # HF 风格：正 logits 除 penalty；负 logits 乘 penalty（更负）
            penalized = torch.where(vals > 0, vals / self.penalty, vals * self.penalty)
            row.scatter_(0, idx, penalized)

        if self._call_count % 500 == 0:
            print(
                f"[ThinkRep] calls={self._call_count} "
                f"penalized={self._penalize_count} "
                f"skip_after_think={self._skip_count}",
                flush=True,
            )
        return scores
