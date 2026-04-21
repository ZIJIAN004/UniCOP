"""DRY (Don't Repeat Yourself) LogitsProcessor，仅作用于 <think> 段。

用途：压制 reasoning 阶段的长程短语/句子重复，避免思维链陷入退化循环
(observed 上千次字面重复)。对最终答案段 (Route 输出) 不做任何干预。

算法要点：
  1. 每步生成时，只看截至当前已生成的 token 序列
  2. 若序列中尚未出现 `</think>` 的 token 序列，则应用 DRY 惩罚
  3. 对每个候选 token t：计算 "若追加 t，尾部与历史子串匹配的最大长度 L"
  4. 当 L > allowed_length 时，logit[t] -= multiplier * base ** (L - allowed_length)

实现复杂度 O(N * max_match)，N 为自 think 开始的 token 数。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from transformers import LogitsProcessor


class ThinkOnlyDRYProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        multiplier: float = 0.8,
        base: float = 1.75,
        allowed_length: int = 4,
        max_match: int = 20,
        sequence_breakers: Iterable[str] | None = None,
        think_end: str = "</think>",
    ):
        self.multiplier = float(multiplier)
        self.base = float(base)
        self.allowed_length = int(allowed_length)
        self.max_match = int(max_match)

        # 三层兜底查找 </think> 的 token 表示，优先用单 special token
        think_end_ids = self._resolve_think_end_ids(tokenizer, think_end)
        self.think_end_ids = think_end_ids
        if not self.think_end_ids:
            raise ValueError(f"tokenizer 无法编码 think_end 标记 {think_end!r}")

        # 启动诊断：让用户一眼看到门控是否可能失效
        print(
            f"[DRY] init: think_end={think_end!r} → token ids {self.think_end_ids} "
            f"({len(self.think_end_ids)} tokens); "
            f"multiplier={multiplier}, base={base}, allowed={allowed_length}, "
            f"max_match={max_match}",
            flush=True,
        )
        self._call_count = 0
        self._penalize_count = 0
        self._skip_count = 0

        breaker_ids: set[int] = set()
        for s in sequence_breakers or ():
            for tid in tokenizer.encode(s, add_special_tokens=False):
                breaker_ids.add(int(tid))
        self.breaker_ids = breaker_ids

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_think_end_ids(tokenizer, think_end: str) -> list[int]:
        """优先用 added_tokens_decoder 找单 special token；否则退回普通 encode。

        R1 蒸馏模型里 </think> 是训练时的单特殊 token，普通 encode 会被 BPE 拆成
        多个子词导致门控永远失效——先在 added tokens 里查。
        """
        # 1) 检查 added/special tokens
        added = getattr(tokenizer, "added_tokens_decoder", None)
        if added:
            for tid, tok_obj in added.items():
                if str(tok_obj) == think_end:
                    return [int(tid)]
        # 2) 检查 special_tokens_map
        sp_map = getattr(tokenizer, "special_tokens_map", {}) or {}
        for key, val in sp_map.items():
            if val == think_end:
                tid = tokenizer.convert_tokens_to_ids(val)
                if isinstance(tid, int) and tid >= 0:
                    return [tid]
        # 3) 普通 encode 兜底（多 token 也接受）
        return tokenizer.encode(think_end, add_special_tokens=False)

    def _contains_subseq(self, ids: list[int], target: list[int]) -> bool:
        n, m = len(ids), len(target)
        if m == 0 or n < m:
            return False
        for i in range(n - m + 1):
            if ids[i : i + m] == target:
                return True
        return False

    def _last_breaker_pos(self, ids: list[int]) -> int:
        if not self.breaker_ids:
            return -1
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] in self.breaker_ids:
                return i
        return -1

    def _compute_penalties(self, context: list[int]) -> dict[int, float]:
        """给 context 尾部追加候选 token 时的最长匹配长度 → 惩罚值。

        纯 numpy 向量化实现：
          1. 构造 A[p, L] = (ctx[p-L] == ctx[n-1-L])；无效位置 = False
          2. 对 axis=1 做 cumprod 求"行首连续 1 的个数" → m[p]
          3. L_ext[p] = m[p-1] + 1 (p>=1) 或 1；去掉 p=n-1 避免自匹配
          4. 对 (token, L_ext) lexsort 取每个 token 的最大 L_ext
          5. 按 L_ext > allowed_length 过滤，指数计算惩罚
        """
        n = len(context)
        if n < 2:
            return {}

        effective_max = min(self.max_match, n)
        ctx = np.asarray(context, dtype=np.int64)

        # 步骤 1：比较矩阵。A[p, L] = ctx[p-L] == ctx[n-1-L]；p<L 的格子保持 False
        A = np.zeros((n, effective_max), dtype=np.int8)
        for L in range(effective_max):
            right_val = ctx[n - 1 - L]
            A[L:, L] = (ctx[: n - L] == right_val).astype(np.int8)

        # 步骤 2：行方向 cumprod → "行首连续 1 的个数"
        # 一旦遇到 0 就整行后续全 0；sum 即原算法的 m[p]
        cum = np.cumprod(A, axis=1)
        m_arr = cum.sum(axis=1)

        # 步骤 3：L_ext[p] = m[p-1] + 1 (p>=1), 1 (p=0)；去掉 p=n-1
        L_ext = np.empty(n, dtype=np.int64)
        L_ext[0] = 1
        L_ext[1:] = m_arr[:-1] + 1
        tokens = ctx[: n - 1]
        L_ext_valid = L_ext[: n - 1]

        # 步骤 4：按 (token asc, L_ext desc) 排序，unique 取首项即每 token 的最大 L_ext
        order = np.lexsort((-L_ext_valid, tokens))
        sorted_tokens = tokens[order]
        sorted_L = L_ext_valid[order]
        first_of_group = np.concatenate(
            ([True], sorted_tokens[1:] != sorted_tokens[:-1])
        )
        best_tokens = sorted_tokens[first_of_group]
        best_L = sorted_L[first_of_group]

        # 步骤 5：过滤 + 指数
        mask = best_L > self.allowed_length
        pen_tokens = best_tokens[mask]
        pen_L = best_L[mask]
        if pen_tokens.size == 0:
            return {}

        pen_values = self.multiplier * np.power(
            self.base, (pen_L - self.allowed_length).astype(np.float64)
        )
        return dict(zip(pen_tokens.tolist(), pen_values.tolist()))

    # ------------------------------------------------------------------
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        think_end = list(self.think_end_ids)
        self._call_count += 1
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            if self._contains_subseq(ids, think_end):
                self._skip_count += 1
                continue

            search_start = self._last_breaker_pos(ids) + 1
            context = ids[search_start:]
            if len(context) < 2:
                continue

            penalties = self._compute_penalties(context)
            if not penalties:
                continue

            self._penalize_count += 1
            idx = torch.tensor(list(penalties.keys()), device=scores.device, dtype=torch.long)
            val = torch.tensor(list(penalties.values()), device=scores.device, dtype=scores.dtype)
            scores[b].index_add_(0, idx, -val)

        # 每 500 次调用打印一次简报，肉眼判断是否正常工作
        if self._call_count % 500 == 0:
            print(
                f"[DRY] calls={self._call_count} "
                f"penalized={self._penalize_count} "
                f"skip_after_think={self._skip_count}",
                flush=True,
            )
        return scores
