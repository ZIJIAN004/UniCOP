"""
vLLM V1 NoRepeatNgram LogitsProcessor (OpenRLHF 版)

与父目录 utils/vllm_ngram_processor.py 的区别:
    - 父目录用 trl 的 AdapterLogitsProcessor (通过 trl vllm-serve --logits-processors 注册)
    - 本版本直接用 vLLM 原生 v1 LogitsProcessor 接口, 供 OpenRLHF 的 vllm engine
      通过 custom logits processor 入口加载

注册方式 (OpenRLHF + vLLM V1):
    方案 A: 环境变量 VLLM_LOGITS_PROCESSORS
        export VLLM_LOGITS_PROCESSORS="openrlhf.custom.ngram_processor:NoRepeatNgramProcessor"
    方案 B: pyproject.toml entry point (需要把 openrlhf/ 装为 pip package)

本项目用方案 A, 见 configs/train_grpo_tsp10_1.5b.sh.

参数:
    ngram_size: 通过 SamplingParams.extra_args={"no_repeat_ngram_size": 6} 传入
"""

from __future__ import annotations
from typing import Any
import torch

from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class _NoRepeatNgramCallable:
    """单 request 的 n-gram 禁止逻辑。"""

    def __init__(self, ngram_size: int) -> None:
        self.n = int(ngram_size)

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        n = self.n
        if n <= 1:
            return logits

        # 拼接 prompt + output 作为完整历史
        all_ids = list(prompt_token_ids) + list(output_token_ids)
        if len(all_ids) < n - 1:
            return logits

        # 当前"末尾 n-1 个 token"
        tail = tuple(all_ids[-(n - 1):])

        # 扫描 all_ids 的每个长度 n 子串, 如果其前 n-1 个 = tail,
        # 则第 n 个就是 "会触发重复的禁用 token"
        banned = set()
        for i in range(len(all_ids) - n + 1):
            if tuple(all_ids[i:i + n - 1]) == tail:
                banned.add(all_ids[i + n - 1])

        if banned:
            banned_tensor = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
            logits[banned_tensor] = float("-inf")

        return logits


class NoRepeatNgramProcessor(AdapterLogitsProcessor):
    """
    vLLM V1 AdapterLogitsProcessor 子类 (per-request wrapper).

    读取 SamplingParams.extra_args["no_repeat_ngram_size"], 为每 request
    构造一个 _NoRepeatNgramCallable.
    """

    def is_argmax_invariant(self) -> bool:
        # 禁用 tokens 后可能改变 argmax, 所以不是 argmax-invariant
        return False

    def new_req_logits_processor(
        self,
        params: Any,
    ) -> RequestLogitsProcessor | None:
        extra = getattr(params, "extra_args", None) or {}
        n = extra.get("no_repeat_ngram_size", 0)
        if n is None or int(n) <= 1:
            return None
        return _NoRepeatNgramCallable(int(n))
