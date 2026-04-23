"""
vLLM V1 NoRepeatNgram LogitsProcessor (OpenRLHF 版)

与父目录 utils/vllm_ngram_processor.py 的区别:
    - 父目录用 trl 的 AdapterLogitsProcessor (通过 trl vllm-serve 注册)
    - 本版本直接用 vLLM 原生 v1 LogitsProcessor 接口, 供 OpenRLHF 的 vllm engine
      通过 custom logits processor 入口加载

注册方式:
    OpenRLHF 0.10.2 不透传 vLLM 的 --logits-processors CLI 参数,
    也不存在 VLLM_LOGITS_PROCESSORS 环境变量。

    可行方案:
    A. pyproject.toml entry point (需 pip install -e . 注册)
       [project.entry-points."vllm.logits_processors"]
       no_repeat_ngram = "openrlhf.custom.ngram_processor:NoRepeatNgramProcessor"

    B. 修改 OpenRLHF 源码的 vllm_engine.py, 在 create_vllm_engines() 中
       传入自定义 processor

    当前推荐方案 A。在服务器上执行:
       cd /Data04/.../UniCOP-Reason/openrlhf && pip install -e .

参数获取:
    1. 优先从 SamplingParams.extra_args["no_repeat_ngram_size"] 读取
    2. fallback 到环境变量 NO_REPEAT_NGRAM_SIZE (默认 6)
    这样即使 OpenRLHF 无法透传 extra_args, 只要设了环境变量就能生效
"""

from __future__ import annotations
import os
from typing import Any
import torch

from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

_DEFAULT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "6"))


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

        all_ids = list(prompt_token_ids) + list(output_token_ids)
        if len(all_ids) < n - 1:
            return logits

        tail = tuple(all_ids[-(n - 1):])

        banned = set()
        for i in range(len(all_ids) - n + 1):
            if tuple(all_ids[i:i + n - 1]) == tail:
                banned.add(all_ids[i + n - 1])

        if banned:
            banned_tensor = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
            logits[banned_tensor] = float("-inf")

        return logits


class NoRepeatNgramProcessor(AdapterLogitsProcessor):
    """vLLM V1 AdapterLogitsProcessor 子类。"""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: Any,
    ) -> RequestLogitsProcessor | None:
        extra = getattr(params, "extra_args", None) or {}
        n = extra.get("no_repeat_ngram_size", _DEFAULT_NGRAM_SIZE)
        if n is None or int(n) <= 1:
            return None
        return _NoRepeatNgramCallable(int(n))
