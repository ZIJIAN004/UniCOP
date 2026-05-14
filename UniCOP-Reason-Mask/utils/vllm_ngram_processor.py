"""
vLLM V1 自定义 LogitsProcessor：no_repeat_ngram_size 等价实现。

背景：
    vLLM 官方拒绝原生实现 no_repeat_ngram_size (vllm#7842 closed as not planned),
    理由是与 PagedAttention + continuous batching 的无状态高并发设计冲突。
    我们项目训练靠 n_gram=6 硬禁压制 <think> 段退化循环 (实验证据: 笔记库
    实验进度扫描 n=4/5/6/7,n=6 Parse 66%、TSP20 dist=4.93 最优),
    不可替代,必须自己实现。

    vLLM V1 (0.7+) 废弃了 V0 风格的 SamplingParams(logits_processors=[...]),
    唯一路径是全局注册 AdapterLogitsProcessor 子类 + per-request 通过
    SamplingParams.extra_args 传参。两种注册方式:
      1. pyproject.toml 的 [project.entry-points."vllm.logits_processors"]
         (需 pip install -e .,vLLM 启动自动 discover)
      2. trl vllm-serve --logits-processors utils.vllm_ngram_processor:NoRepeatNgramAdapterLP
         (CLI 显式指定,需 PYTHONPATH 能 import 到本模块)

    本项目采用方式 2 (见 auto_train.sh)。

用法 (训练端):
    GRPOConfig(
        use_vllm=True,
        vllm_mode="server",  # 或 "colocate"
        generation_kwargs={"extra_args": {"no_repeat_ngram_size": 6}},
    )

参考:
    - https://docs.vllm.ai/en/stable/features/custom_logitsprocs/
    - vllm#7842 (官方拒绝实现 no_repeat_ngram_size)
    - vllm/v1/sample/logits_processor/interface.py (AdapterLogitsProcessor API)
"""

from __future__ import annotations

from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


# ══════════════════════════════════════════════════════════════════════
#  Per-request callable: 真正执行 n-gram 禁止的逻辑
# ══════════════════════════════════════════════════════════════════════


class _NoRepeatNgramProcessor:
    """
    单条 request 的 n-gram 禁止处理器。vLLM V1 通过 Adapter 包装调用。

    签名遵循 vllm.logits_process.LogitsProcessor (3-arg 版本):
        __call__(prompt_token_ids, output_token_ids, logits) -> logits

    语义等价 HF generate 的 no_repeat_ngram_size=n:
        禁止生成会使"末尾 n 个 token"与 (prompt+output) 中某个历史位置
        的 n-gram 完全重合的 token。把这些候选的 logit 置 -inf。

    复杂度: 每 step O(N*n),其中 N = len(prompt)+len(output_so_far)。
    对 N=4096, n=6 每 step ~24k 次 tuple 比较,bs×num_gen 并行下
    相对 generation 时间可忽略 (~60ms/step)。
    """

    def __init__(self, ngram_size: int) -> None:
        self.n = int(ngram_size)

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        n = self.n
        # n<=1 退化为完全不禁止 (和关闭等价),上游已应过滤掉,防御性 return
        if n <= 1:
            return logits

        # 历史 = prompt 全部 + 已生成的 output
        # HF no_repeat_ngram_size 默认也把 prompt 当历史 (参与 n-gram 匹配)
        all_ids = list(prompt_token_ids) + list(output_token_ids)
        if len(all_ids) < n:
            # 还不够一个完整 n-gram,无从禁起
            return logits

        # 末尾 n-1 个 token: 任何候选 token t 若使 (suffix + [t]) 等于
        # all_ids 中某个已有 n-gram,则禁 t。
        suffix = tuple(all_ids[-(n - 1):])

        banned: set[int] = set()
        # i 从 0 到 len(all_ids) - n 闭区间,保证 i+n-1 是合法索引。
        # 自动排除"suffix 自己"这个平凡匹配 (i = len(all_ids) - n + 1
        # 不在 range 内)。
        for i in range(len(all_ids) - n + 1):
            if tuple(all_ids[i:i + n - 1]) == suffix:
                banned.add(all_ids[i + n - 1])

        if banned:
            # logits 是 1D (vocab_size,) — V1 per-request callable 收到的是单样本
            logits[list(banned)] = float("-inf")
        return logits


# ══════════════════════════════════════════════════════════════════════
#  Adapter: vLLM V1 全局 LogitsProcessor,包装上面的 per-request callable
# ══════════════════════════════════════════════════════════════════════


class NoRepeatNgramAdapterLP(AdapterLogitsProcessor):
    """
    vLLM V1 AdapterLogitsProcessor 子类。入口点 (entry point) 或
    `vllm-serve --logits-processors` CLI 注册后,每次收到新 request
    就调用 new_req_logits_processor 决定是否为该 request 启用 n-gram 禁止。

    per-request 控制:
        SamplingParams(extra_args={"no_repeat_ngram_size": 6})

    不设 / 设 0 / 设 1 的 request 不启用 (返回 None → vLLM 跳过)。
    """

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        """在 sampling params 进入引擎前校验一次,不合法直接报错,避免跑起来才炸。"""
        ea = params.extra_args
        if ea is None:
            return
        val = ea.get("no_repeat_ngram_size")
        if val is None:
            return
        if not isinstance(val, int) or val < 0:
            raise ValueError(
                f"no_repeat_ngram_size 必须是非负整数,当前: {val!r} "
                f"(type={type(val).__name__})"
            )

    def is_argmax_invariant(self) -> bool:
        """
        我们会把某些 token logits 置 -inf → 可能改变 argmax (最高位) → False。
        影响: greedy 解码时 vLLM 会认为此 processor 会干预选择,
        让 sampler 走非 shortcut 路径 (正确行为)。
        """
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        """
        每条新 request 调一次。从 extra_args 取 no_repeat_ngram_size,
        返回 per-request callable;None 表示该 request 不应用本 processor。
        """
        ea = params.extra_args
        if ea is None:
            return None
        n = ea.get("no_repeat_ngram_size")
        if not isinstance(n, int) or n <= 1:
            # 0/1 等价关闭,None 让 vLLM 把该 request 从 Adapter 的 state 字典中剔除
            return None
        return _NoRepeatNgramProcessor(n)


# ══════════════════════════════════════════════════════════════════════
#  最小单元测试 (不需 vLLM 引擎,可本地 python 直接跑验证逻辑)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    def _check(desc, prompt, output, n, expect_banned):
        proc = _NoRepeatNgramProcessor(n)
        vocab = 100
        logits = torch.zeros(vocab)
        out = proc(prompt, output, logits)
        banned = set(torch.where(out == float("-inf"))[0].tolist())
        ok = banned == set(expect_banned)
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {desc}: banned={sorted(banned)} "
              f"expected={sorted(expect_banned)}")
        if not ok:
            sys.exit(1)

    # 测试 1: 历史 [1,2,3,4,5,1,2,3,4],n=5,suffix=[1,2,3,4]。
    # 在位置 0 有 [1,2,3,4],其后 token=5 → 禁 5。
    _check("n=5 禁重复 5-gram", [1, 2, 3, 4, 5], [1, 2, 3, 4], 5, [5])

    # 测试 2: n=6,历史不到 6 个 → 无禁。
    _check("历史不足 n", [1, 2], [3], 6, [])

    # 测试 3: suffix 只在末尾出现一次 (自己) → 不自禁。
    _check("suffix 只匹配自己", [1, 2, 3, 4], [5], 5, [])

    # 测试 4: 多次历史匹配 → 禁多个。
    # history = [7,8,9,10, 7,8,9,11, 7,8,9],suffix=[7,8,9],n=4
    # 位置 0: [7,8,9] → 禁 10;位置 4: [7,8,9] → 禁 11。
    _check("多位置匹配", [7, 8, 9, 10, 7, 8, 9, 11], [7, 8, 9],
           4, [10, 11])

    # 测试 5: n=1 关闭 → 不禁任何。
    _check("n=1 关闭", [1, 2, 3], [4, 5], 1, [])

    # 测试 6: n=6,和训练默认值一致。
    # history 长度足够,suffix=[2,3,4,5,6],若在前面同样出现 [2,3,4,5,6] 后接 7
    # 则禁 7。
    _check("n=6 训练默认",
           [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
           [],
           6, [7])

    print("\n所有本地单元测试通过。")
