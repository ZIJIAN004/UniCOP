"""启动标准 vLLM OpenAI-compatible 服务器，带 NoRepeatNgram 全局注入。

与 UniCOP-Reason/utils/vllm_serve_ngram.py 的区别：
  - 本脚本启动的是标准 vLLM OpenAI API 服务器（支持 /v1/chat/completions）
  - 那个是 TRL 的 vllm-serve（供 GRPOTrainer 使用）

用法：
    python vllm_serve_ngram.py \
        --model /path/to/model \
        --port 8000 \
        --no_repeat_ngram_size 6 \
        [其他 vLLM api_server 参数原样传递]
"""
import sys


def _extract_ngram_size():
    ngram_size = 0
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--no_repeat_ngram_size" and i + 1 < len(sys.argv):
            ngram_size = int(sys.argv[i + 1])
            i += 2
        else:
            new_argv.append(sys.argv[i])
            i += 1
    sys.argv = new_argv
    return ngram_size


def _patch_sampling_params(ngram_size: int):
    import torch
    from vllm import SamplingParams

    class _NoRepeatNgramV0:
        __slots__ = ("n",)

        def __init__(self, n: int):
            self.n = n

        def __call__(self, prompt_token_ids: list[int], output_token_ids: list[int],
                     logits: torch.Tensor) -> torch.Tensor:
            n = self.n
            all_ids = prompt_token_ids + output_token_ids
            if len(all_ids) < n:
                return logits
            suffix = tuple(all_ids[-(n - 1):])
            banned = []
            for i in range(len(all_ids) - n + 1):
                if tuple(all_ids[i: i + n - 1]) == suffix:
                    banned.append(all_ids[i + n - 1])
            if banned:
                logits[banned] = float("-inf")
            return logits

    proc = _NoRepeatNgramV0(ngram_size)
    orig_init = SamplingParams.__init__

    def patched_init(self, *args, **kwargs):
        procs = list(kwargs.get("logits_processors") or [])
        procs.append(proc)
        kwargs["logits_processors"] = procs
        orig_init(self, *args, **kwargs)

    SamplingParams.__init__ = patched_init
    print(f"[ngram] NoRepeatNgramLogitsProcessor(n={ngram_size}) injected")


if __name__ == "__main__":
    ngram_size = _extract_ngram_size()

    if ngram_size > 1:
        _patch_sampling_params(ngram_size)

    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
