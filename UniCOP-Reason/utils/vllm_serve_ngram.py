"""
Wrapper: monkey-patch SamplingParams 注入 NoRepeatNgram，然后启动 trl vllm-serve。

vLLM 0.7.3 (V0) 的 SamplingParams 支持 logits_processors=[callable, ...]，
但 TRL 0.16.0 的 trl vllm-serve 不暴露此参数。
本脚本在 SamplingParams.__init__ 里注入 V0 兼容的 ngram processor，
对 TRL 透明——它照常构造 SamplingParams，processor 自动附加。

用法 (替代 trl vllm-serve):
    python utils/vllm_serve_ngram.py \
        --no_repeat_ngram_size 6 \
        --model /path/to/model \
        --port 8000 \
        [其他 trl vllm-serve 参数原样传递]
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
        """vLLM V0 logits_processor: 3-arg signature (prompt_ids, output_ids, logits).
        V0 uses inspect.signature to detect arg count; 3-arg gets separated lists."""
        __slots__ = ("n",)

        def __init__(self, n: int):
            self.n = n

        def __call__(self, prompt_token_ids: list[int], output_token_ids: list[int],
                     logits: torch.Tensor) -> torch.Tensor:
            n = self.n
            all_ids = prompt_token_ids + output_token_ids
            if len(all_ids) < n:
                return logits
            suffix = tuple(all_ids[-(n - 1) :])
            banned = []
            for i in range(len(all_ids) - n + 1):
                if tuple(all_ids[i : i + n - 1]) == suffix:
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

    # TRL 0.16.0 的 vllm_serve 入口
    import runpy

    runpy.run_module("trl.scripts.vllm_serve", run_name="__main__")
