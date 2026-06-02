#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_vllm_caps.py — 探测当前 vLLM 对"重复控制 / 自定义 logits processor"的支持情况。

目的: 决定 thinking 评测的"灭循环"方案 B(大 n no-repeat-ngram)能否在你这版 vLLM 上做。
用法 (集群, unicop 环境):
    conda activate unicop
    python check_vllm_caps.py
把最后的 ===== 结论 ===== 段贴回来即可。
"""
import sys


def main():
    print("=" * 64)
    try:
        import vllm
        print("vllm.__version__ =", getattr(vllm, "__version__", "?"))
    except Exception as e:
        print("✗ 导入 vllm 失败:", repr(e))
        sys.exit(1)

    # ── 引擎 v0/v1 探测 (v1 引擎改过自定义 logits processor 接口) ──
    engine = "?"
    try:
        import vllm.envs as envs
        v1 = getattr(envs, "VLLM_USE_V1", None)
        engine = f"VLLM_USE_V1={v1}"
    except Exception as e:
        engine = f"(envs 读取失败: {e!r})"
    print("引擎标记:", engine, " (1=v1 引擎, 自定义 logits processor 接口与 v0 不同)")

    # ── SamplingParams 字段存在性 ──
    print("-" * 64)
    from vllm import SamplingParams
    try:
        sp = SamplingParams()
    except Exception as e:
        print("✗ 构造 SamplingParams() 失败:", repr(e))
        sys.exit(1)

    fields = ["no_repeat_ngram_size", "logits_processors", "repetition_detection",
              "repetition_penalty", "frequency_penalty", "presence_penalty", "bad_words"]
    have = {}
    for k in fields:
        have[k] = hasattr(sp, k)
        print(f"  SamplingParams.{k:22s} : {have[k]}")

    # ── 实测能否塞自定义 logits_processor (方案 B 的前提) ──
    print("-" * 64)
    lp_ok = False
    if have.get("logits_processors"):
        try:
            def _dummy(past_token_ids, logits):
                return logits
            _ = SamplingParams(logits_processors=[_dummy], max_tokens=1)
            lp_ok = True
            print("  ✓ SamplingParams(logits_processors=[fn]) 构造成功 (v0 风格回调接口)")
        except Exception as e:
            print("  ✗ 传 logits_processors 构造报错:", repr(e))
            print("    (可能是 v1 引擎: 自定义 logits processor 需走新的注册接口, 非此参数)")
    else:
        print("  SamplingParams 无 logits_processors 字段")

    # ── repetition_detection 字段细看 (注: 这是'检测后停', 非'阻止 n-gram') ──
    if have.get("repetition_detection"):
        print("  注: repetition_detection 是'检测到循环就提前停', 解决不了 thinking 答案被截断")

    # ── 结论 ──
    print("=" * 64)
    print("===== 结论 =====")
    if have.get("no_repeat_ngram_size"):
        print("方案B 直接可用: 这版居然有原生 no_repeat_ngram_size, 直接在采样里设大 n 即可。")
    elif lp_ok:
        print("方案B 可做(走自定义 logits processor): logits_processors 可用 → 我写个大 n")
        print("        no-repeat-ngram 处理器接进 evaluate.py 即可。把本段贴回来。")
    else:
        print("方案B 这版走不通: 既无原生 no_repeat_ngram_size, 自定义 logits_processors 也")
        print("        塞不进 (大概率 v1 引擎)。需换路子: budget forcing(A) 或 调整 vLLM。把本段贴回来。")
    print("=" * 64)


if __name__ == "__main__":
    main()
