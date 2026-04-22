"""
vLLM 烟雾测试 - 确认 vLLM + GPU + 小模型能打通

用法:
    conda activate /Data04/yangzhihan/envs/openrlhf_env

    # 方式 A: 用服务器上已有的 SFT ckpt
    python smoke_test_vllm.py --model /Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/output_sft_auto_20260423_024302/merged_model

    # 方式 B: 用 HF hub 自动下载小模型 (需要联网,约 1GB)
    python smoke_test_vllm.py --model Qwen/Qwen2.5-0.5B

注意: 这只是验证 vLLM 能跑起来,不是验证 OpenRLHF 训练链路
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="模型路径 (本地路径或 HF hub id)"
    )
    parser.add_argument("--gpu_mem", type=float, default=0.5)
    parser.add_argument("--max_model_len", type=int, default=512)
    args = parser.parse_args()

    print("=" * 60)
    print(f"vLLM smoke test: model = {args.model}")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    print("[1/3] Loading model...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=True,       # 跳过 CUDA Graph 编译,加速测试启动
        trust_remote_code=True,
    )
    print(f"      load OK ({time.time() - t0:.1f}s)")

    print("[2/3] Running generation...")
    prompts = [
        "Hello, world. The capital of France is",
        "Solving TSP step by step: given 5 cities, the shortest route is",
    ]
    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
    )
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"      generate OK ({time.time() - t1:.1f}s)")

    print("[3/3] Results:")
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        print(f"  [{i}] prompt:     {output.prompt[:50]}...")
        print(f"      completion: {text}")
        print()

    print("=" * 60)
    print(f"TOTAL: {time.time() - t0:.1f}s  ->  vLLM + GPU 链路正常")
    print("=" * 60)


if __name__ == "__main__":
    main()
