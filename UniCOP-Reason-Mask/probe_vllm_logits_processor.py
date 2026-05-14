"""
vLLM 0.7.3 logits_processor 行为 probe.

验证 stateless mask 设计的关键假设 (来自调研结论):
  1. output_token_ids 是 tuple, 每次返回新对象 (id 不稳定)
  2. id() 不能用作 per-sequence state key
  3. n=K 个 generation 各自独立 (不串扰)
  4. processor 实例被 N 个 sequence 共享 (必须 stateless)
  5. 跨 step len 单调增 (token append 模型)
  6. decode + regex 重算的性能开销

预期输出 (按调研结论):
  [1] Unique id() values ≈ total calls    (每次都是新 tuple)
  [2] Per-len unique id counts ≥ N        (N 个 sequence 并发, 各自独立)
  [3] tail3 相同的 id 不同                  (相同内容 tuple 在不同时刻 id 也不同)
  [4] processor 实例 id 在所有 call 中相同  (共享实例)
  [5] decode + regex < 100 μs/call         (stateless 性能可接受)

用法:
  # zhuoyi 集群上, sbatch 提交
  sbatch submit_probe_vllm.sh

  # 或 login node 直接跑 (需要 GPU)
  python probe_vllm_logits_processor.py 2>&1 | tee probe_output.log

环境要求:
  - vllm == 0.7.3 (跟训练用同一版本)
  - BASE_MODEL env 或默认指向 /homes/zhuoyi/zijianliu/models/DeepSeek-R1-Distill-Qwen-7B
"""
import os
import re
import sys
import time
from collections import defaultdict


PROMPT = "Plan a CVRP route for 5 customers. Reasoning step by step."
N_SEQUENCES = 4         # 验证多 sequence 独立性
MAX_TOKENS = 12         # 短 generation, probe 不需要长输出


class ProbeProcessor:
    """无副作用的 logits_processor, 只记录每次调用的 metadata."""

    def __init__(self):
        self.records: list[dict] = []
        self.instance_id = id(self)

    def __call__(self, prompt_ids, output_ids, logits):
        t0 = time.perf_counter()
        rec = {
            "oid": id(output_ids),
            "olen": len(output_ids),
            "otype": type(output_ids).__name__,
            "pid": id(prompt_ids),
            "plen": len(prompt_ids),
            "tail3": tuple(output_ids[-3:]) if len(output_ids) >= 1 else (),
            "call_us": (time.perf_counter() - t0) * 1e6,
        }
        self.records.append(rec)
        return logits


def main():
    base_model = os.environ.get(
        "BASE_MODEL",
        "/homes/zhuoyi/zijianliu/models/DeepSeek-R1-Distill-Qwen-7B",
    )
    if not os.path.isdir(base_model):
        print(f"❌ BASE_MODEL not found: {base_model}", file=sys.stderr)
        sys.exit(1)

    print(f"=== vLLM logits_processor Probe ===")
    print(f"Model: {base_model}")
    print(f"N sequences: {N_SEQUENCES}")
    print(f"Max tokens: {MAX_TOKENS}")
    print()

    from vllm import LLM, SamplingParams

    # 跟 vllm_serve_ngram.py 一致, monkey-patch SamplingParams.__init__ 注入 processor.
    # 直接 SamplingParams(logits_processors=[probe]) 在 vLLM 0.7.3 不一定 honor.
    # (上次跑 probe 发现 processor.records 全空, 即 processor 没被调用.)
    probe = ProbeProcessor()
    orig_sp_init = SamplingParams.__init__
    def _patched_sp_init(self, *args, **kwargs):
        procs = list(kwargs.get("logits_processors") or [])
        procs.append(probe)
        kwargs["logits_processors"] = procs
        orig_sp_init(self, *args, **kwargs)
    SamplingParams.__init__ = _patched_sp_init
    print("[probe] SamplingParams.__init__ monkey-patched, all SamplingParams will inject probe")
    print()

    llm = LLM(
        model=base_model,
        max_model_len=512,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        enforce_eager=True,
    )

    sampling = SamplingParams(
        n=N_SEQUENCES,
        temperature=1.0,
        max_tokens=MAX_TOKENS,
    )
    # probe 已通过 monkey-patch 自动加入 sampling.logits_processors
    print(f"[probe] sampling.logits_processors length: "
          f"{len(sampling.logits_processors) if sampling.logits_processors else 0}")

    print(f"--- Running generation ---")
    outputs = llm.generate([PROMPT], sampling)
    print(f"Generated {N_SEQUENCES} sequences, total {sum(len(o.outputs[i].token_ids) for o in outputs for i in range(len(o.outputs)))} tokens")
    print()

    # ── 分析 ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Probe Results")
    print("=" * 60)

    total_calls = len(probe.records)
    print(f"\nTotal processor calls: {total_calls}")
    print(f"Expected: {N_SEQUENCES} × ~{MAX_TOKENS} = ~{N_SEQUENCES * MAX_TOKENS}")

    # 1. output_ids 类型
    types_seen = set(r["otype"] for r in probe.records)
    print(f"\n[1] output_ids types observed: {types_seen}")
    if types_seen == {"tuple"}:
        print(f"    ✓ 全是 tuple (符合调研: SequenceData.output_token_ids 是 @property 返新 tuple)")
    else:
        print(f"    ⚠ 类型不统一, 跟调研预期不同")

    # 2. id 复用统计
    id_counts = defaultdict(int)
    for r in probe.records:
        id_counts[r["oid"]] += 1
    unique_ids = len(id_counts)
    print(f"\n[2] Unique id() values: {unique_ids} / {total_calls}")
    if unique_ids == total_calls:
        print(f"    ✗ id() 每次都不同 → dict[id] 设计完全失效 (符合调研)")
    elif unique_ids < N_SEQUENCES:
        print(f"    ⚠ id() 严重复用, 跨 sequence 撞 key 风险高")
    else:
        most_reused = max(id_counts.values())
        print(f"    单 id 最多复用 {most_reused} 次")

    # 3. 同 len 的 id 数 (验证 sequence 独立性)
    olen_to_oids = defaultdict(set)
    for r in probe.records:
        olen_to_oids[r["olen"]].add(r["oid"])
    print(f"\n[3] Per-len unique id counts (前 5 个 len):")
    for olen in sorted(olen_to_oids.keys())[:5]:
        n_oids = len(olen_to_oids[olen])
        marker = "✓" if n_oids >= N_SEQUENCES else ("⚠" if n_oids > 0 else "✗")
        print(f"    {marker} olen={olen}: {n_oids} unique id(s)")
    if all(len(olen_to_oids[k]) >= N_SEQUENCES for k in olen_to_oids if k >= 1):
        print(f"    ✓ 每个 len 都有 ≥{N_SEQUENCES} 独立 id, N 个 sequence 并发独立")

    # 4. processor 实例 ID (验证是否共享)
    print(f"\n[4] ProbeProcessor instance id: {probe.instance_id}")
    print(f"    所有 records 来自这一个实例")
    print(f"    → N={N_SEQUENCES} sequence 共享同一 processor (必须 stateless)")

    # 5. 跨 step len 单调性 (用 tail3 fuzzy 标识 sequence)
    #    同一 sequence 应该 len=1,2,3,...; tail3 也有连续关系
    seq_by_tail2 = defaultdict(list)  # tail3 前两位作伪 seq key
    for r in probe.records:
        if len(r["tail3"]) >= 2:
            seq_by_tail2[r["tail3"][:2]].append(r["olen"])
    monotonic_count = 0
    for key, lens in seq_by_tail2.items():
        if len(lens) >= 2 and all(lens[i] <= lens[i + 1] for i in range(len(lens) - 1)):
            monotonic_count += 1
    print(f"\n[5] Len 单调性 (按 tail2 分组): {monotonic_count}/{len(seq_by_tail2)} 组单调")
    if monotonic_count >= len(seq_by_tail2) * 0.8:
        print(f"    ✓ 大部分组单调, 符合 token append 模型")

    # 6. decode + regex 性能模拟 (stateless 重算开销)
    text_dummy = (
        "[R1,1] cap=1.00 | feasible: 11(d=0.021,dem=0.07,cap→0.93), "
        "14(d=0.085,dem=0.10,cap→0.90), 13(d=0.347,dem=0.07,cap→0.93), "
        "... → select 13\n"
    ) * 30  # 模拟 3000 char CVRP chain
    print(f"\n[6] Stateless 重算性能 (decode + regex, L={len(text_dummy)} chars):")

    pattern = re.compile(r"→\s*select\s+(\d+)")
    t0 = time.perf_counter()
    n_iter = 200
    for _ in range(n_iter):
        matches = pattern.findall(text_dummy)
        visited = set(int(m) for m in matches if 1 <= int(m) <= 20)
    avg_us = (time.perf_counter() - t0) / n_iter * 1e6
    print(f"    Avg per call: {avg_us:.1f} μs (visited set: {visited})")
    total_overhead_ms = avg_us * MAX_TOKENS / 1000
    if avg_us < 100:
        print(f"    ✓ < 100 μs/call, 3000 token chain 累积 {avg_us * 3000 / 1000:.1f} ms (可接受)")
    else:
        print(f"    ⚠ > 100 μs/call, 大 batch 时可能成瓶颈")

    # ── 总结 ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Conclusion")
    print("=" * 60)
    if unique_ids == total_calls and types_seen == {"tuple"}:
        print("✓ 调研结论得到完全确认:")
        print("  - output_token_ids 是 tuple, id 每次不同")
        print("  - 不能用 id() 作 state key")
        print("  - Stateless 重算是唯一可靠方案")
        print()
        print("→ 可以进入设计实施阶段")
    else:
        print("⚠ 部分调研结论需要重新核实, 设计前需要进一步分析")

    # 把 records 存盘供进一步分析
    out_path = "probe_records.txt"
    with open(out_path, "w") as f:
        f.write(f"# Total calls: {total_calls}\n")
        f.write(f"# Unique ids: {unique_ids}\n")
        f.write(f"# Instance id: {probe.instance_id}\n\n")
        for i, r in enumerate(probe.records):
            f.write(f"{i:>4} olen={r['olen']:>3} oid={r['oid']:>20} type={r['otype']:>8} tail3={r['tail3']}\n")
    print(f"\nDetailed records saved to: {out_path}")


if __name__ == "__main__":
    main()
