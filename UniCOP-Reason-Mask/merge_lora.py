"""把 GRPO 训练出的 LoRA adapter 合并进基座 → 全量 merged_model (eval 吃这个)。

从 launcher 内联 heredoc 抽出来独立成脚本, 修两个老毛病:
  1. 内联 merge 不落盘 → zhihan 无 SLURM, tmux 滚屏/断连后"失败且无 log";
     本脚本所有输出由 launcher tee 到 $OUT_DIR/merge.log, 失败可复盘。
  2. CPU 合并 4B (base bf16 ~8GB + adapter + merged 副本 → 峰值 16-24GB) 易被
     OOM killer SIGKILL(exit 137 无 traceback)。这里 low_cpu_mem_usage=True 降峰值,
     且打印可用 RAM, 真 OOM 时日志里能看到内存紧张, 不再是哑崩。

校验 (CLAUDE.md: ZeRO-3+LoRA 须验证保存后权重非空):
  config.json 存在 + 至少一个非空 *.safetensors + 抽样张量非全零。

用法:
    python merge_lora.py --adapter <adapter_dir> --merged <out_dir>
    # base 默认从 adapter_config.json 的 base_model_name_or_path 读; 可 --base 覆盖
"""
import argparse
import json
import os
import sys
import traceback


def _avail_ram_gb():
    """Linux 可用内存 (GB); 读不到返回 None。"""
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 1024 / 1024
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter 目录 (含 adapter_config.json)")
    ap.add_argument("--merged", required=True, help="合并后全量权重输出目录")
    ap.add_argument("--base", default=None, help="基座路径 (默认从 adapter_config 读)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"[merge] adapter = {args.adapter}", flush=True)
    print(f"[merge] merged  = {args.merged}", flush=True)

    # ── 前置检查: adapter 存在 ──────────────────────────────────────────
    acfg_path = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.isfile(acfg_path):
        print(f"[merge][FATAL] 缺 adapter_config.json: {acfg_path}", flush=True)
        sys.exit(2)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base or json.load(open(acfg_path))["base_model_name_or_path"]
    print(f"[merge] base    = {base}", flush=True)
    if not (os.path.isdir(base) or "/" not in base):  # 本地目录或 HF repo id
        print(f"[merge][WARN] base 路径看起来不是本地目录, 若非 HF repo id 会失败: {base}", flush=True)

    ram = _avail_ram_gb()
    try:
        import peft, transformers
        print(f"[merge] env: torch={torch.__version__} transformers={transformers.__version__} "
              f"peft={peft.__version__}  可用RAM={f'{ram:.1f}GB' if ram else '?'}", flush=True)
    except Exception:
        pass
    if ram is not None and ram < 20:
        print(f"[merge][WARN] 可用 RAM 仅 {ram:.1f}GB, CPU 合并 4B 峰值可能不够 → 若进程被 SIGKILL "
              f"(exit 137) 即为 CPU OOM, 不是脚本 bug。考虑空出内存或换大内存节点。", flush=True)

    dtype = getattr(torch, args.dtype)
    try:
        print("[merge] 加载 base (CPU, low_cpu_mem_usage)...", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=dtype, trust_remote_code=True,
            device_map="cpu", low_cpu_mem_usage=True)
        print("[merge] 挂 adapter...", flush=True)
        pm = PeftModel.from_pretrained(m, args.adapter)
        print("[merge] merge_and_unload...", flush=True)
        merged = pm.merge_and_unload()
        os.makedirs(args.merged, exist_ok=True)
        print("[merge] save_pretrained (safetensors)...", flush=True)
        merged.save_pretrained(args.merged, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True).save_pretrained(args.merged)
        print("[merge] 权重已写出, 开始校验...", flush=True)
    except Exception:
        print("[merge][FATAL] 合并过程抛异常:", flush=True)
        traceback.print_exc()
        sys.exit(3)

    # ── 校验: config + 非空权重 + 抽样非全零 ────────────────────────────
    if not os.path.isfile(os.path.join(args.merged, "config.json")):
        print(f"[merge][FATAL] 校验失败: 缺 {args.merged}/config.json", flush=True)
        sys.exit(4)
    shards = [f for f in os.listdir(args.merged)
              if f.endswith(".safetensors") and os.path.getsize(os.path.join(args.merged, f)) > 0]
    if not shards:
        print(f"[merge][FATAL] 校验失败: {args.merged} 无非空 *.safetensors", flush=True)
        try:
            print("  目录内容:", os.listdir(args.merged), flush=True)
        except Exception:
            pass
        sys.exit(5)
    # 抽样一个张量确认非全零 (防 ZeRO-3+LoRA 存出空壳)
    try:
        from safetensors import safe_open
        sp = os.path.join(args.merged, sorted(shards)[0])
        with safe_open(sp, framework="pt") as f:
            keys = list(f.keys())
            sample = None
            for k in keys:                       # 优先抽一个 weight 张量
                if k.endswith(".weight"):
                    sample = k
                    break
            sample = sample or keys[0]
            t = f.get_tensor(sample)
            nz = float(t.abs().sum())
        print(f"[merge] 抽样张量 '{sample}' (shard {sorted(shards)[0]}): abs_sum={nz:.3e} "
              f"{'✅ 非空' if nz > 0 else '❌ 全零!'}", flush=True)
        if nz <= 0:
            print("[merge][FATAL] 抽样张量全零, merged 权重无效 (疑似空壳)", flush=True)
            sys.exit(6)
    except ImportError:
        print("[merge][WARN] 无 safetensors 库, 跳过抽样非零校验 (文件非空已过)", flush=True)

    total_gb = sum(os.path.getsize(os.path.join(args.merged, f)) for f in shards) / 1024**3
    print(f"[merge] ✅ 完成且权重非空: {args.merged}  ({len(shards)} shards, {total_gb:.2f}GB)", flush=True)
    print("=" * 72, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
