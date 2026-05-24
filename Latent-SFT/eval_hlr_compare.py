"""
HLR vs Baseline 对比 eval.

流程:
  Step 1: baseline 模型路径解决
            - 用户传 --baseline_model: 直接用
            - 不传: 把 HLR LoRA merge 到 {hlr_checkpoint}/baseline_merged/ (有缓存就跳过)
  Step 2: 跑 baseline (UniCOP-Reason/evaluate.py --backend local)
  Step 3: 跑 HLR    (UniCOP-Reason/evaluate.py --backend hlr)
  Step 4: 读两个 JSON, 并排打印 + 保存 compare.json
            - 任务指标 (parse / feasibility / dist)
            - 显式 token 节省比例
            - FLOPs 节省比例 (理论估算: 2 × params × tokens)
            - wall-clock 节省 (per-sample HLR + 两轮总耗时)

用法 (cwd = UniCOP 根目录):
  python Latent-SFT/eval_hlr_compare.py \\
    --hlr_checkpoint /path/to/output_hlr/checkpoint-final \\
    --problem cvrp --problem_size 20 --num_test 100
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], label: str) -> float:
    """Run subprocess, return elapsed seconds. sys.exit if non-zero."""
    print(f"\n{'=' * 60}")
    print(f"  Run: {label}")
    print(f"  Cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    if ret.returncode != 0:
        print(f"  ❌ {label} 失败 (exit {ret.returncode})")
        sys.exit(ret.returncode)
    print(f"  ✓ {label} 完成, 耗时 {elapsed:.1f}s")
    return elapsed


def _find_latest_json(out_dir: Path, prefix: str) -> Path:
    cands = sorted(
        out_dir.glob(f"{prefix}*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"未在 {out_dir} 找到 {prefix}*.json")
    return cands[-1]


def _params_from_safetensors(model_path: str) -> int:
    """
    从 safetensors metadata 读总参数量, 不加载模型权重.
    bf16 占 2 bytes/param, total_bytes // 2 = param count.
    """
    mp = Path(model_path)
    index = mp / "model.safetensors.index.json"
    if index.exists():
        with open(index, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta["metadata"]["total_size"] // 2
    single = mp / "model.safetensors"
    if single.exists():
        # 单文件: safetensors header 8 bytes + json header + 数据
        # 简化: 用 file size 近似 (误差 < 0.01%, header 通常 <10KB)
        return single.stat().st_size // 2
    raise FileNotFoundError(f"{model_path} 既无 .index.json 也无 model.safetensors")


def _params_from_state_dict(pt_path: Path) -> int:
    import torch
    state = torch.load(pt_path, map_location="cpu")
    return sum(v.numel() for v in state.values())


def main():
    parser = argparse.ArgumentParser(description="HLR vs Baseline 对比 eval")

    parser.add_argument("--hlr_checkpoint", type=str, required=True,
                        help="HLR checkpoint 目录 (含 adapter_config.json + latent_reasoner.pt)")
    parser.add_argument("--hlr_base_model", type=str, default=None,
                        help="HLR 基座模型路径, 不传则从 adapter_config.json 自动读")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="baseline 已 merged 模型路径; 不传则自动 merge HLR LoRA "
                             "到 {hlr_checkpoint}/baseline_merged/ (缓存)")

    # 评估参数 (透传 evaluate.py)
    parser.add_argument("--problem", type=str, default="cvrp")
    parser.add_argument("--problem_size", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt_mode", type=str, default="think")
    parser.add_argument("--model_type", type=str, default="reasoning")
    parser.add_argument("--baseline_batch_size", type=int, default=4,
                        help="baseline (local) batch_size; HLR 强制 1")

    parser.add_argument("--out_dir", type=str, default=None,
                        help="对比输出目录, 默认 {hlr_checkpoint}/compare_eval/")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="跳过 baseline run (复用已有 JSON)")
    parser.add_argument("--skip_hlr", action="store_true",
                        help="跳过 HLR run (复用已有 JSON)")

    args = parser.parse_args()

    # ── 路径定位 ──
    repo_root = Path(__file__).resolve().parent.parent
    evaluate_py = repo_root / "UniCOP-Reason" / "evaluate.py"
    merge_py = repo_root / "UniCOP-Distill" / "stage1_solution" / "merge_adapter.py"
    hlr_ckpt = Path(args.hlr_checkpoint).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (hlr_ckpt / "compare_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (hlr_ckpt / "latent_reasoner.pt").exists():
        sys.exit(f"❌ {hlr_ckpt}/latent_reasoner.pt 不存在")

    # ── Step 1: baseline 模型 ──
    if args.baseline_model:
        baseline_path = Path(args.baseline_model).resolve()
        print(f"\n>>> Step 1: 用户指定 baseline = {baseline_path}")
    else:
        baseline_path = hlr_ckpt / "baseline_merged"
        if (baseline_path / "config.json").exists():
            print(f"\n>>> Step 1: baseline_merged 缓存命中: {baseline_path}")
        else:
            print(f"\n>>> Step 1: merge HLR LoRA → {baseline_path}")
            merge_cmd = [
                sys.executable, str(merge_py),
                "--adapter_path", str(hlr_ckpt),
                "--output_path", str(baseline_path),
            ]
            if args.hlr_base_model:
                merge_cmd += ["--base_model", args.hlr_base_model]
            _run(merge_cmd, "merge LoRA")

    # ── Step 2: baseline eval ──
    if not args.skip_baseline:
        baseline_cmd = [
            sys.executable, str(evaluate_py),
            "--backend", "local",
            "--model_path", str(baseline_path),
            "--problem", args.problem,
            "--problem_size", str(args.problem_size),
            "--num_test", str(args.num_test),
            "--num_samples", "1",
            "--temperature", str(args.temperature),
            "--max_completion_length", str(args.max_completion_length),
            "--batch_size", str(args.baseline_batch_size),
            "--prompt_mode", args.prompt_mode,
            "--model_type", args.model_type,
            "--save_dir", str(out_dir),
        ]
        baseline_wall = _run(baseline_cmd, "baseline (纯显式) eval")
    else:
        baseline_wall = None
        print(f"\n>>> Step 2: SKIP baseline run")

    # ── Step 3: HLR eval ──
    if not args.skip_hlr:
        hlr_cmd = [
            sys.executable, str(evaluate_py),
            "--backend", "hlr",
            "--hlr_checkpoint", str(hlr_ckpt),
            "--problem", args.problem,
            "--problem_size", str(args.problem_size),
            "--num_test", str(args.num_test),
            "--num_samples", "1",
            "--temperature", str(args.temperature),
            "--max_completion_length", str(args.max_completion_length),
            "--batch_size", "1",
            "--prompt_mode", args.prompt_mode,
            "--model_type", args.model_type,
            "--save_dir", str(out_dir),
        ]
        if args.hlr_base_model:
            hlr_cmd += ["--hlr_base_model", args.hlr_base_model]
        hlr_wall = _run(hlr_cmd, "HLR (latent 模式) eval")
    else:
        hlr_wall = None
        print(f"\n>>> Step 3: SKIP HLR run")

    # ── Step 4: 读 JSON + 对比 ──
    baseline_prefix = os.path.basename(str(baseline_path).rstrip("/\\"))
    hlr_prefix = "hlr_" + os.path.basename(str(hlr_ckpt).rstrip("/\\"))
    baseline_json = _find_latest_json(out_dir, baseline_prefix)
    hlr_json = _find_latest_json(out_dir, hlr_prefix)
    print(f"\n  baseline JSON: {baseline_json.name}")
    print(f"  HLR JSON:      {hlr_json.name}")

    with open(baseline_json, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(hlr_json, "r", encoding="utf-8") as f:
        hlr = json.load(f)

    # 参数量
    main_p = _params_from_safetensors(str(baseline_path))
    lr_p = _params_from_state_dict(hlr_ckpt / "latent_reasoner.pt")
    print(f"  main params: {main_p / 1e9:.3f}B  |  LR params: {lr_p / 1e6:.2f}M")

    # 单 combo (本对比脚本只跑一个 problem / size)
    b_res = baseline["results"][0]
    h_res = hlr["results"][0]
    h_sum = h_res.get("hlr_summary", {})

    n_test = b_res["num_test"]
    b_tok = b_res["avg_completion_tokens"]
    h_exp = h_sum.get("avg_explicit_tokens", 0)
    h_lat = h_sum.get("avg_latent_steps", 0)
    h_equiv = h_res["avg_completion_tokens"]
    h_segs = h_sum.get("avg_latent_segments", 0)

    # FLOPs per sample (decoder-only 估算 2 × params × tokens)
    b_flops = 2 * main_p * b_tok
    h_flops = 2 * main_p * h_exp + 2 * lr_p * h_lat
    flops_save = (1 - h_flops / b_flops) if b_flops else 0.0

    # 显式 token 节省: HLR 显式 token vs baseline 显式 token
    tok_save = (1 - h_exp / b_tok) if b_tok else 0.0

    h_wall = h_sum.get("avg_wall_time_sec", 0.0)

    compare = {
        "baseline_json": baseline_json.name,
        "hlr_json": hlr_json.name,
        "n_test": n_test,
        "params": {
            "main_params": main_p,
            "lr_params":   lr_p,
        },
        "task_metrics": {
            "format_match_rate": {
                "baseline": b_res["format_match_rate"],
                "hlr":      h_res["format_match_rate"],
                "delta":    round(h_res["format_match_rate"] - b_res["format_match_rate"], 4),
            },
            "global_feasibility_rate": {
                "baseline": b_res["global_feasibility_rate"],
                "hlr":      h_res["global_feasibility_rate"],
                "delta":    round(h_res["global_feasibility_rate"] - b_res["global_feasibility_rate"], 4),
            },
            "instance_feasibility_rate": {
                "baseline": b_res["instance_feasibility_rate"],
                "hlr":      h_res["instance_feasibility_rate"],
                "delta":    round(h_res["instance_feasibility_rate"] - b_res["instance_feasibility_rate"], 4),
            },
            "avg_best_dist": {
                "baseline": b_res["avg_best_dist"],
                "hlr":      h_res["avg_best_dist"],
            },
            "truncation_rate": {
                "baseline": b_res["truncation_rate"],
                "hlr":      h_res["truncation_rate"],
            },
        },
        "compute_savings": {
            "avg_baseline_tokens":        b_tok,
            "avg_hlr_explicit_tokens":    h_exp,
            "avg_hlr_latent_steps":       h_lat,
            "avg_hlr_latent_segments":    h_segs,
            "avg_hlr_equivalent_tokens":  h_equiv,
            "explicit_token_savings_pct": round(100 * tok_save, 2),
            "flops_savings_pct":          round(100 * flops_save, 2),
            "avg_hlr_per_sample_wall_sec": h_wall,
            "baseline_total_wall_sec":    round(baseline_wall, 1) if baseline_wall else None,
            "hlr_total_wall_sec":         round(hlr_wall, 1) if hlr_wall else None,
        },
    }

    # ── 控制台对比表 ──
    print("\n" + "=" * 72)
    print("  HLR vs Baseline (同模型, 推理模式对比)")
    print("=" * 72)
    print(f"  样本数: {n_test}  |  task: {args.problem}-{args.problem_size}\n")

    print(f"  ── 任务指标 ──  (Δ = HLR − baseline)")
    tm = compare["task_metrics"]
    for k in ("format_match_rate", "global_feasibility_rate",
              "instance_feasibility_rate", "truncation_rate"):
        d = tm[k]
        delta_str = f"  Δ={d['delta']:+.4f}" if "delta" in d else ""
        print(f"    {k:30s}  baseline={d['baseline']:.4f}  hlr={d['hlr']:.4f}{delta_str}")
    d = tm["avg_best_dist"]
    b_dist = f"{d['baseline']:.4f}" if d['baseline'] is not None else "N/A"
    h_dist = f"{d['hlr']:.4f}" if d['hlr'] is not None else "N/A"
    print(f"    {'avg_best_dist':30s}  baseline={b_dist}  hlr={h_dist}")

    print(f"\n  ── 计算量节省 ──")
    cs = compare["compute_savings"]
    print(f"    baseline 平均 token (纯显式) : {cs['avg_baseline_tokens']:.1f}")
    print(f"    HLR explicit tokens          : {cs['avg_hlr_explicit_tokens']:.1f}")
    print(f"    HLR latent steps             : {cs['avg_hlr_latent_steps']:.1f} "
          f"(段数 {cs['avg_hlr_latent_segments']:.2f})")
    print(f"    HLR equivalent tokens        : {cs['avg_hlr_equivalent_tokens']:.1f}")
    print(f"    ★ 显式 token 节省 (vs base)  : {cs['explicit_token_savings_pct']:6.2f}%")
    print(f"    ★ FLOPs 节省 (2·params·tok)  : {cs['flops_savings_pct']:6.2f}%")
    print(f"    HLR per-sample wall          : {cs['avg_hlr_per_sample_wall_sec']:.3f}s")
    if cs['baseline_total_wall_sec'] and cs['hlr_total_wall_sec']:
        print(f"    全程 wall-clock              : baseline={cs['baseline_total_wall_sec']:.0f}s, "
              f"hlr={cs['hlr_total_wall_sec']:.0f}s (含模型加载)")
    print("=" * 72)

    cmp_path = out_dir / "compare.json"
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2, ensure_ascii=False)
    print(f"\n  对比 JSON: {cmp_path}\n")


if __name__ == "__main__":
    main()
