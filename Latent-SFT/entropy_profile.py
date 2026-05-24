"""
Entropy profiling：对 CoT 逐 token 算熵，自动标记 latent/explicit 段。

预处理工具，训练前运行一次。输出 profiled jsonl 供 train.py 读取。

用法 (cwd = UniCOP 根目录):
    python Latent-SFT/entropy_profile.py \
        --model "$BASE_MODEL" \
        --data UniCOP-Distill/data/chains_template_cvrp20.jsonl \
        --output Latent-SFT/data/profiled_cvrp20.jsonl
"""

import argparse
import json
import math
import traceback
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCKEY = "SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
_POSTHOC_SYSTEM_MARKER = "\n\nYour output MUST start with <think>"
_POSTHOC_USER_MARKER = "\n\nTarget solution ("


def _notify(title, content=""):
    try:
        url = f"https://sctapi.ftqq.com/{_SCKEY}.send?title={title}&desp={content}"
        urlopen(Request(url), timeout=10)
    except Exception:
        pass


def _strip_posthoc(text, marker):
    idx = text.find(marker)
    return text[:idx] if idx != -1 else text


def summarize_latent_coverage(records, compression_ratio: int) -> dict:
    """
    遍历 profiled records 算 token-level 压缩比例报送.

    依赖每条 record 含 `cot_token_count` (显式 CoT 总 token 数)
    + `latent_segments` (list[{"start","end"}], inclusive).

    返回字典:
      - n_total, n_with_seg
      - cot_total_tokens, latent_covered_tokens, latent_steps
      - covered_ratio    = 显式 token 被 latent 替代的比例
      - savings_ratio    = 压缩后 CoT 等效长度减少的比例 (考虑 compression_ratio)
      - avg_seg_per_sample, avg_seg_len
    """
    n_total = len(records)
    n_with_seg = 0
    total_cot = 0
    total_latent_covered = 0
    total_latent_steps = 0
    seg_counts = []
    seg_lens = []

    for r in records:
        cot_n = r.get("cot_token_count", 0)
        if cot_n <= 0:
            continue
        total_cot += cot_n
        segs = r.get("latent_segments", [])
        if segs:
            n_with_seg += 1
        seg_counts.append(len(segs))
        for s in segs:
            seg_len = s["end"] - s["start"] + 1
            seg_lens.append(seg_len)
            total_latent_covered += seg_len
            total_latent_steps += math.ceil(seg_len / compression_ratio)

    if total_cot == 0:
        return {
            "n_total": n_total, "n_with_seg": 0,
            "cot_total_tokens": 0, "latent_covered_tokens": 0, "latent_steps": 0,
            "covered_ratio": 0.0, "savings_ratio": 0.0,
            "avg_seg_per_sample": 0.0, "avg_seg_len": 0.0,
        }

    covered_ratio = total_latent_covered / total_cot
    equivalent_len = total_cot - total_latent_covered + total_latent_steps
    savings_ratio = 1.0 - equivalent_len / total_cot

    return {
        "n_total": n_total,
        "n_with_seg": n_with_seg,
        "cot_total_tokens": total_cot,
        "latent_covered_tokens": total_latent_covered,
        "latent_steps": total_latent_steps,
        "covered_ratio": covered_ratio,
        "savings_ratio": savings_ratio,
        "avg_seg_per_sample": sum(seg_counts) / max(n_total, 1),
        "avg_seg_len": sum(seg_lens) / max(len(seg_lens), 1) if seg_lens else 0.0,
    }


def print_coverage_summary(stats: dict, title: str = "Latent 覆盖统计", prefix: str = "  "):
    n = max(stats["n_total"], 1)
    print(f"{prefix}━━━ {title} ━━━")
    print(f"{prefix}样本数             : {stats['n_total']}")
    print(f"{prefix}含 latent 段       : {stats['n_with_seg']}/{stats['n_total']} "
          f"({100 * stats['n_with_seg'] / n:.1f}%)")
    print(f"{prefix}平均段数/样本      : {stats['avg_seg_per_sample']:.2f}")
    print(f"{prefix}平均段长 (token)   : {stats['avg_seg_len']:.1f}")
    print(f"{prefix}显式 CoT 总 token  : {stats['cot_total_tokens']}")
    print(f"{prefix}被 latent 覆盖     : {stats['latent_covered_tokens']} tokens")
    print(f"{prefix}压缩为 latent step : {stats['latent_steps']} "
          f"(compression_ratio 已应用)")
    print(f"{prefix}{'─' * 56}")
    print(f"{prefix}★ 显式 token 替代率: {100 * stats['covered_ratio']:6.2f}%   "
          f"(显式 CoT 被 latent 接管的比例)")
    print(f"{prefix}★ CoT 算力节省比例 : {100 * stats['savings_ratio']:6.2f}%   "
          f"(CoT forward 等效长度减少多少)")
    print(f"{prefix}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


@torch.no_grad()
def compute_entropies(model, input_ids):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]
    entropies = []
    for start in range(0, logits.size(0), 512):
        chunk = logits[start : start + 512].float()
        log_p = F.log_softmax(chunk, dim=-1)
        ent = -(log_p.exp() * log_p).sum(dim=-1)
        entropies.extend(ent.cpu().tolist())
    return entropies


def detect_latent_segments(
    entropies,
    window,
    min_segment,
    max_segment,
    quantile=0.5,
    cooldown=24,
):
    """
    检测可压缩的低熵段 (确定性 / 机械重复). 训练侧与推理侧规则镜像一致.

    进入 latent (低熵段):
      1. 连续 window 步熵下降
      2. 当前熵 < quantile 百分位 (默认 50 分位数 = 中位数)
      3. 距上次段结束至少 cooldown 个 token (防抖)
    退出 latent:
      1. 连续 window 步熵上升  OR
      2. 段长 >= max_segment (强制切断)
    段长检查:
      段长 >= min_segment 才保留 (太短的段没有压缩价值)

    Args:
        entropies: list[float], CoT 区间逐 token 熵
        window:    趋势窗口 (默认 3)
        min_segment: 段长下限 (token 数), 一般 = min_latent_steps × compression_ratio
        max_segment: 段长上限, 一般 = max_latent_steps × compression_ratio
        quantile:  绝对熵阈值的分位数 (0..1)
        cooldown:  段间最小间隔 (token 数)

    Returns:
        list[{"start": int, "end": int}]  inclusive 区间
    """
    n = len(entropies)
    if n == 0:
        return []

    # 该 CoT 自己的熵分位数作为绝对阈值
    sorted_e = sorted(entropies)
    q_idx = min(int(len(sorted_e) * quantile), len(sorted_e) - 1)
    threshold = sorted_e[q_idx]

    segments = []
    in_latent = False
    entry = None
    last_exit = -1

    for i in range(window, n):
        if not in_latent:
            falling = all(
                entropies[i - window + j] > entropies[i - window + j + 1]
                for j in range(window)
            )
            below_threshold = entropies[i] < threshold
            cooldown_ok = (last_exit < 0) or (i - last_exit) >= cooldown

            if falling and below_threshold and cooldown_ok:
                candidate = i - window
                if candidate > last_exit:
                    entry = candidate
                    in_latent = True
        else:
            rising = all(
                entropies[i - window + j] < entropies[i - window + j + 1]
                for j in range(window)
            )
            force_exit = (i - entry + 1) >= max_segment

            if rising or force_exit:
                seg_len = i - entry + 1
                if seg_len >= min_segment:
                    segments.append({"start": entry, "end": i})
                    last_exit = i
                in_latent = False
                entry = None

    # 收尾: 末尾未退出的段
    if in_latent and entry is not None:
        seg_len = n - entry
        if seg_len >= min_segment:
            end = min(entry + max_segment - 1, n - 1)
            segments.append({"start": entry, "end": end})

    return segments


@torch.no_grad()
def auto_profile_inplace(
    model,
    tokenizer,
    raw_data_path: str,
    output_path: str,
    entropy_window: int = 3,
    entropy_quantile: float = 0.5,
    min_segment: int = 12,
    max_segment: int = 32,
    cooldown: int = 24,
    max_length: int = 8192,
    save_entropies: bool = False,
    filter_problems=None,
    filter_sizes=None,
):
    """
    train_hlr 内联版: 用已加载的 model + tokenizer 跑 entropy profile, 不重新加载.

    用于 cfg.auto_rebuild_data=True 时, train_hlr 启动时把原始 chains jsonl 转成 profiled jsonl.
    复用主模型避免 subprocess 二次加载 ~15B 模型.

    输出格式跟 profile_dataset 一致: 原 record 附加 latent_segments 字段写回 output_path.
    """
    import os
    model.eval()

    records = []
    with open(raw_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if filter_problems and r.get("problem_type") not in filter_problems:
                continue
            if filter_sizes and r.get("n") not in filter_sizes:
                continue
            records.append(r)

    results = []
    skipped = 0

    for r in tqdm(records, desc="auto entropy profile"):
        output = r.get("output", "")
        if not output or not output.strip():
            skipped += 1
            continue

        orig_system = _strip_posthoc(r["prompt"]["system"], _POSTHOC_SYSTEM_MARKER)
        orig_user = _strip_posthoc(r["prompt"]["user"], _POSTHOC_USER_MARKER)

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": orig_system},
             {"role": "user", "content": orig_user}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt_text.rstrip().endswith("<think>"):
            prompt_text += "<think>\n"

        output_stripped = output.lstrip()
        if output_stripped.startswith("<think>"):
            output_stripped = output_stripped[len("<think>"):].lstrip("\n")
        think_end = output_stripped.find("</think>")
        if think_end == -1:
            skipped += 1
            continue
        cot_text = output_stripped[:think_end]
        if not output_stripped[think_end + len("</think>"):].strip():
            skipped += 1
            continue

        teacher_text = prompt_text + output_stripped + tokenizer.eos_token
        teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)
        if len(teacher_ids) > max_length:
            skipped += 1
            continue

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
        if len(cot_ids) < min_segment:
            results.append({**r, "latent_segments": []})
            continue

        cot_start = len(prompt_ids)
        input_tensor = torch.tensor([teacher_ids], device=model.device)
        entropies = compute_entropies(model, input_tensor)
        cot_entropies = entropies[cot_start - 1 : cot_start + len(cot_ids) - 1]

        segments = detect_latent_segments(
            cot_entropies,
            window=entropy_window,
            min_segment=min_segment,
            max_segment=max_segment,
            quantile=entropy_quantile,
            cooldown=cooldown,
        )

        item = {**r, "cot_token_count": len(cot_ids), "latent_segments": segments}
        if save_entropies:
            item["cot_entropies"] = [round(e, 4) for e in cot_entropies]
        results.append(item)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  [auto-profile] {len(results)} 有效 / {skipped} 跳过")
    print(f"  写入: {output_path}")
    # 压缩比例报送 (token-level)
    stats = summarize_latent_coverage(results, compression_ratio=4)
    print_coverage_summary(stats, title="Latent 覆盖统计 (auto_profile_inplace)")

    return len(results), skipped


def profile_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    records = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if getattr(args, "limit", 0) > 0:
        print(f"[limit] 只处理前 {args.limit} 条 (smoke 模式)")
        records = records[: args.limit]

    # ── 分片支持 (4 卡并行用): 各 rank 只处理自己那一份 ──
    if getattr(args, "num_shards", 1) > 1:
        n_total = len(records)
        per = (n_total + args.num_shards - 1) // args.num_shards
        s = args.shard_rank * per
        e = min(s + per, n_total)
        records = records[s:e]
        print(f"[shard] rank {args.shard_rank}/{args.num_shards}: "
              f"处理 {len(records)} 条 [{s}, {e}) of {n_total}")

    results = []
    skipped = 0

    for r in tqdm(records, desc="Entropy profiling"):
        output = r.get("output", "")
        if not output or not output.strip():
            skipped += 1
            continue

        orig_system = _strip_posthoc(r["prompt"]["system"], _POSTHOC_SYSTEM_MARKER)
        orig_user = _strip_posthoc(r["prompt"]["user"], _POSTHOC_USER_MARKER)

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": orig_system},
             {"role": "user", "content": orig_user}],
            tokenize=False, add_generation_prompt=True,
        )
        if not prompt_text.rstrip().endswith("<think>"):
            prompt_text += "<think>\n"

        output_stripped = output.lstrip()
        if output_stripped.startswith("<think>"):
            output_stripped = output_stripped[len("<think>"):].lstrip("\n")

        think_end = output_stripped.find("</think>")
        if think_end == -1:
            skipped += 1
            continue

        cot_text = output_stripped[:think_end]
        solution_text = output_stripped[think_end + len("</think>"):].lstrip("\n")
        if not solution_text.strip():
            skipped += 1
            continue

        # Distill Stage 2 风原样拼接: prompt + output_stripped + eos
        # output_stripped 已剥重复 <think>, 仍含 </think> + 原始换行 + solution
        teacher_text = prompt_text + output_stripped + tokenizer.eos_token
        teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)

        if len(teacher_ids) > args.max_length:
            skipped += 1
            continue

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)

        if len(cot_ids) < args.min_segment:
            results.append({**r, "latent_segments": []})
            continue

        cot_start = len(prompt_ids)
        input_tensor = torch.tensor([teacher_ids], device=model.device)
        entropies = compute_entropies(model, input_tensor)
        # entropies[i] = 预测 token i+1 的熵，偏移 -1 使 cot_entropies[k] 对应 cot_ids[k]
        cot_entropies = entropies[cot_start - 1 : cot_start + len(cot_ids) - 1]

        segments = detect_latent_segments(
            cot_entropies,
            window=args.entropy_window,
            min_segment=args.min_segment,
            max_segment=args.max_segment,
            quantile=args.entropy_quantile,
            cooldown=args.cooldown,
        )

        item = {**r, "cot_token_count": len(cot_ids), "latent_segments": segments}
        if args.save_entropies:
            item["cot_entropies"] = [round(e, 4) for e in cot_entropies]
        results.append(item)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nProfiling 完成: {len(results)} 有效, {skipped} 跳过")
    print(f"  输出: {args.output}")
    # token-level 压缩比例报送
    stats = summarize_latent_coverage(results, compression_ratio=4)
    print_coverage_summary(stats, title="Latent 覆盖统计 (entropy_profile)")
    return len(results), skipped


def main():
    parser = argparse.ArgumentParser(
        description="Entropy profiling for HLR training "
                    "(标注低熵确定性段, 训练侧规则与未来推理侧需镜像一致)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--save_entropies", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条样本 (0 = 全量); smoke 模式用 200 条快速验证")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="数据分片总数 (>1 时并行多 GPU 用, 配合 --shard_rank)")
    parser.add_argument("--shard_rank", type=int, default=0,
                        help="本进程负责哪个 shard (0..num_shards-1); 各 rank 独立 output 文件由调用方拼名")

    # Latent 进出 trigger (与 HLRConfig 默认值对齐)
    parser.add_argument("--entropy_window", type=int, default=3,
                        help="趋势窗口")
    parser.add_argument("--entropy_quantile", type=float, default=0.5,
                        help="分位数阈值 (默认 0.5 = 中位数), 当前熵 < 该分位数才算低熵")
    parser.add_argument("--min_segment", type=int, default=12,
                        help="段长下限 (token 数) = min_latent_steps × compression_ratio (默认 3×4=12)")
    parser.add_argument("--max_segment", type=int, default=32,
                        help="段长上限 (token 数) = max_latent_steps × compression_ratio (默认 8×4=32)")
    parser.add_argument("--cooldown", type=int, default=24,
                        help="段间最小间隔 (显式 token 数, 默认 24 ≈ 6 latent step)")

    args = parser.parse_args()

    try:
        n_ok, n_skip = profile_dataset(args)
        _notify("Entropy profiling 完成", f"有效 {n_ok}，跳过 {n_skip}")
    except Exception:
        _notify("Entropy profiling 失败", traceback.format_exc()[:500])
        raise


if __name__ == "__main__":
    main()
