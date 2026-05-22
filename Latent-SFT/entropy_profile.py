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

        item = {**r, "latent_segments": segments}
        if save_entropies:
            item["cot_entropies"] = [round(e, 4) for e in cot_entropies]
        results.append(item)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_segs = sum(len(r["latent_segments"]) for r in results)
    with_latent = sum(1 for r in results if r["latent_segments"])
    print(f"  [auto-profile] {len(results)} 有效 / {skipped} 跳过")
    print(f"  含 latent 段: {with_latent}/{len(results)}, 总段数: {total_segs}")
    print(f"  写入: {output_path}")

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

        item = {**r, "latent_segments": segments}
        if args.save_entropies:
            item["cot_entropies"] = [round(e, 4) for e in cot_entropies]
        results.append(item)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_segs = sum(len(r["latent_segments"]) for r in results)
    with_latent = sum(1 for r in results if r["latent_segments"])
    print(f"\nProfiling 完成: {len(results)} 有效, {skipped} 跳过")
    print(f"  含 latent 段样本: {with_latent}/{len(results)}")
    print(f"  总 latent 段数: {total_segs}")
    print(f"  输出: {args.output}")

    return len(results), skipped


def main():
    parser = argparse.ArgumentParser(
        description="Entropy profiling for HLR training "
                    "(标注低熵确定性段, 训练侧规则与 inference.py 镜像一致)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--save_entropies", action="store_true")

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
