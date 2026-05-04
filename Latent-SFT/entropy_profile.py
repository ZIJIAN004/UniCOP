"""
Entropy profiling：对 CoT 逐 token 算熵，自动标记 latent/explicit 段。

预处理工具，训练前运行一次。输出 profiled jsonl 供 train.py 读取。

用法:
    python entropy_profile.py \
        --model ./output_grpo/final_model \
        --data ../UniCOP-Distill/data/chains_self_cvrp20.jsonl \
        --output ./data/profiled_cvrp20.jsonl
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
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return ent[0].cpu().tolist()


def detect_latent_segments(entropies, window, min_segment):
    """
    基于熵趋势检测 latent 段。

    连续 window 步上升 → 进入 latent（entry = 上升起点）
    连续 window 步下降 → 退出 latent（exit = 下降终点）
    """
    n = len(entropies)
    segments = []
    in_latent = False
    entry = None

    for i in range(window, n):
        if not in_latent:
            rising = all(
                entropies[i - window + j] < entropies[i - window + j + 1]
                for j in range(window)
            )
            if rising:
                entry = i - window
                in_latent = True
        else:
            falling = all(
                entropies[i - window + j] > entropies[i - window + j + 1]
                for j in range(window)
            )
            if falling:
                if i - entry >= min_segment:
                    segments.append({"start": entry, "end": i})
                in_latent = False
                entry = None

    if in_latent and entry is not None:
        if n - 1 - entry >= min_segment:
            segments.append({"start": entry, "end": n - 1})

    return segments


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

        teacher_text = (
            prompt_text + cot_text + "</think>\n"
            + solution_text + tokenizer.eos_token
        )
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
        cot_entropies = entropies[cot_start : cot_start + len(cot_ids)]

        segments = detect_latent_segments(
            cot_entropies, args.entropy_window, args.min_segment
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
        description="Entropy profiling for mixed latent/explicit training"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--entropy_window", type=int, default=3)
    parser.add_argument("--min_segment", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--save_entropies", action="store_true")
    args = parser.parse_args()

    try:
        n_ok, n_skip = profile_dataset(args)
        _notify("Entropy profiling 完成", f"有效 {n_ok}，跳过 {n_skip}")
    except Exception:
        _notify("Entropy profiling 失败", traceback.format_exc()[:500])
        raise


if __name__ == "__main__":
    main()
