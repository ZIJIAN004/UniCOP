"""诊断 Stage 2 SFT 退化原因。

在服务器上运行:
    cd /home/ntu/lzj/UniCOP/UniCOP-Distill
    python diagnose_stage2.py

检查项:
    1. Stage 1 / Stage 2 模型目录完整性（是完整模型还是残留 adapter）
    2. Stage 2 adapter_config.json 中 base_model 是否指向 Stage 1
    3. chains 数据经 max_length=4096 过滤后，实际参与训练的样本数
    4. Stage 1 vs Stage 2 权重差异（确认 LoRA 非空合并）
"""

import json
import os
import sys

# ── 路径配置 ──────────────────────────────────────────────────────────
DISTILL_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_MODEL = os.path.join(DISTILL_DIR, "output_sft_stage1_cvrp20", "final_model")
STAGE2_MODEL = os.path.join(DISTILL_DIR, "output_sft_stage2_cvrp20", "final_model")
CHAINS_FILE  = os.path.join(DISTILL_DIR, "data", "chains_v3_clean.jsonl")

PROBLEM = "cvrp"
SIZE = 20
MAX_LENGTH = 4096        # pipeline 实际传入值
MAX_OUTPUT_LENGTH = 4096  # train_sft_stage2.py 默认值


def check_model_dir(path, label):
    print(f"\n{'='*60}")
    print(f"  检查 {label}: {path}")
    print(f"{'='*60}")

    if not os.path.isdir(path):
        print(f"  ✗ 目录不存在!")
        return False

    files = os.listdir(path)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_bin = any(f.endswith(".bin") and "adapter" not in f for f in files)
    has_adapter_config = "adapter_config.json" in files
    has_adapter_model = any(f.startswith("adapter_model") for f in files)
    has_config = "config.json" in files
    has_tokenizer = "tokenizer.json" in files or "tokenizer_config.json" in files

    print(f"  config.json:        {'✓' if has_config else '✗'}")
    print(f"  tokenizer:          {'✓' if has_tokenizer else '✗'}")
    print(f"  safetensors/bin:    {'✓' if has_safetensors or has_bin else '✗'}")
    print(f"  adapter_config:     {'✓ (残留!)' if has_adapter_config else '✗ (正常)'}")
    print(f"  adapter_model:      {'✓ (残留!)' if has_adapter_model else '✗ (正常)'}")

    if has_adapter_config and not has_safetensors and not has_bin:
        print(f"  ⚠️ 只有 adapter 文件，没有完整模型权重 — merge 可能失败了!")
        return False

    if has_adapter_config:
        adapter_cfg_path = os.path.join(path, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base = adapter_cfg.get("base_model_name_or_path", "UNKNOWN")
        print(f"  adapter base_model: {base}")

    if has_safetensors:
        total_mb = sum(
            os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
            for f in files if f.endswith(".safetensors")
        )
        print(f"  模型文件总大小:    {total_mb:.0f} MB")
        if total_mb < 100:
            print(f"  ⚠️ 模型文件过小 (<100MB)，可能是空权重!")
            return False

    return True


def check_data_filtering():
    print(f"\n{'='*60}")
    print(f"  检查训练数据过滤 (max_length={MAX_LENGTH})")
    print(f"{'='*60}")

    if not os.path.isfile(CHAINS_FILE):
        print(f"  ✗ 数据文件不存在: {CHAINS_FILE}")
        return

    from transformers import AutoTokenizer

    tokenizer_path = STAGE1_MODEL if os.path.isdir(STAGE1_MODEL) else None
    if tokenizer_path is None:
        print("  ✗ Stage 1 模型目录不存在，无法加载 tokenizer")
        return

    print(f"  加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    total = 0
    matched = 0
    survived_output = 0
    survived_total = 0
    output_lens = []
    total_lens = []

    _POSTHOC_SYSTEM_MARKER = "\n\nYour output MUST start with <think>"
    _POSTHOC_USER_MARKER   = "\n\nTarget solution ("

    with open(CHAINS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1

            if r.get("problem_type") != PROBLEM or r.get("n") != SIZE:
                continue
            matched += 1

            system = r["prompt"]["system"]
            user = r["prompt"]["user"]
            output = r["output"]

            idx_s = system.find(_POSTHOC_SYSTEM_MARKER)
            if idx_s != -1:
                system = system[:idx_s]
            idx_u = user.find(_POSTHOC_USER_MARKER)
            if idx_u != -1:
                user = user[:idx_u]

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True,
            )

            output_stripped = output.lstrip()
            if output_stripped.startswith("<think>"):
                output_stripped = output_stripped[len("<think>"):].lstrip("\n")
            completion_text = output_stripped + tokenizer.eos_token

            prompt_tokens = len(tokenizer.encode(prompt_text))
            completion_tokens = len(tokenizer.encode(completion_text))
            total_tokens = prompt_tokens + completion_tokens

            output_lens.append(completion_tokens)
            total_lens.append(total_tokens)

            if completion_tokens <= MAX_OUTPUT_LENGTH:
                survived_output += 1
            if total_tokens <= MAX_LENGTH:
                survived_total += 1

    print(f"\n  chains 文件总行数:              {total}")
    print(f"  {PROBLEM.upper()} n={SIZE} 匹配:         {matched}")
    print(f"  通过 output ≤ {MAX_OUTPUT_LENGTH} 过滤:   {survived_output}")
    print(f"  通过 total ≤ {MAX_LENGTH} 过滤:    {survived_total}")
    print(f"  ⭐ 实际参与训练的样本数:          {survived_total}")

    if output_lens:
        output_lens.sort()
        total_lens.sort()
        print(f"\n  completion token 分布:")
        print(f"    min={output_lens[0]}, median={output_lens[len(output_lens)//2]}, "
              f"max={output_lens[-1]}, mean={sum(output_lens)/len(output_lens):.0f}")
        print(f"  prompt+completion token 分布:")
        print(f"    min={total_lens[0]}, median={total_lens[len(total_lens)//2]}, "
              f"max={total_lens[-1]}, mean={sum(total_lens)/len(total_lens):.0f}")

    if survived_total < matched:
        pct = (1 - survived_total / matched) * 100 if matched > 0 else 0
        print(f"\n  ⚠️ max_length=4096 过滤掉了 {pct:.0f}% 的样本!")
        print(f"     建议: pipeline 中 --max_length 改为 8192 (脚本默认值)")


def check_weight_diff():
    print(f"\n{'='*60}")
    print(f"  检查 Stage 1 vs Stage 2 权重差异")
    print(f"{'='*60}")

    if not os.path.isdir(STAGE1_MODEL) or not os.path.isdir(STAGE2_MODEL):
        print("  ✗ 模型目录不完整，跳过")
        return

    try:
        from safetensors import safe_open
    except ImportError:
        print("  ✗ 需要 safetensors 库")
        return

    s1_files = sorted(f for f in os.listdir(STAGE1_MODEL) if f.endswith(".safetensors"))
    s2_files = sorted(f for f in os.listdir(STAGE2_MODEL) if f.endswith(".safetensors"))

    if not s1_files or not s2_files:
        print("  ✗ 缺少 safetensors 文件")
        return

    import torch

    n_same = 0
    n_diff = 0
    max_diff = 0.0
    diff_layers = []

    with safe_open(os.path.join(STAGE1_MODEL, s1_files[0]), framework="pt") as f1, \
         safe_open(os.path.join(STAGE2_MODEL, s2_files[0]), framework="pt") as f2:

        keys1 = set(f1.keys())
        keys2 = set(f2.keys())
        common = keys1 & keys2

        for key in sorted(common)[:50]:
            t1 = f1.get_tensor(key)
            t2 = f2.get_tensor(key)
            if t1.shape != t2.shape:
                diff_layers.append(f"{key} (shape不同!)")
                n_diff += 1
                continue
            d = (t1.float() - t2.float()).abs().max().item()
            if d < 1e-6:
                n_same += 1
            else:
                n_diff += 1
                max_diff = max(max_diff, d)
                if len(diff_layers) < 5:
                    diff_layers.append(f"{key} (max_diff={d:.6f})")

    print(f"  比较了前 50 个 tensor key:")
    print(f"    相同: {n_same}, 不同: {n_diff}, 最大差异: {max_diff:.6f}")
    if diff_layers:
        print(f"    差异示例:")
        for dl in diff_layers:
            print(f"      {dl}")

    if n_diff == 0:
        print(f"  ⚠️ Stage 1 和 Stage 2 权重完全相同 — LoRA 合并可能是空操作!")
    else:
        print(f"  ✓ 权重有差异，LoRA 合并生效")


def main():
    print("=" * 60)
    print("  Stage 2 SFT 退化诊断")
    print("=" * 60)

    ok1 = check_model_dir(STAGE1_MODEL, "Stage 1 模型")
    ok2 = check_model_dir(STAGE2_MODEL, "Stage 2 模型")

    check_data_filtering()

    if ok1 and ok2:
        check_weight_diff()

    print(f"\n{'='*60}")
    print(f"  诊断完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
