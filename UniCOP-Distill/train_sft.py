"""
SFT 训练脚本：用蒸馏数据（Gemini 推理链）微调 DeepSeek-R1-Distill 模型。

数据来源：generate_chains.py 生成的 chains.jsonl
训练目标：模型根据原始问题描述，直接生成 <think>...</think> 推理链 + 格式化答案

单卡运行：
    python train_sft.py
    python train_sft.py --model /path/to/model --data data/chains.jsonl

多卡运行：
    accelerate launch --num_processes 3 train_sft.py --zero_stage 2 --gradient_checkpointing
"""

import argparse
import json
import os

import torch

# 预留 95% 显存，防止其他进程挤占导致训练 OOM
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(0.95, i)

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


# ── 默认参数 ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "F:/HuaweiMoveData/Users/Carl/Desktop/代码/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
DEFAULT_DATA  = "data/chains.jsonl"

# generate_chains.py 中追加到 system prompt 的后验推理后缀（用于定位和剥离）
_POSTHOC_SYSTEM_MARKER = "\n\nYour output MUST start with <think>"
# 追加到 user prompt 中的 LKH 答案前缀
_POSTHOC_USER_MARKER   = "\n\nTarget solution ("


# ── DeepSpeed 配置 ──────────────────────────────────────────────────────────────

def make_deepspeed_config(zero_stage: int) -> dict | None:
    if zero_stage == 0:
        return None

    base = {
        "bf16":              {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size":               "auto",
        "gradient_accumulation_steps":    "auto",
        "steps_per_print":                50,
    }

    if zero_stage == 2:
        base["zero_optimization"] = {
            "stage":                2,
            "overlap_comm":         True,
            "contiguous_gradients": True,
            "reduce_bucket_size":   5e8,
            "reduce_scatter":       True,
        }
    elif zero_stage == 3:
        base["zero_optimization"] = {
            "stage":                          3,
            "overlap_comm":                   True,
            "contiguous_gradients":           True,
            "reduce_bucket_size":             "auto",
            "stage3_prefetch_bucket_size":    "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters":     1e9,
            "stage3_max_reuse_distance":      1e9,
            "gather_16bit_weights_on_model_save": True,
        }

    return base


# ── 数据加载与处理 ──────────────────────────────────────────────────────────────

def strip_posthoc_system(system: str) -> str:
    """从 post-hoc system prompt 中剥离后验推理指令，还原原始 system prompt。"""
    idx = system.find(_POSTHOC_SYSTEM_MARKER)
    if idx != -1:
        return system[:idx]
    return system


def strip_posthoc_user(user: str) -> str:
    """从 post-hoc user prompt 中剥离 LKH 答案，还原原始问题描述。"""
    idx = user.find(_POSTHOC_USER_MARKER)
    if idx != -1:
        return user[:idx]
    return user


def load_sft_dataset(data_path: str, tokenizer, max_length: int) -> Dataset:
    """
    从 chains.jsonl 加载数据，构建 SFT 训练集。

    每条样本：
      - 还原原始 prompt（剥离 post-hoc 部分）
      - 用 chat template 拼接 [system, user, assistant(output)]
      - assistant 内容 = Gemini 生成的 <think>...</think> + 答案
    """
    records = []
    skipped = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # 还原原始 prompt
            orig_system = strip_posthoc_system(r["prompt"]["system"])
            orig_user   = strip_posthoc_user(r["prompt"]["user"])
            output      = r["output"]

            if not output or not output.strip():
                skipped += 1
                continue

            # 构建 chat 消息
            messages = [
                {"role": "system", "content": orig_system},
                {"role": "user",   "content": orig_user},
                {"role": "assistant", "content": output},
            ]

            # 用 chat template 渲染为完整文本
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            records.append({
                "text": text,
                "problem_type": r.get("problem_type", "unknown"),
                "n": r.get("n", 0),
            })

    if skipped:
        print(f"  跳过 {skipped} 条无效记录")

    # 按问题类型统计
    from collections import Counter
    type_counts = Counter(r["problem_type"] for r in records)
    size_counts = Counter(r["n"] for r in records)
    print(f"  加载 {len(records)} 条样本")
    print(f"  问题类型分布: {dict(sorted(type_counts.items()))}")
    print(f"  规模分布:     {dict(sorted(size_counts.items()))}")

    # 统计超长样本
    num_over = 0
    for r in records:
        token_len = len(tokenizer.encode(r["text"]))
        if token_len > max_length:
            num_over += 1
    if num_over:
        print(f"  WARNING: {num_over}/{len(records)} 条样本超过 max_length={max_length}，训练时将被截断")

    return Dataset.from_dict({
        "text": [r["text"] for r in records],
    })


# ── 主函数 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT 微调：用蒸馏推理链训练推理模型")

    # 模型
    parser.add_argument("--model",        type=str, default=DEFAULT_MODEL,
                        help="基础模型路径")
    parser.add_argument("--no_lora",      action="store_true",
                        help="禁用 LoRA，全参数微调")
    parser.add_argument("--lora_rank",    type=int, default=16)
    parser.add_argument("--lora_alpha",   type=int, default=32)

    # 数据
    parser.add_argument("--data",         type=str, default=DEFAULT_DATA,
                        help="chains.jsonl 文件路径")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="最大序列长度（prompt + completion）")
    parser.add_argument("--val_ratio",    type=float, default=0.05,
                        help="验证集比例（默认 5%%）")

    # 训练
    parser.add_argument("--seed",         type=int,   default=42,
                        help="全局随机种子（可复现性）")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=2e-5,
                        help="学习率（SFT 通常比 RL 高一个数量级）")
    parser.add_argument("--batch_size",   type=int,   default=1,
                        help="每卡 batch size")
    parser.add_argument("--grad_accum",   type=int,   default=8,
                        help="梯度累积步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # 多卡
    parser.add_argument("--zero_stage",   type=int, default=0, choices=[0, 2, 3])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # 输出
    parser.add_argument("--output_dir",   type=str, default="./output_sft")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps",   type=int,  default=100)
    parser.add_argument("--use_wandb",    action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    print(f"{'='*60}")
    print(f"  SFT 蒸馏训练")
    print(f"  模型:       {args.model}")
    print(f"  数据:       {args.data}")
    print(f"  LoRA:       {'关闭' if args.no_lora else f'rank={args.lora_rank}'}")
    print(f"  最大序列长: {args.max_length}")
    print(f"  ZeRO:       {args.zero_stage}")
    print(f"  输出:       {args.output_dir}")
    print(f"{'='*60}\n")

    # ── 加载 tokenizer ───────────────────────────────────────────────────
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 加载数据 ─────────────────────────────────────────────────────────
    print("加载训练数据...")
    dataset = load_sft_dataset(args.data, tokenizer, args.max_length)

    # 划分训练集和验证集
    if args.val_ratio > 0 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=args.val_ratio, seed=42)
        train_dataset = split["train"]
        eval_dataset  = split["test"]
        print(f"训练集: {len(train_dataset)}  验证集: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset  = None
        print(f"训练集: {len(train_dataset)}  （无验证集）")

    # ── 加载模型 ─────────────────────────────────────────────────────────
    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA 配置
    peft_config = None
    if not args.no_lora:
        print(f"启用 LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # ── SFT 配置 ─────────────────────────────────────────────────────────
    ds_config = make_deepspeed_config(args.zero_stage)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
        deepspeed=ds_config,
        gradient_checkpointing=args.gradient_checkpointing,
        dataset_text_field="text",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
    )

    # ── 训练 ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n开始 SFT 训练...")
    trainer.train()

    # ── 保存 ─────────────────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存到: {save_path}")

    # ── 生成推理样例（仅主进程执行，避免多 rank 冲突） ──────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        if args.zero_stage == 3:
            print("\nZeRO-3 模式：从保存的 checkpoint 重新加载模型以生成样例...")
            del model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                save_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to("cuda")
        _save_examples(model, tokenizer, args)


def _save_examples(model, tokenizer, args, num_examples=3):
    """
    用训练好的模型对几个样本问题生成推理样例，保存到 examples.json。
    直接从 chains.jsonl 取前几个样本的原始 prompt 进行推理。
    统计输出长度分布和自然结束率。
    """
    print("\n生成推理样例...")
    model.eval()

    max_new = 4096

    data_path = args.data
    if not os.path.exists(data_path):
        print("  数据文件不存在，跳过样例生成")
        return

    # 读取前 num_examples 条记录的原始 prompt
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            samples.append(r)
            if len(samples) >= num_examples:
                break

    examples = []
    all_completion_tokens = []
    num_natural_end = 0

    for i, r in enumerate(samples):
        orig_system = strip_posthoc_system(r["prompt"]["system"])
        orig_user   = strip_posthoc_user(r["prompt"]["user"])

        messages = [
            {"role": "system", "content": orig_system},
            {"role": "user",   "content": orig_user},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_ids    = outputs[0][prompt_tokens:]
        completion_tokens = len(completion_ids)
        completion_text   = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # 判断是否自然结束（最后一个 token 是 eos）
        natural_end = (completion_ids[-1].item() == tokenizer.eos_token_id) if len(completion_ids) > 0 else False

        all_completion_tokens.append(completion_tokens)
        if natural_end:
            num_natural_end += 1

        examples.append({
            "id":                r.get("id", f"sample_{i}"),
            "problem_type":      r.get("problem_type", "unknown"),
            "n":                 r.get("n", 0),
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "natural_end":       natural_end,
            "prompt_text":       chat_text,
            "completion_text":   completion_text,
            "reference_output":  r["output"][:500],
        })

        status = "自然结束" if natural_end else "截断"
        print(f"  [{r.get('problem_type')}] 样例{i+1}: "
              f"prompt={prompt_tokens}tok  completion={completion_tokens}tok  [{status}]")

    # ── 输出统计 ─────────────────────────────────────────────────────────
    import numpy as np
    arr = np.array(all_completion_tokens)
    total = len(arr)
    print(f"\n  === 样例统计 (max_new_tokens={max_new}) ===")
    print(f"  样本数:     {total}")
    print(f"  长度 min:   {arr.min()}")
    print(f"  长度 max:   {arr.max()}")
    print(f"  长度 mean:  {arr.mean():.1f}")
    print(f"  长度 median:{np.median(arr):.1f}")
    print(f"  自然结束:   {num_natural_end}/{total} ({num_natural_end/total:.1%})")
    print(f"  截断:       {total - num_natural_end}/{total} ({(total - num_natural_end)/total:.1%})")

    out_path = os.path.join(args.output_dir, "final_model", "examples.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"\n  样例已保存到: {out_path}")


if __name__ == "__main__":
    main()
