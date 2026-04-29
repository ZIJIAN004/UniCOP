"""
Stage 1 SFT：用 solver 解训练 Qwen2.5-Instruct 生成合法的解（不训练思维链）。

基座模型：Qwen2.5-7B-Instruct（非推理模型，无 <think> 机制）
训练目标：模型看到问题描述后，直接输出格式正确、节点不遗漏不重复的解。
数据来源：generate_solutions.py 生成的 solutions.jsonl
损失函数：completion_only_loss=True，只在 solution token 上算 loss。

与 Stage 2 的区别：
  - Stage 1: Qwen2.5-Instruct + prompt → solution (无 <think>，只学可行性)
  - Stage 2: Stage1 ckpt + prompt → <think>...</think> + solution (从零学推理链)

运行示例:
    python stage1_solution/train_sft_stage1.py
    accelerate launch --num_processes 4 stage1_solution/train_sft_stage1.py --zero_stage 2
"""

import argparse
import json
import os
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


DEFAULT_MODEL = "/home/ntu/lzj/Model/model/Qwen2.5-7B-Instruct"
DEFAULT_DATA = "data/solutions.jsonl"


def _strip_think_instructions(system: str) -> str:
    """从 system prompt 中剥离 <think> 推理指令（Stage 1 不需要思维链引导）。"""
    system = re.sub(
        r'Before answering, think through the problem in <think>\.\.\.</think>\.[^\n]*\n?',
        '', system,
    )
    system = system.replace("After completing your analysis, output", "Output")
    system = re.sub(r'\n{3,}', '\n\n', system).strip()
    return system


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
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto", "betas": "auto",
                "eps": "auto", "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0, "warmup_max_lr": "auto",
                "warmup_num_steps": "auto", "total_num_steps": "auto",
            },
        },
    }

    if zero_stage == 2:
        base["zero_optimization"] = {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "reduce_scatter": True,
        }
    elif zero_stage == 3:
        base["zero_optimization"] = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        }

    return base


def _setup_pad_token(tokenizer):
    """确保 pad_token != eos_token，否则 EOS 会被 label mask 吃掉。"""
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        print(f"  pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
        return False

    for cand in ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>"]:
        tid = tokenizer.convert_tokens_to_ids(cand)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
            tokenizer.pad_token = cand
            print(f"  pad_token = {cand!r} (id={tid})")
            return False

    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    print(f"  新建 pad_token '<|pad|>' (id={tokenizer.pad_token_id})")
    return True  # 需要 resize embedding


def _detect_think_suffix(tokenizer) -> str:
    """
    检测 chat_template add_generation_prompt 末尾是否带 <think>。
    Stage 1 需要剥离它，使 prompt 止于 <|Assistant|>。
    不碰 <think> 机制，避免压制推理模型的思维链能力。
    """
    probe = tokenizer.apply_chat_template(
        [{"role": "system", "content": "p"}, {"role": "user", "content": "p"}],
        tokenize=False, add_generation_prompt=True,
    )
    if probe.rstrip().endswith("<think>"):
        idx = probe.rfind("<think>")
        suffix = probe[idx:]
        print(f"  chat_template 末尾带 <think>，Stage 1 将剥离（suffix={suffix!r}）")
        return suffix
    print(f"  chat_template 末尾无 <think>")
    return ""


def load_stage1_dataset(data_path: str, tokenizer, max_length: int,
                        think_suffix: str,
                        filter_problems=None, filter_sizes=None) -> Dataset:
    """
    加载 solutions.jsonl，构建 Stage 1 训练集。

    训练 text 结构（不含 <think>，不干扰推理模型的思维链机制）:
        prompt:     ...<|User|>[问题描述]<|Assistant|>
        completion: [solution]\\n<eos>

    Stage 1 只教模型"什么是合法的解"，完全不碰 <think> 相关 token。
    Stage 2 再引入 <think>...\\n</think> 格式教推理链。
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

            if filter_problems and r.get("problem_type") not in filter_problems:
                continue
            if filter_sizes and r.get("n") not in filter_sizes:
                continue

            system = _strip_think_instructions(r["prompt"]["system"])
            user = r["prompt"]["user"]
            solution = r["solution"]

            if not solution or not solution.strip():
                skipped += 1
                continue

            # 1. 渲染 prompt（chat_template 会加 <|Assistant|><think>\n）
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )

            # 2. 剥离 <think> 后缀，使 prompt 止于 <|Assistant|>
            if think_suffix and prompt_text.endswith(think_suffix):
                prompt_text = prompt_text[:-len(think_suffix)]
            elif think_suffix:
                idx = prompt_text.rstrip().rfind("<think>")
                if idx != -1:
                    prompt_text = prompt_text[:idx]

            # 3. completion = solution + eos（纯解，不含任何 think 标签）
            completion_text = solution + "\n" + tokenizer.eos_token

            # 超长过滤
            total_len = len(tokenizer.encode(prompt_text + completion_text))
            if total_len > max_length:
                skipped += 1
                continue

            records.append({
                "prompt": prompt_text,
                "completion": completion_text,
                "problem_type": r.get("problem_type", "unknown"),
                "n": r.get("n", 0),
            })

    if records:
        first = records[0]
        print(f"  [首条验证] prompt 末尾 80 字: {first['prompt'][-80:]!r}")
        print(f"              completion 前 120 字: {first['completion'][:120]!r}")
        has_think = "<think>" in first["prompt"][-30:] or "<think>" in first["completion"]
        if has_think:
            print(f"  [WARN] 检测到残留 <think> 标签，Stage 1 不应包含!")
        else:
            print(f"  [OK] 无 <think> 标签，不干扰推理能力")

    if skipped:
        print(f"  跳过 {skipped} 条记录")

    from collections import Counter
    type_counts = Counter(r["problem_type"] for r in records)
    size_counts = Counter(r["n"] for r in records)
    print(f"  加载 {len(records)} 条样本")
    print(f"  问题类型: {dict(sorted(type_counts.items()))}")
    print(f"  规模分布: {dict(sorted(size_counts.items()))}")

    return Dataset.from_dict({
        "prompt": [r["prompt"] for r in records],
        "completion": [r["completion"] for r in records],
    })


def main():
    parser = argparse.ArgumentParser(description="Stage 1 SFT：solver 解训练（无思维链）")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--filter_problems", type=str, nargs="+", default=None,
                        help="只训练指定问题类型，如 tsp cvrp")
    parser.add_argument("--filter_sizes", type=int, nargs="+", default=None,
                        help="只训练指定规模，如 20 50")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Stage 1 不含推理链，序列较短，4096 足够")
    parser.add_argument("--val_ratio", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2,
                        help="Stage 1 不宜过多 epoch，避免压制推理能力")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="解比推理链短很多，可以开更大 batch")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    parser.add_argument("--zero_stage", type=int, default=0, choices=[0, 2, 3])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--output_dir", type=str, default="./output_sft_stage1")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Stage 1 SFT: Solver 解训练（无思维链）")
    print(f"  模型:   {args.model}")
    print(f"  数据:   {args.data}")
    print(f"  LoRA:   {'关闭' if args.no_lora else f'rank={args.lora_rank}'}")
    print(f"  epochs: {args.epochs}  batch: {args.batch_size}×{args.grad_accum}")
    print(f"  ZeRO:   {args.zero_stage}")
    print(f"{'='*60}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    need_resize = _setup_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id

    think_suffix = _detect_think_suffix(tokenizer)

    # ── 数据 ─────────────────────────────────────────────────────────────
    print("加载训练数据...")
    dataset = load_stage1_dataset(args.data, tokenizer, args.max_length, think_suffix,
                                  filter_problems=args.filter_problems,
                                  filter_sizes=args.filter_sizes)

    if args.val_ratio > 0 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=args.val_ratio, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
        print(f"训练集: {len(train_dataset)}  验证集: {len(eval_dataset)}")
    else:
        train_dataset, eval_dataset = dataset, None
        print(f"训练集: {len(train_dataset)}")

    # ── 模型 ─────────────────────────────────────────────────────────────
    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    if need_resize:
        print(f"  resize embeddings → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = None
    if not args.no_lora:
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
        save_total_limit=2,
        bf16=True,
        seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
        deepspeed=ds_config,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        completion_only_loss=True,
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

    print("\n开始 Stage 1 SFT 训练...")
    trainer.train()

    save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    if peft_config is not None:
        if trainer.accelerator.is_main_process:
            print("合并 LoRA adapter 到基座模型...")
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
                trust_remote_code=True, device_map="cpu",
            )
            merged = PeftModel.from_pretrained(base, save_path).merge_and_unload()
            merged.save_pretrained(save_path)
            del base, merged
            torch.cuda.empty_cache()
        trainer.accelerator.wait_for_everyone()

    print(f"\n模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
