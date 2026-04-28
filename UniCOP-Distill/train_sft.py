"""
SFT 训练脚本：用蒸馏数据（Gemini 推理链）微调 DeepSeek-R1-Distill 模型。

数据来源：generate_chains.py 生成的 chains.jsonl
训练目标：模型根据原始问题描述，直接生成 <think>...</think> 推理链 + 格式化答案

⚠️ 重要: 数据构造方式在 2026-04-21 后改为"绕开 chat_template 手动拼接",
    原因是 R1-Distill 的 chat_template 会吃掉 assistant 消息里的 <think>...</think>
    推理链(多轮对话优化的副作用),直接 apply_chat_template 会让 SFT 实际训练数据
    丢失全部 thinking 内容,模型学不到推理链。详见技术配置库 LLM训练踩坑.md。

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

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


# ── 默认参数 ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "/home/ntu/lzj/Model/model/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DATA  = "data/chains_v3_clean.jsonl"

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
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr":           "auto",
                "betas":        "auto",
                "eps":          "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr":    0,
                "warmup_max_lr":    "auto",
                "warmup_num_steps": "auto",
                "total_num_steps":  "auto",
            },
        },
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
            "offload_optimizer": {
                "device":     "cpu",
                "pin_memory": True,
            },
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


def load_sft_dataset(data_path: str, tokenizer, max_length: int,
                     max_output_length: int = 4096) -> Dataset:
    """
    从 chains.jsonl 加载数据，构建 SFT 训练集。

    ⚠️ 关键踩坑修复 (2026-04-21):
        R1-Distill 的 chat_template 对 **所有** role='assistant' 的 content
        自动执行 `content.split('</think>')[-1]`,把 <think>...</think> 推理链
        整段吃掉。这是 R1 为多轮对话设计的"历史剥离",但在 SFT 场景下会让
        模型看到的训练 text 里完全没有 <think>thinking</think>, 模型根本学
        不到推理链,只学到"<Assistant> 后直接跟 Route:"这个空架子。
        参见技术配置库: LLM 训练踩坑.md § R1 chat_template 吃 think 链。

    正确做法:
      - 用 apply_chat_template 只渲染 [system, user] 部分 + add_generation_prompt=True,
        末尾天然得到 "...<|Assistant|><think>\\n" (chat_template 官方行为)
      - 手动剥去 Gemini output 开头的 <think>\\n (避免和 prompt 末尾重复)
      - 拼成完整 text + eos_token

    每条样本渲染后的训练 text 结构:
        <bos>...<|User|>Plan route...<|Assistant|><think>\\n
        I need to find the shortest route...
        </think>
        Route: 0 -> ... -> 0
        <eos>
    """
    records = []
    skipped = 0
    skipped_too_long = 0

    # ── 探针: 首次渲染时验证 chat_template 是否如预期在末尾加 <think>\n ──
    # (如果 R1 换代导致 chat_template 行为变化, 早期告警避免静默退化)
    probe_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "probe"},
         {"role": "user",   "content": "probe"}],
        tokenize=False, add_generation_prompt=True,
    )
    probe_ends_with_think = probe_prompt.rstrip().endswith("<think>")
    if probe_ends_with_think:
        print("  [OK  ] chat_template 末尾自动加 <think>, 使用绕过 chat_template 的手动拼接方案")
    else:
        print("  [WARN] chat_template 末尾没有 <think> 前缀, 将在手动拼接时显式补一个")
        print(f"         (探针末尾 50 字: {probe_prompt[-50:]!r})")

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

            # 1. 用 chat_template 只渲染到 <|Assistant|><think>\n 为止 (作为 prompt)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": orig_system},
                 {"role": "user",   "content": orig_user}],
                tokenize=False, add_generation_prompt=True,
            )

            # 2. 探针兜底: 如果 chat_template 没自动加 <think>,手动补
            if not probe_ends_with_think:
                prompt_text = prompt_text.rstrip() + "<think>\n"

            # 3. Gemini output 开头必有 <think> (generate_chains.py 强制),
            #    和 prompt 末尾的 <think> 重复,剥掉
            output_stripped = output.lstrip()
            if output_stripped.startswith("<think>"):
                output_stripped = output_stripped[len("<think>"):].lstrip("\n")

            # 4. completion = thinking + </think> + answer + eos
            #    (eos_token 显式加, 让模型学会"完整答案结束就停止")
            completion_text = output_stripped + tokenizer.eos_token

            # output token 超长过滤
            completion_token_len = len(tokenizer.encode(completion_text))
            if completion_token_len > max_output_length:
                skipped_too_long += 1
                continue

            records.append({
                "prompt":     prompt_text,
                "completion": completion_text,
                "problem_type": r.get("problem_type", "unknown"),
                "n": r.get("n", 0),
            })

    # ── 抽样打印第 1 条的 prompt 末尾 + completion 开头, 人眼验证结构 ──
    if records:
        first = records[0]
        print(f"  [首条样本验证] prompt 末尾 80 字: {first['prompt'][-80:]!r}")
        print(f"                  completion 开头 120 字: {first['completion'][:120]!r}")
        has_think_close = "</think>" in first["completion"]
        has_route = ("Route" in first["completion"]) or ("路径" in first["completion"]) or ("路线" in first["completion"])
        print(f"                  completion 含 </think>: {has_think_close}, 含 Route 类关键词: {has_route}")
        if not has_think_close:
            print(f"  [FAIL] completion 没有 </think>, thinking chain 不完整!")

    if skipped:
        print(f"  跳过 {skipped} 条无效记录")
    if skipped_too_long:
        print(f"  过滤 {skipped_too_long} 条 output token > {max_output_length} 的样本")

    # 按问题类型统计
    from collections import Counter
    type_counts = Counter(r["problem_type"] for r in records)
    size_counts = Counter(r["n"] for r in records)
    print(f"  加载 {len(records)} 条样本")
    print(f"  问题类型分布: {dict(sorted(type_counts.items()))}")
    print(f"  规模分布:     {dict(sorted(size_counts.items()))}")

    # 统计超长样本 (prompt + completion 合计)
    num_over = 0
    for r in records:
        token_len = len(tokenizer.encode(r["prompt"])) + len(tokenizer.encode(r["completion"]))
        if token_len > max_length:
            num_over += 1
    if num_over:
        print(f"  WARNING: {num_over}/{len(records)} 条样本超过 max_length={max_length}，训练时将被截断")

    return Dataset.from_dict({
        "prompt":     [r["prompt"]     for r in records],
        "completion": [r["completion"] for r in records],
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
    parser.add_argument("--max_length", type=int, default=8192,
                        help="最大序列长度（prompt + completion）。COP 大规模问题"
                             "(CVRP/VRPTW n=50/100) 的 prompt 就有 2000-3000 token,"
                             "加 <think> 链后 4096 会砍掉 50%+ 样本尾部的答案, 故默认 8192。"
                             "tokenizer.model_max_length=16384 还有余量,显存紧可降。")
    parser.add_argument("--max_output_length", type=int, default=4096,
                        help="completion（output）部分的最大 token 数，超过此值的样本"
                             "在训练前被过滤掉，不参与 SFT。")
    parser.add_argument("--val_ratio",    type=float, default=0.0,
                        help="验证集比例（默认 0 不验证）")

    # 训练
    parser.add_argument("--seed",         type=int,   default=42,
                        help="全局随机种子（可复现性）")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="学习率 (LoRA 默认 1e-4, 全参微调 --no_lora 时建议降到 2e-5)。"
                             "当前默认针对 LoRA 场景;官方 DeepSeek-R1-Distill LoRA SFT 配置 1e-4~2e-4。")
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
    print(f"  最大output: {args.max_output_length}")
    print(f"  ZeRO:       {args.zero_stage}")
    print(f"  输出:       {args.output_dir}")
    print(f"{'='*60}\n")

    # ── 加载 tokenizer ───────────────────────────────────────────────────
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ⚠️ 关键坑: 不能 pad_token = eos_token
    #    原因: collator 会把所有 pad_token 位置 label 置 -100 (不参与 loss),
    #    如果 pad_token == eos_token, 则所有 eos 位置也被一起 mask,
    #    模型永远学不到"何时生成 EOS",训练后会 无限生成 / 不停止。
    #    正确: R1 系列已自带专用 pad token `<｜▁pad▁｜>`,优先用它。
    _pad_candidates = ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>"]
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        print(f"  ✓ tokenizer 自带独立 pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    else:
        _pad_set = False
        for cand in _pad_candidates:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
                tokenizer.pad_token = cand
                print(f"  ✓ 设置 pad_token = {cand!r} (id={tid})")
                _pad_set = True
                break
        if not _pad_set:
            # 最后兜底: 新加一个 pad special token + resize embedding
            print("  ⚠️ tokenizer 无专用 pad,新建 '<|pad|>' 作 pad_token")
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print(f"     新 pad_token_id = {tokenizer.pad_token_id}")
            print("     下游会 resize_token_embeddings(len(tokenizer))")
        assert tokenizer.pad_token_id != tokenizer.eos_token_id, \
            "pad_token == eos_token 还是没逃掉! 请检查 tokenizer 配置"

    tokenizer.padding_side = "right"
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}, "
          f"padding_side={tokenizer.padding_side}")

    # ── 加载数据 ─────────────────────────────────────────────────────────
    print("加载训练数据...")
    dataset = load_sft_dataset(args.data, tokenizer, args.max_length, args.max_output_length)

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

    # 如果 pad_token 阶段我们新加了 token,embedding 必须 resize
    # (否则 pad_token_id 超出 model embedding 范围,前向 forward 立刻 index error)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        print(f"  resize_token_embeddings {model.get_input_embeddings().num_embeddings} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # ZeRO-3 + LoRA + gradient_checkpointing 三件套必需: 在 PEFT wrap (SFTTrainer 内部)
    # 之前显式调 enable_input_require_grads,否则 backward 可能报
    # "element 0 of tensors does not require grad" / "Recomputed values shape [0]"。
    # 参考 transformers#26334, peft#1142。
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

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
        # ZeRO-3 + LoRA + gc 组合下,non-reentrant checkpoint 和 ZeRO-3 partition 交互
        # 会报 "Recomputed values shape [0]"。社区 workaround 是强制 use_reentrant=True
        # (trl#2514, peft#1142)。与 UniCOP-Reason/train.py 保持一致。
        gradient_checkpointing_kwargs={"use_reentrant": True},
        # 不再用 dataset_text_field="text" (language modeling 模式,会对 prompt 也做 loss)
        # 改用 prompt+completion 格式,显式 completion_only_loss=True 只对 completion 做 loss。
        # 这样 prompt (system/user/<|Assistant|><think>\n) 被 mask 成 -100,
        # 梯度全部用在 thinking + </think> + Route + <eos> 上, 效率翻倍。
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

    print("\n开始 SFT 训练...")
    trainer.train()

    # ── 保存 ─────────────────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存到: {save_path}")



if __name__ == "__main__":
    main()
