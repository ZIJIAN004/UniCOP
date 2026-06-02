#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_sft_foarl.py — FOARL CVRP SFT (Stage-1), Qwen3-4B-Instruct-2507 + chat template。

复现 FOARL (LLMCoSolver) 的 Stage-1 SFT: 在直接解(无推理链)上微调, 学会输出
"Routes: [[0,..,0],..], Objective: X"。内容用 FOARL 原版 (instruction/input/output,
见 build_foarl_cvrp_data.py), 外壳用 Instruct-2507 的 chat template, completion-only
只在 assistant 段算 loss。无 <think>——这正是"无推理"对照臂的关键。

与 UniCOP-Distill/stage2_reasoning/train_sft_stage2.py 复用同一套栈
(transformers+TRL+DeepSpeed ZeRO-3 + LoRA + completion-only collator), 但去掉了
chat<think> 注入 / posthoc 剥离等思维链专用逻辑。

chat 渲染 (每条样本):
    <|im_start|>system\n{FOARL preamble}<|im_end|>
    <|im_start|>user\n{instruction}\n\n{input}<|im_end|>
    <|im_start|>assistant\n            ← response_template, 之前全 mask
    Routes: [[0,..,0],..], Objective: X<eos>

单卡:  python train_sft_foarl.py --model <Qwen3-4B-Instruct-2507> --data data/foarl_cvrp20.jsonl
多卡:  accelerate launch --num_processes 4 train_sft_foarl.py --zero_stage 3 \
           --gradient_checkpointing --model <...> --data data/foarl_cvrp20.jsonl
⚠️ 首次跑务必看 [首条样本验证] 探针: 确认 response_template 命中、completion 以 Routes 开头。
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from peft import LoraConfig

# FOARL Alpaca 模板的开场白 (放进 system role, 保留 FOARL 原版内容)
FOARL_PREAMBLE = (
    "Below is an instruction describing a combinatorial optimization problem. "
    "It is paired with an input that provides the data of the instance. "
    "Your task is to produce a feasible solution that optimizes (minimizes or maximizes) "
    "the given objective."
)


# ── DeepSpeed 配置 (与 stage2 一致) ──────────────────────────────────────────────
def make_deepspeed_config(zero_stage: int):
    if zero_stage == 0:
        return None
    base = {
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "steps_per_print": 50,
        "optimizer": {"type": "AdamW",
                      "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"}},
        "scheduler": {"type": "WarmupDecayLR",
                      "params": {"warmup_min_lr": 0, "warmup_max_lr": "auto",
                                 "warmup_num_steps": "auto", "total_num_steps": "auto"}},
    }
    if zero_stage == 2:
        base["zero_optimization"] = {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                                     "reduce_bucket_size": 5e8, "reduce_scatter": True}
    elif zero_stage == 3:
        base["zero_optimization"] = {
            "stage": 3, "overlap_comm": True, "contiguous_gradients": True,
            "reduce_bucket_size": "auto", "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9, "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        }
    return base


def _detect_response_template(tokenizer) -> str:
    """探测 chat_template 的 assistant generation prompt 作为 completion-only 的分界。

    Instruct-2507 是非思维模型, generation prompt 形如 '<|im_start|>assistant\\n'
    (不含 <think>)。这里动态取真实值, 不写死, 避免 chat_template 改版后静默错位。
    """
    msgs = [{"role": "user", "content": "Hi"}]
    without_gp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    with_gp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return with_gp[len(without_gp):]


def load_foarl_dataset(data_paths, tokenizer, max_length, max_output_length):
    """加载 FOARL jsonl ({instruction,input,output}), chat 渲染成 prompt+completion。"""
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    # 探针: 看 generation prompt 是否被 chat_template 塞了 <think> (Instruct 不应有)
    probe = tokenizer.apply_chat_template(
        [{"role": "system", "content": "probe"}, {"role": "user", "content": "probe"}],
        tokenize=False, add_generation_prompt=True)
    if "<think>" in probe[-40:]:
        print("  [WARN] chat_template 末尾出现 <think>! 这不像非思维 Instruct, 确认基座是否选错"
              f" (探针末尾: {probe[-40:]!r})")
    else:
        print("  [OK  ] chat_template generation prompt 不含 <think> (符合非思维 Instruct)")

    records = []
    skipped_empty = skipped_long = 0
    for path in data_paths:
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                instruction = r.get("instruction", "")
                inp = r.get("input", "")
                output = r.get("output", "")
                if not output.strip():
                    skipped_empty += 1
                    continue

                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": FOARL_PREAMBLE},
                     {"role": "user", "content": f"{instruction}\n\n{inp}"}],
                    tokenize=False, add_generation_prompt=True)
                completion_text = output + tokenizer.eos_token

                if len(tokenizer.encode(completion_text)) > max_output_length:
                    skipped_long += 1
                    continue
                records.append({"prompt": prompt_text, "completion": completion_text})
                loaded += 1
        print(f"  [{path}] 加载 {loaded} 条")

    # 长度过滤
    before = len(records)
    records = [r for r in records
               if len(tokenizer.encode(r["prompt"])) + len(tokenizer.encode(r["completion"])) <= max_length]
    if before - len(records):
        print(f"  过滤 {before - len(records)} 条 prompt+completion > max_length={max_length}")
    if skipped_empty:
        print(f"  跳过 {skipped_empty} 条空 output")
    if skipped_long:
        print(f"  过滤 {skipped_long} 条 output token > {max_output_length}")

    # 首条样本验证 (人眼/CI 都能看)
    if records:
        first = records[0]
        print(f"  [首条样本验证] prompt 末尾 60 字: {first['prompt'][-60:]!r}")
        print(f"                  completion 开头 60 字: {first['completion'][:60]!r}")
        if "Routes:" not in first["completion"]:
            print("  [FAIL] completion 不含 'Routes:', 数据/格式可能不对!")
    else:
        raise RuntimeError("没有可用样本, 检查数据路径与 max_length")

    print(f"  最终 {len(records)} 条")
    return Dataset.from_dict({"text": [r["prompt"] + r["completion"] for r in records]})


def main():
    ap = argparse.ArgumentParser(description="FOARL CVRP Stage-1 SFT (Instruct + chat template, 无 think)")
    ap.add_argument("--model", required=True, help="Qwen3-4B-Instruct-2507 路径")
    ap.add_argument("--data", nargs="+", default=["data/foarl_cvrp20.jsonl"])
    ap.add_argument("--output_dir", default="./output_sft_foarl_cvrp20")
    # LoRA: 对齐 UniCOP-Distill 思维臂 (r=64, alpha=128), 保证消融只差"有无 think"
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    # 训练: 对齐 distill 思维臂超参 (lr=2e-5, 3 epoch, bs1·ga8, warmup0.05), 非 FOARL recipe
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=4864,
                    help="对齐 distill SFT_MAX_LENGTH=4864 (CVRP n=20 实际只需 ~800 tok)")
    ap.add_argument("--max_output_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0, help="只取前 N 条 sanity, 0=全量")
    # 多卡
    ap.add_argument("--zero_stage", type=int, default=0, choices=[0, 2, 3])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None, help="路径或 'auto'")
    # 输出/日志
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--use_wandb", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print("  FOARL CVRP Stage-1 SFT (Instruct + chat template, 无 think)")
    print(f"  模型: {args.model}")
    print(f"  数据: {args.data}")
    print(f"  LoRA: {'关闭' if args.no_lora else f'r={args.lora_rank}/α={args.lora_alpha}'} | "
          f"lr={args.lr} epochs={args.epochs} ZeRO={args.zero_stage}")
    print(f"  输出: {args.output_dir}")
    print("=" * 60)

    # ── tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if getattr(tokenizer, "add_bos_token", False):
        tokenizer.add_bos_token = False  # chat_template 已渲染 BOS, 防双 BOS
        print("  ✓ add_bos_token = False")

    # pad_token 不能 == eos_token (否则 collator 把 eos 也 mask, 模型学不会停)
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        print(f"  ✓ 独立 pad_token: {tokenizer.pad_token!r}")
    else:
        _added = False
        for cand in ["<|pad|>", "<|PAD_TOKEN|>"]:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
                tokenizer.pad_token = cand
                print(f"  ✓ pad_token = {cand!r}")
                _added = True
                break
        if not _added:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print(f"  ⚠️ 新建 pad_token <|pad|> (id={tokenizer.pad_token_id}), 下游 resize embedding")
        assert tokenizer.pad_token_id != tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── 数据 ─────────────────────────────────────────────────────────────
    dataset = load_foarl_dataset(args.data, tokenizer, args.max_length, args.max_output_length)
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
        print(f"  [sanity] 截取前 {args.max_samples} 条")

    # ── 模型 ─────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        print(f"  resize embedding → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()  # ZeRO-3+LoRA+gc 三件套必需

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    sft_config = SFTConfig(
        output_dir=args.output_dir, max_length=args.max_length,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps, save_steps=args.save_steps, save_total_limit=3,
        bf16=True, seed=args.seed, report_to="wandb" if args.use_wandb else "none",
        deepspeed=make_deepspeed_config(args.zero_stage),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": args.zero_stage == 3},
        lr_scheduler_type="cosine", weight_decay=0.01,
    )

    # completion-only: 只在 assistant 段算 loss
    resp_str = _detect_response_template(tokenizer)
    resp_ids = tokenizer.encode(resp_str, add_special_tokens=False)
    print(f"  response_template: {resp_str!r} → IDs: {resp_ids}")
    collator = DataCollatorForCompletionOnlyLM(response_template=resp_ids, tokenizer=tokenizer)

    # ── completion-only 自检 (金标准): 解码首样本 label!=-100 的 token, 应只剩答案 ──
    #    只在 rank0 跑一次。空 → response_template 没命中, mask 全掉, loss 学不到东西;
    #    含 prompt 内容 → mask 边界错位。正确结果: 只有 "Routes: ..., Objective: ...<eos>"。
    if os.environ.get("LOCAL_RANK", "0") == "0":
        _enc = tokenizer(dataset[0]["text"], add_special_tokens=False)
        _b = collator([{"input_ids": _enc["input_ids"], "attention_mask": _enc["attention_mask"]}])
        _kept = [t for t, l in zip(_enc["input_ids"], _b["labels"][0].tolist()) if l != -100]
        _dec = tokenizer.decode(_kept)
        print(f"  [completion-only 自检] 参与 loss token 数={len(_kept)}; 解码={_dec[:160]!r}")
        if len(_kept) == 0:
            print("  [FAIL] 无 token 参与 loss → response_template 没命中, completion-only 失效!")
        elif "Routes:" not in _dec:
            print("  [WARN] 参与 loss 的内容不含 'Routes:', mask 边界可能错位")

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=peft_config, data_collator=collator)

    resume = args.resume_from_checkpoint
    if resume == "auto":
        ckpts = [(int(d.split("-")[1]), os.path.join(args.output_dir, d))
                 for d in (os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else [])
                 if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
        resume = max(ckpts)[1] if ckpts else None
        print(f"  resume=auto → {resume}")
    print("\n开始 SFT...")
    trainer.train(resume_from_checkpoint=resume) if resume else trainer.train()

    save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存到: {save_path}")
    if peft_config is not None and trainer.accelerator.is_main_process:
        print("LoRA adapter 已存; eval/RL 前需 merge")


if __name__ == "__main__":
    main()
