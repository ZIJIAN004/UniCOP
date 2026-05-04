"""
Latent-SFT 训练脚本：CODI 式隐式推理训练。

使用 accelerate 做分布式，自定义训练循环处理 teacher/student 双 forward。

单卡运行:
    python train.py
    python train.py --model ./path/to/grpo_model --data ../UniCOP-Distill/data/chains_self_cvrp20.jsonl

多卡运行:
    accelerate launch --num_processes 4 train.py --zero_stage 2 --gradient_checkpointing
"""

import argparse
import math
import os
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, set_seed
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from config import LatentSFTConfig
from data_utils import CODIDataset, add_special_tokens, collate_codi
from model import LatentEmbeddings, compute_codi_loss


def make_deepspeed_config(zero_stage: int) -> dict | None:
    if zero_stage == 0:
        return None

    base = {
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
    }

    if zero_stage == 2:
        base["zero_optimization"] = {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
        }
    elif zero_stage == 3:
        base["zero_optimization"] = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "gather_16bit_weights_on_model_save": True,
        }

    return base


def train(cfg: LatentSFTConfig):
    ds_config = make_deepspeed_config(cfg.zero_stage)
    ds_plugin = None
    if ds_config is not None:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=ds_plugin,
    )

    set_seed(cfg.seed)

    if accelerator.is_main_process:
        print(f"{'=' * 60}")
        print(f"  Latent-SFT (CODI) 训练")
        print(f"  模型:          {cfg.model_name}")
        print(f"  数据:          {cfg.data_path}")
        print(f"  压缩比:        {cfg.latent_compression_ratio}:1")
        print(f"  Loss 权重:     α={cfg.alpha} β={cfg.beta} γ={cfg.gamma}")
        print(f"  ZeRO:          {cfg.zero_stage}")
        print(f"{'=' * 60}\n")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    if getattr(tokenizer, "add_bos_token", False):
        tokenizer.add_bos_token = False

    # pad_token 处理（与 Distill 保持一致）
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        _pad_candidates = ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>"]
        _pad_set = False
        for cand in _pad_candidates:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
                tokenizer.pad_token = cand
                _pad_set = True
                break
        if not _pad_set:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"

    num_added, latent_id = add_special_tokens(tokenizer)

    # ── 模型 ──
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # LoRA
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    # ── Latent embeddings ──
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.to_dict().get("hidden_size", 3584)
    latent_emb = LatentEmbeddings(hidden_size, cfg.latent_init_std)

    # ── 数据 ──
    dataset = CODIDataset(
        cfg.data_path, tokenizer,
        max_length=cfg.max_length,
        latent_compression_ratio=cfg.latent_compression_ratio,
        filter_problems=cfg.filter_problems,
        filter_sizes=cfg.filter_sizes,
    )

    collate_fn = partial(collate_codi, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer: 模型参数走 DeepSpeed，latent embeddings 单独管理 ──
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    latent_optimizer = AdamW(latent_emb.parameters(), lr=cfg.latent_lr, weight_decay=0.0)

    # ── Accelerate prepare（先 prepare，再用 sharded dataloader 长度算 scheduler）──
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    latent_emb = latent_emb.to(accelerator.device)

    num_training_steps = math.ceil(
        len(dataloader) / cfg.gradient_accumulation_steps
    ) * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    # ── 训练循环 ──
    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                total_loss, t_loss, s_loss, a_loss = compute_codi_loss(
                    model=model,
                    latent_emb=latent_emb,
                    teacher_input_ids=batch["teacher_input_ids"],
                    teacher_attention_mask=batch["teacher_attention_mask"],
                    teacher_labels=batch["teacher_labels"],
                    student_input_ids=batch["student_input_ids"],
                    student_attention_mask=batch["student_attention_mask"],
                    student_labels=batch["student_labels"],
                    latent_positions=batch["latent_positions"],
                    align_pairs=batch["align_pairs"],
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                )

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    if accelerator.num_processes > 1 and latent_emb.embedding.grad is not None:
                        torch.distributed.all_reduce(latent_emb.embedding.grad, op=torch.distributed.ReduceOp.AVG)

                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(latent_emb.parameters(), cfg.max_grad_norm)

                    latent_optimizer.step()
                    latent_optimizer.zero_grad()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / (step + 1)
                    lr_current = scheduler.get_last_lr()[0]
                    print(
                        f"  [epoch {epoch+1}/{cfg.num_epochs}] "
                        f"step {global_step} | "
                        f"loss={avg_loss:.4f} "
                        f"(teacher={t_loss:.4f} student={s_loss:.4f} align={a_loss:.4f}) | "
                        f"lr={lr_current:.2e}"
                    )

                if global_step % cfg.save_steps == 0:
                    _save_checkpoint(accelerator, model, latent_emb, tokenizer, cfg, global_step)

    # ── 最终保存 ──
    _save_checkpoint(accelerator, model, latent_emb, tokenizer, cfg, "final")

    if accelerator.is_main_process:
        print(f"\n训练完成！模型保存到: {cfg.output_dir}")


def _save_checkpoint(accelerator, model, latent_emb, tokenizer, cfg, tag):
    accelerator.wait_for_everyone()
    save_path = os.path.join(cfg.output_dir, f"checkpoint-{tag}")
    unwrapped = accelerator.unwrap_model(model)
    # ZeRO-3: all ranks must participate in weight gather
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        unwrapped.save_pretrained(save_path, state_dict=state_dict)
        tokenizer.save_pretrained(save_path)
        torch.save(latent_emb.state_dict(), os.path.join(save_path, "latent_embeddings.pt"))
        print(f"  ✓ checkpoint 保存到: {save_path}")
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description="Latent-SFT (CODI) 训练")

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--compression_ratio", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--latent_lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--zero_stage", type=int, default=None, choices=[0, 2, 3])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=None)

    args = parser.parse_args()

    cfg = LatentSFTConfig()

    # 命令行覆盖
    if args.model is not None: cfg.model_name = args.model
    if args.data is not None: cfg.data_path = args.data
    if args.compression_ratio is not None: cfg.latent_compression_ratio = args.compression_ratio
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.beta is not None: cfg.beta = args.beta
    if args.gamma is not None: cfg.gamma = args.gamma
    if args.lr is not None: cfg.lr = args.lr
    if args.latent_lr is not None: cfg.latent_lr = args.latent_lr
    if args.epochs is not None: cfg.num_epochs = args.epochs
    if args.batch_size is not None: cfg.per_device_batch_size = args.batch_size
    if args.grad_accum is not None: cfg.gradient_accumulation_steps = args.grad_accum
    if args.zero_stage is not None: cfg.zero_stage = args.zero_stage
    if args.gradient_checkpointing: cfg.gradient_checkpointing = True
    if args.output_dir is not None: cfg.output_dir = args.output_dir
    if args.save_steps is not None: cfg.save_steps = args.save_steps

    train(cfg)


if __name__ == "__main__":
    main()
