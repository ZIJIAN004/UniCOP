"""
HLR 训练脚本 (Hierarchical Latent Reasoner).

使用 accelerate 做分布式, 自定义训练循环处理 teacher/student 双 forward + LR 段对齐.

单卡运行 (cwd = UniCOP 根目录):
    python Latent-SFT/train.py --model "$BASE_MODEL" --data Latent-SFT/data/profiled_cvrp20.jsonl

多卡运行 (cwd = UniCOP 根目录):
    accelerate launch --num_processes 4 Latent-SFT/train.py --zero_stage 3 --gradient_checkpointing
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

from config import HLRConfig
from data_utils import HLRDataset, collate_hlr
from model import build_latent_reasoner_from_main, compute_hlr_loss


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


def train_hlr(cfg: HLRConfig):
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
        print(f"  HLR (Hierarchical Latent Reasoner) 训练")
        print(f"  模型:           {cfg.model_name}")
        print(f"  数据:           {cfg.data_path}")
        print(f"  LR 层数:        {cfg.lr_num_layers}")
        print(f"  LR hidden:      {cfg.lr_hidden_size}")
        print(f"  GQA:            Q={cfg.lr_num_heads} KV={cfg.lr_num_kv_heads} d={cfg.lr_head_dim}")
        print(f"  压缩比:         {cfg.latent_compression_ratio}:1")
        print(f"  Loss 权重:      α={cfg.alpha} β={cfg.beta} γ={cfg.gamma}")
        print(f"  ZeRO:           {cfg.zero_stage}")
        print(f"{'=' * 60}\n")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    if getattr(tokenizer, "add_bos_token", False):
        tokenizer.add_bos_token = False

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

    # ── 主模型 ──
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # ── Auto-rebuild profiled jsonl (用户没传 --data 时强制重新构造) ──
    # 在 LoRA 包装 / gradient checkpointing 之前用 base model 跑 entropy profile.
    # 这样保证训练数据跟当前用的基座模型的 chat_template / tokenizer 严格一致.
    if cfg.auto_rebuild_data:
        if accelerator.is_main_process:
            from entropy_profile import auto_profile_inplace
            print(f"\n[auto-rebuild] 重新构造 profiled jsonl ...")
            print(f"  原始数据: {cfg.raw_chains_path}")
            print(f"  输出:     {cfg.data_path}")
            device = accelerator.device
            model = model.to(device)
            min_seg = cfg.min_latent_steps * cfg.latent_compression_ratio
            max_seg = cfg.max_latent_steps * cfg.latent_compression_ratio
            auto_profile_inplace(
                model=model,
                tokenizer=tokenizer,
                raw_data_path=cfg.raw_chains_path,
                output_path=cfg.data_path,
                entropy_window=cfg.entropy_window,
                entropy_quantile=cfg.entropy_quantile,
                min_segment=min_seg,
                max_segment=max_seg,
                cooldown=cfg.latent_cooldown,
                max_length=cfg.max_length,
                filter_problems=cfg.filter_problems,
                filter_sizes=cfg.filter_sizes,
            )
            # 移回 CPU, 让 accelerator.prepare 自己决定 device 放置
            model = model.cpu()
        accelerator.wait_for_everyone()

    if cfg.gradient_checkpointing:
        # use_reentrant=True: ZeRO-3 + LoRA + GC 三件套 workaround (踩坑 #14)
        model.config.use_cache = False  # 抑制 GC + use_cache 互斥 warning
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
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

    # ── LatentReasoner ──
    latent_reasoner = build_latent_reasoner_from_main(model, cfg)
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in latent_reasoner.parameters())
        print(f"  LatentReasoner 参数量: {n_params/1e6:.2f}M")

    # ── 数据 ──
    dataset = HLRDataset(
        cfg.data_path, tokenizer,
        max_length=cfg.max_length,
        latent_compression_ratio=cfg.latent_compression_ratio,
        filter_problems=cfg.filter_problems,
        filter_sizes=cfg.filter_sizes,
    )

    collate_fn = partial(collate_hlr, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,   # Phase 1 强制 1
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizers ──
    main_optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    lr_optimizer = AdamW(
        latent_reasoner.parameters(),
        lr=cfg.latent_reasoner_lr,
        weight_decay=cfg.weight_decay,
    )

    # ── Accelerate prepare ──
    # 主模型 + 主 optimizer + dataloader 进 DeepSpeed; LatentReasoner 单独管理
    model, main_optimizer, dataloader = accelerator.prepare(
        model, main_optimizer, dataloader
    )
    latent_reasoner = latent_reasoner.to(accelerator.device)

    num_training_steps = math.ceil(
        len(dataloader) / cfg.gradient_accumulation_steps
    ) * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    main_scheduler = get_cosine_schedule_with_warmup(
        main_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    main_scheduler = accelerator.prepare(main_scheduler)

    lr_scheduler = get_cosine_schedule_with_warmup(
        lr_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ── 训练循环 ──
    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        latent_reasoner.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                total_loss, t_loss, s_loss, a_loss = compute_hlr_loss(
                    model=model,
                    latent_reasoner=latent_reasoner,
                    teacher_input_ids=batch["teacher_input_ids"],
                    teacher_attention_mask=batch["teacher_attention_mask"],
                    teacher_labels=batch["teacher_labels"],
                    prompt_ids=batch["prompt_ids"],
                    segments=batch["segments"],
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                )

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    # 多卡 LatentReasoner 梯度手动 all-reduce (LR 没进 DeepSpeed sharding)
                    if accelerator.num_processes > 1:
                        for p in latent_reasoner.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(
                                    p.grad, op=torch.distributed.ReduceOp.AVG
                                )

                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(
                        latent_reasoner.parameters(), cfg.max_grad_norm
                    )

                    lr_optimizer.step()
                    lr_scheduler.step()
                    lr_optimizer.zero_grad()

                main_optimizer.step()
                main_scheduler.step()
                main_optimizer.zero_grad()

            epoch_loss += total_loss.item()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / (step + 1)
                    main_lr = main_scheduler.get_last_lr()[0]
                    lr_lr = lr_scheduler.get_last_lr()[0]
                    print(
                        f"  [epoch {epoch+1}/{cfg.num_epochs}] "
                        f"step {global_step} | "
                        f"loss={avg_loss:.4f} "
                        f"(teacher={t_loss:.4f} student={s_loss:.4f} align={a_loss:.4f}) | "
                        f"main_lr={main_lr:.2e} lr_lr={lr_lr:.2e}"
                    )

                if global_step % cfg.save_steps == 0:
                    _save_hlr_checkpoint(
                        accelerator, model, latent_reasoner, tokenizer, cfg, global_step
                    )

    # ── 最终保存 ──
    _save_hlr_checkpoint(accelerator, model, latent_reasoner, tokenizer, cfg, "final")

    if accelerator.is_main_process:
        print(f"\nHLR 训练完成！模型保存到: {cfg.output_dir}")


def _save_hlr_checkpoint(accelerator, model, latent_reasoner, tokenizer, cfg, tag):
    accelerator.wait_for_everyone()
    save_path = os.path.join(cfg.output_dir, f"checkpoint-{tag}")

    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)   # ZeRO-3 要所有 rank 参与

    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        unwrapped.save_pretrained(save_path, state_dict=state_dict)
        tokenizer.save_pretrained(save_path)

        # LatentReasoner: 每个 rank 都有完整副本 (没走 DeepSpeed sharding), 主进程直接保存
        torch.save(
            latent_reasoner.state_dict(),
            os.path.join(save_path, "latent_reasoner.pt"),
        )

        print(f"  ✓ HLR checkpoint 保存到: {save_path}")

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description="HLR 训练 (Hierarchical Latent Reasoner)")

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--compression_ratio", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--zero_stage", type=int, default=None, choices=[0, 2, 3])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=None)

    parser.add_argument("--latent_reasoner_lr", type=float, default=None,
                        help="LatentReasoner 的 lr")
    parser.add_argument("--lr_num_layers", type=int, default=None,
                        help="LatentReasoner 层数 (0 = auto 从主模型推断)")
    parser.add_argument("--lr_hidden_size", type=int, default=None,
                        help="LatentReasoner hidden 维度 (0 = auto)")

    args = parser.parse_args()

    cfg = HLRConfig()

    if args.latent_reasoner_lr is not None: cfg.latent_reasoner_lr = args.latent_reasoner_lr
    if args.lr_num_layers is not None: cfg.lr_num_layers = args.lr_num_layers
    if args.lr_hidden_size is not None: cfg.lr_hidden_size = args.lr_hidden_size

    # 用户没显式传 --data → 用当前 --model 重跑 entropy profile, 覆盖 data_path,
    # 确保 profiled jsonl 和本次训练基座的 tokenizer/概率分布严格一致
    if args.data is None:
        cfg.auto_rebuild_data = True
        print("⚠ 未指定 --data, 训练前会用 --model 重新跑 entropy profile")
        print(f"   原始 chains: {cfg.raw_chains_path}")
        print(f"   输出 profiled: {cfg.data_path}")

    if args.model is not None: cfg.model_name = args.model
    if args.data is not None: cfg.data_path = args.data
    if args.compression_ratio is not None: cfg.latent_compression_ratio = args.compression_ratio
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.beta is not None: cfg.beta = args.beta
    if args.gamma is not None: cfg.gamma = args.gamma
    if args.lr is not None: cfg.lr = args.lr
    if args.epochs is not None: cfg.num_epochs = args.epochs
    if args.batch_size is not None: cfg.per_device_batch_size = args.batch_size
    if args.grad_accum is not None: cfg.gradient_accumulation_steps = args.grad_accum
    if args.zero_stage is not None: cfg.zero_stage = args.zero_stage
    if args.gradient_checkpointing: cfg.gradient_checkpointing = True
    if args.output_dir is not None: cfg.output_dir = args.output_dir
    if args.save_steps is not None: cfg.save_steps = args.save_steps

    train_hlr(cfg)


if __name__ == "__main__":
    main()
