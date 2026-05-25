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
import time
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, set_seed
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from hlr_config import HLRConfig
from data_utils import HLRDataset, collate_hlr
from model import (
    build_latent_reasoner_from_main,
    compute_hlr_loss,
    hlr_timer,
    hlr_timing_add,
    hlr_timing_report,
    hlr_timing_tick_microstep,
)


# ── Rank-level 时间戳打点 (用于定位 ZeRO-3 hang 发生在哪一步) ──────────────
# 用 env RANK (accelerate launch 自动设, 4 卡场景 0..3),
# 比 accelerator.process_index 早可用 (后者要等 Accelerator() 构造完).
# 每条打点强制 flush, 让日志立刻可见即使后续 hang.
_HLR_T0 = time.time()
_HLR_RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
# 默认关 (生产干净), export HLR_DEBUG=1 开启全部 rank-level 诊断 stamp.
# 再遇 ZeRO-3 hang 时直接 HLR_DEBUG=1 重跑即可定位.
_HLR_DEBUG = os.environ.get("HLR_DEBUG", "0") == "1"


def _stamp(msg: str) -> None:
    if not _HLR_DEBUG:
        return
    elapsed = time.time() - _HLR_T0
    print(f"[STAMP rank={_HLR_RANK} +{elapsed:6.1f}s] {msg}", flush=True)


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
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True,
            # 优化器状态 offload 到 CPU: 释放 GPU 显存, 缓解 HLR 双 forward +
            # output_hidden_states 的显存压力 (cache flush 抖动). 与 UniCOP-Distill SFT 一致.
            # 注: 只 offload DeepSpeed 管理的主(LoRA)优化器; LatentReasoner 的 lr_optimizer
            #     在 DeepSpeed 外, 仍在 GPU (137M, 不大).
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        }

    # ── ZeRO 通信剖分 (export HLR_DS_PROFILE=1 开启) ──
    # comms_logger(prof_all): 在 comm wrapper 层记录每类 collective 的 调用数/数据量/总耗时,
    #   走 socket(无 NVLink) 也能正确计时 (不依赖 GPU kernel, 故不会漏掉 socket 等待).
    #   all_gather = ZeRO-3 参数按层聚合 (fwd 各层 + bwd GC 重算各层); reduce_scatter = 梯度规约.
    #   训练循环里定期调 deepspeed.comm.log_summary() 打印汇总 (见 train_hlr 主循环).
    # ⚠ 不能开 wall_clock_breakdown: 它给 engine 装 fwd/bwd_microstep 计时器, 假设严格
    #   1 forward : 1 backward. 本项目一次 backward 前跑两次 forward (teacher + student)
    #   → 注册两个 backward-prologue 钩子 → bwd_microstep 被 start 两次 → AssertionError
    #   "bwd_microstep timer has already been started" 崩溃 (DeepSpeed#617 自定义循环已知坑).
    #   fwd/bwd 的 wall-clock 拆分改由 hlr_timer (teacher_fwd/student_fwd/backward) 提供.
    if os.environ.get("HLR_DS_PROFILE", "0") == "1":
        base["comms_logger"] = {
            "enabled": True,
            "verbose": False,
            "prof_all": True,
            "debug": False,
        }

    return base


def train_hlr(cfg: HLRConfig):
    _stamp("enter train_hlr()")
    ds_config = make_deepspeed_config(cfg.zero_stage)
    ds_plugin = None
    if ds_config is not None:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
    _stamp("before Accelerator()")
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=ds_plugin,
    )
    _stamp(f"after Accelerator()  process_index={accelerator.process_index}/{accelerator.num_processes}")

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
    _stamp("before from_pretrained (main model)")
    # attn_implementation=sdpa: PyTorch 内置高效注意力 (自动选 flash/mem-efficient kernel,
    # 不 materialize O(T²) 注意力矩阵), 不依赖外部 flash-attn 包. 不支持则回退 HF 默认.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        _attn_impl = "sdpa"
    except (ValueError, RuntimeError):
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _attn_impl = "default"
    if accelerator.is_main_process:
        print(f"  attn_implementation = {_attn_impl}")
    _stamp("after from_pretrained (main model)")

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # ── 数据预处理要求 ──
    # entropy profiling 必须在 sbatch 脚本里作为独立 Step 0 用单 GPU 跑
    # (不能在这里 inline 做: ZeRO-3 init context 已经 partition embedding.weight 成
    # 1D, model.forward 会 'weight' must be 2-D 报错).
    # 见 submit_train_eval_hlr.sh / submit_smoke_hlr.sh 的 Step 0 模板.
    import os
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"profiled jsonl 不存在: {cfg.data_path}\n"
            f"请先用单 GPU 跑:\n"
            f"  CUDA_VISIBLE_DEVICES=0 python Latent-SFT/entropy_profile.py \\\n"
            f"    --model {cfg.model_name} \\\n"
            f"    --data {cfg.raw_chains_path} \\\n"
            f"    --output {cfg.data_path}\n"
            f"(sbatch 脚本 submit_train_eval_hlr.sh 的 Step 0 已自动做这件事)"
        )

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
        _stamp("before get_peft_model")
        model = get_peft_model(model, peft_config)
        _stamp("after get_peft_model")
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    # LatentReasoner 必须在 accelerator.prepare 之后创建 (跟 smoke_test_hlr.py 一致):
    # ZeRO-3 init context 在 Accelerator(deepspeed_plugin=...) 创建后激活,
    # 在这个 context 内创建的所有 nn.Module 参数都被 zero.Init 自动 partition.
    # LR 如果在 prepare 前创建, 参数会被 partition 但又不在 DeepSpeed engine 管理下,
    # forward 时 ZeRO-3 hook 触发 partial gather collective → 跟 main model 的
    # collective 顺序乱 → 死锁 (rank 间 LR forward 次数不同更易暴露).
    # prepare(model) 会消耗 zero.Init context, 之后创建的 module 不被 partition.
    # 见 smoke_test_hlr.py:332 设计.

    # ── 数据 ──
    # Smoke: limit > 0 时 HLRDataset 读到 N 条就停, 16min → 30sec 大幅缩短诊断 cycle
    _limit = getattr(cfg, "dataset_limit", 0)
    _stamp(f"before HLRDataset() (limit={_limit if _limit else 'full'})")
    dataset = HLRDataset(
        cfg.data_path, tokenizer,
        max_length=cfg.max_length,
        latent_compression_ratio=cfg.latent_compression_ratio,
        filter_problems=cfg.filter_problems,
        filter_sizes=cfg.filter_sizes,
        limit=_limit,
    )
    _stamp(f"after HLRDataset()  n={len(dataset)}")

    collate_fn = partial(collate_hlr, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,   # Phase 1 强制 1
        shuffle=True,
        collate_fn=collate_fn,
        # num_workers=0: batch_size=1 collate 极轻量, worker 并行无收益;
        # num_workers=2 在 NFS (TMPDIR=/homes/zhuoyi/tmp) 上退出时 pymp 临时目录
        # 被 fd 占用 rmtree 失败, 刷一屏 OSError [Errno 16] Device busy (无害,
        # 在 checkpoint 保存之后, 但很吵). num_workers=0 根治.
        num_workers=0,
        pin_memory=True,
    )

    # ── 主 Optimizer (LR optimizer 推迟到 LR 创建之后) ──
    main_optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ── Accelerate prepare (先 prepare 主模型 + opt + dataloader, 不含 LR) ──
    # 注意: LR 必须等 prepare 之后再创建, 避免被 zero.Init partition (见上面注释).
    _stamp("before accelerator.prepare(model, opt, dataloader)  ← ZeRO-3 init")
    model, main_optimizer, dataloader = accelerator.prepare(
        model, main_optimizer, dataloader
    )
    _stamp("after accelerator.prepare(model, opt, dataloader)  ← ZeRO-3 init done")

    # ── 现在 zero.Init context 已退出, 安全创建 LR (每 rank 完整副本, 不被 partition) ──
    _stamp("before build_latent_reasoner_from_main (POST-prepare, no zero.Init)")
    unwrapped_main = accelerator.unwrap_model(model)
    latent_reasoner = build_latent_reasoner_from_main(unwrapped_main, cfg)
    latent_reasoner = latent_reasoner.to(accelerator.device).to(torch.bfloat16)
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in latent_reasoner.parameters())
        print(f"  LatentReasoner 参数量: {n_params/1e6:.2f}M")
    _stamp(f"after latent_reasoner created + .to(bf16)")

    # LR optimizer 在 LR 创建后, 不进 accelerator.prepare (LR 完整副本不 sharded)
    lr_optimizer = AdamW(
        latent_reasoner.parameters(),
        lr=cfg.latent_reasoner_lr,
        weight_decay=cfg.weight_decay,
    )

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
    _stamp("before training loop (entering first epoch)")
    global_step = 0
    # 进度条: total = 实际 optimizer-step 数 (每 epoch floor(len/ga) 次 sync boundary).
    # 主进程独占 (4 卡 disable 其余, 否则 4 条 bar 互相刷屏);
    # SLURM --output 非 tty, mininterval=10 限流, 避免每步一行 \r 撑爆日志.
    steps_per_epoch = len(dataloader) // cfg.gradient_accumulation_steps
    progress_bar = tqdm(
        total=steps_per_epoch * cfg.num_epochs,
        desc="HLR train",
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
        mininterval=10.0,
    )
    for epoch in range(cfg.num_epochs):
        model.train()
        latent_reasoner.train()
        epoch_loss = 0.0

        # GA boundary 手动追踪 (不再用 accelerator.accumulate context):
        # accelerator.accumulate 内部调 model.no_sync(), 跟 DeepSpeed ZeRO-3
        # 不兼容 (Accelerate issue #3481), 导致 compute_hlr_loss forward 死锁.
        # DeepSpeed 自己按 ds_config 的 gradient_accumulation_steps 处理累积:
        #   - 前 GA-1 个 step 的 backward 累积梯度, optimizer.step() 是 no-op
        #   - 第 GA 个 step 真正 all-reduce + optimizer.step()
        # main_optimizer/scheduler 走 DeepSpeed wrap, 始终调用即可;
        # lr_optimizer/scheduler 不在 DeepSpeed 内, 用 _is_sync 手动门控.
        _ga = cfg.gradient_accumulation_steps
        _t_data_prev = time.perf_counter()
        for step, batch in enumerate(dataloader):
            hlr_timing_add("data_fetch", time.perf_counter() - _t_data_prev)
            hlr_timing_tick_microstep()
            _is_sync = ((step + 1) % _ga == 0)
            # 前 3 步打细 stamp (定位 hang 用)
            _DBG = step < 3
            if _DBG:
                _stamp(f"step {step}: batch fetched "
                       f"(teacher_len={batch['teacher_input_ids'].size(1)}) "
                       f"is_sync={_is_sync}")
                _stamp(f"step {step}: BEFORE compute_hlr_loss")
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
            if _DBG:
                _stamp(f"step {step}: AFTER compute_hlr_loss "
                       f"(t={t_loss.item():.3f} s={s_loss.item():.3f} "
                       f"a={a_loss.item():.3f})")

            with hlr_timer("backward"):
                accelerator.backward(total_loss)
            if _DBG:
                _stamp(f"step {step}: AFTER accelerator.backward")

            with hlr_timer("optim"):
                if _is_sync:
                    # 多卡 LatentReasoner 梯度手动 all-reduce (LR 没进 DeepSpeed sharding).
                    # 展平成一个 buffer 做 *一次* all-reduce, 而非每个 param 张量一次:
                    # LR 有几十个 param 张量, socket 每次 collective 延迟高, 合并 N→1 省时间.
                    if accelerator.num_processes > 1:
                        _lr_grads = [p.grad for p in latent_reasoner.parameters()
                                     if p.grad is not None]
                        if _lr_grads:
                            _flat = torch._utils._flatten_dense_tensors(_lr_grads)
                            torch.distributed.all_reduce(_flat, op=torch.distributed.ReduceOp.AVG)
                            for _g, _synced in zip(
                                _lr_grads, torch._utils._unflatten_dense_tensors(_flat, _lr_grads)
                            ):
                                _g.copy_(_synced)

                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(
                        latent_reasoner.parameters(), cfg.max_grad_norm
                    )

                    lr_optimizer.step()
                    lr_scheduler.step()
                    lr_optimizer.zero_grad()

                # main_optimizer 走 DeepSpeed wrap, 内部按 GA boundary 自动跳过非 boundary
                main_optimizer.step()
                main_scheduler.step()
                main_optimizer.zero_grad()
                if _DBG and _is_sync:
                    _stamp(f"step {step}: AFTER optimizer.step (sync boundary)")

            _loss_val = total_loss.item()
            epoch_loss += _loss_val

            if _is_sync:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{_loss_val:.3f}")

                # 所有 rank 都报分段计时 (对比各 rank barrier_wait 看负载失衡)
                if global_step % cfg.logging_steps == 0:
                    hlr_timing_report(accelerator.process_index, global_step)
                    # ZeRO 通信汇总 (all_gather/reduce_scatter 的 调用数/数据量/总耗时).
                    # 所有 rank 在同一 global_step 边界一起调, 避免 collective 不齐死锁.
                    if os.environ.get("HLR_DS_PROFILE", "0") == "1":
                        try:
                            import deepspeed.comm as ds_comm
                            ds_comm.log_summary()
                        except Exception as _e:
                            if accelerator.is_main_process:
                                print(f"[DS_PROFILE] log_summary skipped: {_e}", flush=True)

                if global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / (step + 1)
                    main_lr = main_scheduler.get_last_lr()[0]
                    lr_lr = lr_scheduler.get_last_lr()[0]
                    tqdm.write(
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

            # 本 micro-step 结束, 重置 data_fetch 计时锚点 (测下一步 dataloader 等待)
            _t_data_prev = time.perf_counter()

    progress_bar.close()

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
    parser.add_argument("--logging_steps", type=int, default=None)

    parser.add_argument("--latent_reasoner_lr", type=float, default=None,
                        help="LatentReasoner 的 lr")
    parser.add_argument("--lr_num_layers", type=int, default=None,
                        help="LatentReasoner 层数 (0 = auto 从主模型推断)")
    parser.add_argument("--lr_hidden_size", type=int, default=None,
                        help="LatentReasoner hidden 维度 (0 = auto)")
    # Smoke 用: 限制 dataset 大小让训练 5 min 内跑完几步, 完整复现 train.py 所有
    # 训练路径 (scheduler/lr_optimizer/all_reduce/GA/accumulate context/ZeRO-3),
    # 唯一区别是 dataset 截断到前 N 条. 复现 hang 用 --limit 32 (GA=8 × 4 GPU
    # = effective batch 32, 触发一次 sync_gradients 路径).
    parser.add_argument("--limit", type=int, default=0,
                        help="只用 dataset 前 N 条 (0 = 全量). smoke 复现 hang 用 32.")

    args = parser.parse_args()

    cfg = HLRConfig()

    if args.latent_reasoner_lr is not None: cfg.latent_reasoner_lr = args.latent_reasoner_lr
    if args.lr_num_layers is not None: cfg.lr_num_layers = args.lr_num_layers
    if args.lr_hidden_size is not None: cfg.lr_hidden_size = args.lr_hidden_size

    # entropy profile 必须事先用单 GPU 跑好 (ZeRO-3 init 与 model.forward 不兼容),
    # 见 submit_train_eval_hlr.sh 的 Step 0. 这里只校验 profiled jsonl 是否存在.
    if args.data is None:
        print("⚠ 未指定 --data, 将使用 HLRConfig 默认: " + cfg.data_path)
        print("  (如该文件不存在, train_hlr() 启动后会报错并给出 entropy_profile 命令)")

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
    if args.logging_steps is not None: cfg.logging_steps = args.logging_steps
    # 把 --limit 透传到 cfg, train_hlr 内部截断 dataset
    cfg.dataset_limit = args.limit if args.limit > 0 else 0

    train_hlr(cfg)


if __name__ == "__main__":
    main()
