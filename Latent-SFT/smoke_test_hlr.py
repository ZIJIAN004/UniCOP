"""
HLR Smoke Test —— 严格检查 Hierarchical Latent Reasoner 各模块。

分 7 个 stage，独立检测，任一失败立即报告 + traceback。

  [1] HLRConfig 字段 / 默认值 / 一致性
  [2] 小组件 (RoPE / GQA / SwiGLU / RMSNorm)
  [3] LatentReasoner forward + KV cache 一致性 (incremental == initial)
  [4] HLRDataset 加载 + collate_hlr
  [5] compute_hlr_loss forward + backward + 梯度覆盖
  [6] HLRInferenceEngine.generate
  [7] LatentReasoner state_dict 保存 / 重新加载

运行 (cwd = UniCOP 根目录):
  # 完整 (需要主模型 + profiled jsonl)
  python Latent-SFT/smoke_test_hlr.py --model "$BASE_MODEL" --data Latent-SFT/data/profiled_cvrp20.jsonl

  # 跳过需要主模型的阶段 (4-6)
  python Latent-SFT/smoke_test_hlr.py --no_main_model

退出码 0 = 所有 stage PASS; 非 0 = 至少一个 FAIL。
"""

import argparse
import sys
import traceback
from functools import partial
from pathlib import Path


# ====================================================================
# Stages
# ====================================================================


def stage_1_config():
    from config import HLRConfig

    cfg = HLRConfig()
    # 默认 lr_* 全 0 (auto-infer from main config, 自适应 R1-7B / Qwen3-4B / ...)
    # 真实 LR 架构由 build_latent_reasoner_from_main 推断, stage 3 验证形状
    auto_fields = ["lr_num_layers", "lr_hidden_size", "lr_num_heads",
                   "lr_num_kv_heads", "lr_head_dim", "lr_intermediate_size"]
    for field in auto_fields:
        got = getattr(cfg, field)
        assert got == 0, f"{field} = {got}, 期望默认 0 (auto-infer)"

    assert cfg.latent_compression_ratio == 4
    assert cfg.entropy_window == 3 and cfg.entropy_quantile == 0.5
    assert cfg.min_latent_steps == 3 and cfg.max_latent_steps == 8
    assert cfg.latent_cooldown == 24

    print(f"    config: lr_* 全 0 (auto-infer 自适应基座), compression_ratio={cfg.latent_compression_ratio}")
    print(f"    trigger: window={cfg.entropy_window}, q={cfg.entropy_quantile}, "
          f"min={cfg.min_latent_steps}, max={cfg.max_latent_steps}, cooldown={cfg.latent_cooldown}")


def stage_2_components():
    import torch
    from model import RotaryEmbedding, GQAAttention, SwiGLUFFN, SimpleRMSNorm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    device = {device}")

    # ── RoPE: start_pos 区分 ──
    rope = RotaryEmbedding(dim=128, max_position=128)
    cos_a, sin_a = rope(seq_len=4, device=device, dtype=torch.float32, start_pos=0)
    cos_b, sin_b = rope(seq_len=4, device=device, dtype=torch.float32, start_pos=2)
    assert cos_a.shape == (4, 128) and sin_a.shape == (4, 128)
    assert not torch.allclose(cos_a[0], cos_b[0]), "RoPE start_pos=0 vs 2 应该不同"
    print(f"    RoPE [4,128] OK, start_pos 区分位置")

    # ── RMSNorm: 输出 RMS ≈ 1 ──
    norm = SimpleRMSNorm(896).to(device)
    out = norm(torch.randn(2, 5, 896, device=device))
    rms = out.float().pow(2).mean(-1).sqrt().mean().item()
    assert 0.9 < rms < 1.1, f"RMSNorm 后 RMS={rms} 偏离 1"
    print(f"    SimpleRMSNorm shape OK, RMS={rms:.3f}")

    # ── SwiGLU ──
    ffn = SwiGLUFFN(hidden_size=896, intermediate_size=4736).to(device)
    out = ffn(torch.randn(2, 5, 896, device=device))
    assert out.shape == (2, 5, 896)
    print(f"    SwiGLU [2,5,896] OK")

    # ── GQA initial forward ──
    gqa = GQAAttention(hidden_size=896, num_heads=7, num_kv_heads=1, head_dim=128).to(device)
    x = torch.randn(2, 4, 896, device=device)
    cos, sin = rope(seq_len=4, device=device, dtype=torch.float32)
    out, (k_cache, v_cache) = gqa(x, cos, sin)
    assert out.shape == (2, 4, 896)
    assert k_cache.shape == (2, 1, 4, 128), f"K cache shape: {k_cache.shape}"
    assert v_cache.shape == (2, 1, 4, 128)
    print(f"    GQA initial: out [2,4,896] cache [2,1,4,128] (GQA 7:1, unrepeated)")

    # ── GQA incremental forward (KV 累积) ──
    x_new = torch.randn(2, 1, 896, device=device)
    cos2, sin2 = rope(seq_len=1, device=device, dtype=torch.float32, start_pos=4)
    out2, (k2, v2) = gqa(x_new, cos2, sin2, past_kv=(k_cache, v_cache))
    assert out2.shape == (2, 1, 896)
    assert k2.shape == (2, 1, 5, 128), f"incremental K cache: {k2.shape}, 期望 [2,1,5,128]"
    print(f"    GQA incremental: cache 4 → 5, mask 正确处理")


def stage_3_latent_reasoner():
    import torch
    from model import LatentReasoner

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = LatentReasoner(
        main_hidden_size=3584,
        num_main_layers=28,
        hidden_size=896,
        num_layers=7,
        num_heads=7,
        num_kv_heads=1,
        head_dim=128,
        intermediate_size=4736,
    ).to(device)

    n_params = sum(p.numel() for p in lr.parameters())
    print(f"    LatentReasoner 参数: {n_params/1e6:.2f}M")
    assert 100e6 < n_params < 130e6, f"参数量异常: {n_params/1e6:.2f}M (期望 100-130M)"

    h_in = torch.randn(1, 3584, device=device)

    # ── Initial forward k=5 ──
    lr.eval()
    with torch.no_grad():
        hiddens_5, kv_5 = lr(h_in, k=5)

    assert len(hiddens_5) == 7, f"layer_hiddens 长度: {len(hiddens_5)}"
    for L in range(7):
        assert hiddens_5[L].shape == (1, 5, 896), f"layer {L} hidden: {hiddens_5[L].shape}"
    assert len(kv_5) == 7
    for L in range(7):
        kk, vv = kv_5[L]
        assert kk.shape == (1, 1, 5, 128) and vv.shape == (1, 1, 5, 128)
    print(f"    Initial k=5: 7 hiddens [1,5,896] + KV [1,1,5,128]")

    # ── Incremental forward k=1, past=kv_5 ──
    with torch.no_grad():
        hiddens_inc, kv_inc = lr(h_in, k=1, past_kv=kv_5)
    for L in range(7):
        assert hiddens_inc[L].shape == (1, 1, 896)
        kk, vv = kv_inc[L]
        assert kk.shape == (1, 1, 6, 128), f"incremental layer {L} K: {kk.shape}"
    print(f"    Incremental k=1: KV 累积到 6")

    # ── 一致性: initial(k=6) vs initial(k=5) + incremental(k=1) ──
    with torch.no_grad():
        hiddens_6, _ = lr(h_in, k=6)
    max_diff = 0.0
    for L in range(7):
        full_last = hiddens_6[L][:, -1, :].float()
        inc_last = hiddens_inc[L][:, -1, :].float()
        diff = (full_last - inc_last).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"    一致性 max diff (initial vs incremental) = {max_diff:.2e}")
    threshold = 5e-3  # fp32 下应该接近 0; 浮点累计误差容忍
    assert max_diff < threshold, f"训练/推理 forward 不一致: diff={max_diff:.4f} > {threshold}"

    # ── project_for_main_layer: 同 LR 层 + 不同 layer_emb 应输出不同 hidden ──
    proj_5 = lr.project_for_main_layer(hiddens_5, main_layer_idx=5)    # L_lr=1
    proj_6 = lr.project_for_main_layer(hiddens_5, main_layer_idx=6)    # L_lr=1 (同 LR 层)
    proj_27 = lr.project_for_main_layer(hiddens_5, main_layer_idx=27)  # L_lr=6
    assert proj_5.shape == (1, 5, 3584)
    same_lr_diff = (proj_5 - proj_6).abs().mean().item()
    diff_lr_diff = (proj_5 - proj_27).abs().mean().item()
    assert same_lr_diff > 1e-4, f"同 LR 层不同 layer_emb 应有差异 (layer_emb 没监督?), 差异={same_lr_diff}"
    assert diff_lr_diff > same_lr_diff, "不同 LR 层应比同 LR 层差异更大"
    print(f"    project_for_main_layer: 同 LR 层 layer_emb 差异={same_lr_diff:.4f}, "
          f"跨 LR 层差异={diff_lr_diff:.4f}")


def stage_4_dataset(args):
    if not args.data or not Path(args.data).exists():
        print(f"    ⚠ SKIP: --data 未提供或文件不存在 ({args.data})")
        return None
    if not args.model:
        print(f"    ⚠ SKIP: --model 未提供 (需要 tokenizer)")
        return None

    from transformers import AutoTokenizer
    from data_utils import HLRDataset, collate_hlr

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if getattr(tokenizer, "add_bos_token", False):
        tokenizer.add_bos_token = False
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    dataset = HLRDataset(
        args.data, tokenizer,
        max_length=8192,
        latent_compression_ratio=4,
    )
    assert len(dataset) > 0, "HLRDataset 加载 0 条样本"

    # 找一条有 latent 段的样本作为检查样本 (不挑短样本, 用真实分布)
    sample = None
    n_with_latent = 0
    n_scanned = min(len(dataset), 50)
    for i in range(n_scanned):
        s = dataset[i]
        n_lat = sum(1 for seg in s.segments if seg.type == "latent")
        if n_lat > 0:
            n_with_latent += 1
            if sample is None:
                sample = s
    if sample is None:
        sample = dataset[0]
        print(f"    ⚠ 前 {n_scanned} 条都没 latent 段, 使用首条 (entropy 阈值偏严?)")
    else:
        print(f"    前 {n_scanned} 条中 {n_with_latent} 条有 latent 段, "
              f"用首个 ({sample.teacher_input_ids.size(0)} tokens)")

    n_exp = sum(1 for s in sample.segments if s.type == "explicit")
    n_lat = sum(1 for s in sample.segments if s.type == "latent")
    n_sol = sum(1 for s in sample.segments if s.type == "solution")
    print(f"    数据集: {len(dataset)} 条")
    print(f"    样本 segments: {len(sample.segments)} 段 "
          f"(explicit={n_exp}, latent={n_lat}, solution={n_sol})")
    assert n_sol >= 1, "样本必须有 solution 段"

    # collate
    collate_fn = partial(collate_hlr, pad_token_id=tokenizer.pad_token_id)
    batch = collate_fn([sample])
    expected_keys = {"teacher_input_ids", "teacher_attention_mask",
                     "teacher_labels", "prompt_ids", "segments"}
    actual_keys = set(batch.keys())
    missing = expected_keys - actual_keys
    assert not missing, f"collate 缺字段: {missing}"
    print(f"    collate_hlr 字段完整: {sorted(actual_keys)}")

    return (tokenizer, dataset, sample, batch)


def stage_5_loss(args, data_result):
    if args.no_main_model or data_result is None:
        print(f"    ⚠ SKIP: --no_main_model 或 stage 4 跳过了")
        return None

    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from model import build_latent_reasoner_from_main, compute_hlr_loss
    from config import HLRConfig

    tokenizer, dataset, sample, batch = data_result
    cfg = HLRConfig()

    # ── 镜像训练配置: GC + ZeRO-3 + CPU offload (与 submit_train_hlr.sh 一致) ──
    # 不挑短样本, 用真实长度分布; 必须靠 GC + ZeRO offload 撑住 24GB 单卡
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin

    # 4 GPU + ZeRO-3 切片 (无 CPU offload, 24GB×4=96GB 等效, Qwen3-4B 充裕)
    # 与 submit_train_hlr.sh 完全镜像
    ds_config = {
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": "auto",
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "gather_16bit_weights_on_model_save": True,
        },
    }
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config),
        mixed_precision="bf16",
    )
    print(f"    accelerator: num_processes={accelerator.num_processes}, "
          f"is_main={accelerator.is_main_process}, device={accelerator.device}")

    if accelerator.is_main_process:
        print(f"    加载主模型 {args.model} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # GC 与 use_cache 互斥, 训练时显式 False 抑制 attention 层的 warning
    model.config.use_cache = False
    # GC + input_require_grads 必须在 PEFT wrap 之前 (踩坑 #14)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if accelerator.is_main_process:
        print(f"    [smoke 镜像训练] gradient_checkpointing=True, use_reentrant=True, ZeRO-3 + CPU offload")

    peft_cfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.train()

    # Optimizer 必须传给 accelerate.prepare, ZeRO-3 才能切分 (即使 smoke 只跑 1 step)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    device = accelerator.device

    seq_len = batch["teacher_input_ids"].size(1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"    样本长度: {seq_len} tokens, GPU 剩余 free: {free_gb:.2f} GB (ZeRO-3 init 后)")

    # LatentReasoner 不进 ZeRO-3 (与 train_hlr 一致, 每 rank 独立完整副本 + 手动 all_reduce)
    unwrapped_main = accelerator.unwrap_model(model)
    latent_reasoner = build_latent_reasoner_from_main(unwrapped_main, cfg).to(device).to(torch.bfloat16)
    latent_reasoner.train()
    lr_n = sum(p.numel() for p in latent_reasoner.parameters())
    n_main_layers = unwrapped_main.config.num_hidden_layers  # Qwen3-4B=36, R1-7B=28
    if accelerator.is_main_process:
        print(f"    LatentReasoner: {lr_n/1e6:.2f}M params, n_main_layers={n_main_layers}")

    # batch tensors 移到 device
    batch_dev = {
        "teacher_input_ids": batch["teacher_input_ids"].to(device),
        "teacher_attention_mask": batch["teacher_attention_mask"].to(device),
        "teacher_labels": batch["teacher_labels"].to(device),
        "prompt_ids": batch["prompt_ids"].to(device),
        "segments": batch["segments"],  # list[HLRSegment], 内部 tensors 由 compute_hlr_loss 自处理
    }

    if accelerator.is_main_process:
        print(f"    forward + backward 单 sample ...")
    total, t, s, a = compute_hlr_loss(
        model=model,
        latent_reasoner=latent_reasoner,
        teacher_input_ids=batch_dev["teacher_input_ids"],
        teacher_attention_mask=batch_dev["teacher_attention_mask"],
        teacher_labels=batch_dev["teacher_labels"],
        prompt_ids=batch_dev["prompt_ids"],
        segments=batch_dev["segments"],
        alpha=1.0, beta=1.0, gamma=1.0,
    )

    for name, val in [("teacher_ce", t), ("student_ce", s), ("align_l1", a), ("total", total)]:
        assert torch.isfinite(val).all(), f"{name} 是 NaN/Inf: {val}"
        if accelerator.is_main_process:
            print(f"    {name} = {val.item():.4f}")

    # ZeRO-3 backward 必须走 accelerator (hook param gather)
    accelerator.backward(total)

    # ── 梯度覆盖检查 (LR 每 rank 完整副本, 可直接查; LoRA/main 在 ZeRO-3 下分片, 放宽) ──
    lr_total_params = sum(1 for p in latent_reasoner.parameters() if p.requires_grad)
    lr_has_grad = sum(
        1 for p in latent_reasoner.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    if accelerator.is_main_process:
        print(f"    LR 梯度: {lr_has_grad}/{lr_total_params} "
              f"({100*lr_has_grad/lr_total_params:.0f}%) 有非零梯度")
    assert lr_has_grad >= lr_total_params * 0.5, \
        f"LR 梯度覆盖率过低: {lr_has_grad}/{lr_total_params}"

    # layer_emb 行数 = 主模型层数 (Qwen3-4B=36, R1-7B=28); align loss 应覆盖每层
    layer_emb_grad = latent_reasoner.layer_emb.weight.grad
    assert layer_emb_grad is not None, "layer_emb.weight.grad is None"
    nonzero_rows = (layer_emb_grad.abs().sum(dim=-1) > 0).sum().item()
    if accelerator.is_main_process:
        print(f"    layer_emb 梯度: {nonzero_rows}/{n_main_layers} 行有非零梯度")
    assert nonzero_rows == n_main_layers, \
        f"layer_emb {nonzero_rows}/{n_main_layers} 行有梯度 (align loss 应覆盖每主模型层)"

    if accelerator.is_main_process:
        print(f"    ✓ stage 5 ZeRO-3 + GC + 双 forward 跑通")

    return (model, latent_reasoner, tokenizer)


def stage_6_inference(args, loss_result):
    if loss_result is None:
        print(f"    ⚠ SKIP: stage 5 没产出模型")
        return

    # 在 DeepSpeed ZeRO-3 / 分布式环境下跳过, 推理不需要 ZeRO-3, 而且
    # HLRInferenceEngine 用 device_map='auto' 加载主模型, 与 ZeRO-3 互斥.
    # 训练 smoke 通过后, 用户单独跑 inference.py 验证生成.
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            try:
                rank = dist.get_rank()
            except Exception:
                rank = 0
            if rank == 0:
                print(f"    ⚠ SKIP: 检测到 ZeRO-3 / distributed env, stage 6 不在此跑")
                print(f"       推理验证请单独运行 (zhuoyi 上):")
                print(f"         srun --gpus=1 python Latent-SFT/inference.py \\")
                print(f"           --model {args.model} \\")
                print(f"           --prompt '...'")
            dist.barrier()
            return
    except Exception:
        pass

    import torch
    from inference import HLRInferenceEngine
    from config import HLRConfig

    model, latent_reasoner, tokenizer = loss_result

    # 临时保存 latent_reasoner.pt 给 inference engine 加载
    tmp_dir = Path("./smoke_test_tmp")
    tmp_dir.mkdir(exist_ok=True)
    lr_path = tmp_dir / "latent_reasoner.pt"
    torch.save(latent_reasoner.state_dict(), lr_path)
    print(f"    临时保存 latent_reasoner.pt 到 {lr_path}")

    cfg = HLRConfig()
    print(f"    初始化 HLRInferenceEngine (会重新加载主模型) ...")
    engine = HLRInferenceEngine(
        model_path=args.model,
        latent_reasoner_path=str(lr_path),
        cfg=cfg,
        entropy_window=3,
        max_latent_steps=8,
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Briefly say hello."}],
        tokenize=False, add_generation_prompt=True,
    )

    out_text, info = engine.generate(prompt, max_new_tokens=32, temperature=0.0)
    assert isinstance(out_text, str)
    assert info["num_tokens"] > 0
    print(f"    generate: {info['num_tokens']} tokens, "
          f"{info['latent_steps']} latent steps, "
          f"entropy_history len={len(info['entropy_history'])}")
    print(f"    输出 (前 80 字符): {out_text[:80]!r}")

    # 同步其他 rank
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def stage_7_checkpoint():
    import torch
    from model import LatentReasoner

    lr1 = LatentReasoner(
        main_hidden_size=3584, num_main_layers=28,
        hidden_size=896, num_layers=7, num_heads=7,
        num_kv_heads=1, head_dim=128, intermediate_size=4736,
    )
    lr2 = LatentReasoner(
        main_hidden_size=3584, num_main_layers=28,
        hidden_size=896, num_layers=7, num_heads=7,
        num_kv_heads=1, head_dim=128, intermediate_size=4736,
    )

    state = lr1.state_dict()
    lr2.load_state_dict(state)

    # 参数应该完全相同
    mismatches = []
    for (n1, p1), (n2, p2) in zip(lr1.named_parameters(), lr2.named_parameters()):
        assert n1 == n2, f"param 名字不匹配: {n1} vs {n2}"
        if not torch.allclose(p1, p2):
            mismatches.append(n1)
    assert not mismatches, f"加载后参数不一致: {mismatches}"
    print(f"    state_dict 保存 / 重新加载 OK ({len(state)} 个键)")


# ====================================================================
# Driver
# ====================================================================


def main():
    parser = argparse.ArgumentParser(description="HLR Smoke Test")
    parser.add_argument("--model", type=str, default=None, help="主模型路径")
    parser.add_argument("--data", type=str, default=None, help="profiled jsonl 路径")
    parser.add_argument("--no_main_model", action="store_true",
                        help="跳过需要主模型的阶段 (4-6)")
    args = parser.parse_args()

    print("=" * 72)
    print("  HLR Smoke Test")
    print("=" * 72)
    if args.no_main_model:
        print("  模式: 跳过主模型相关阶段 (stage 4-6)")
    else:
        print(f"  model: {args.model}")
        print(f"  data:  {args.data}")
    print()

    results: dict[str, str] = {}

    def _run(name, fn):
        print(f"\n{name}")
        try:
            return fn(), "PASS"
        except Exception as e:
            print(f"    ✗ FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None, f"FAIL ({type(e).__name__})"

    # Stage 1
    _, results["[1] HLRConfig"] = _run("[1] HLRConfig 字段", stage_1_config)

    # Stage 2
    _, results["[2] 小组件"] = _run("[2] 小组件 (RoPE/GQA/SwiGLU/RMSNorm)", stage_2_components)

    # Stage 3
    _, results["[3] LatentReasoner"] = _run(
        "[3] LatentReasoner forward + KV consistency", stage_3_latent_reasoner
    )

    # Stage 4 (依赖 args)
    data_result, status = _run("[4] HLRDataset + collate", lambda: stage_4_dataset(args))
    results["[4] HLRDataset"] = status

    # Stage 5 (依赖 stage 4 输出)
    loss_result, status = _run(
        "[5] compute_hlr_loss forward+backward",
        lambda: stage_5_loss(args, data_result),
    )
    results["[5] compute_hlr_loss"] = status

    # Stage 6 (依赖 stage 5)
    _, results["[6] HLRInferenceEngine"] = _run(
        "[6] HLRInferenceEngine.generate", lambda: stage_6_inference(args, loss_result)
    )

    # Stage 7
    _, results["[7] checkpoint"] = _run(
        "[7] LatentReasoner state_dict 保存/加载", stage_7_checkpoint
    )

    # 汇总
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    all_pass = True
    for name, status in results.items():
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} {name:35s} {status}")
        if status != "PASS" and "SKIP" not in status:
            all_pass = False

    print()
    if all_pass:
        print("  Smoke test ALL PASS ✓")
        sys.exit(0)
    else:
        print("  Smoke test HAS FAILURES ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
