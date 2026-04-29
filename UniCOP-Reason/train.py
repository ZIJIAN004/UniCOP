"""
主训练脚本：用 GRPO 微调 LLM 求解组合优化问题。

单卡运行：
    python train.py --problem tsptw --problem_size 10
    python train.py --problem tsp cvrp tsptw tspdl --num_gpus 1

多卡运行（需先安装 accelerate，并 accelerate config 完成环境配置）：
    accelerate launch --num_processes 4 train.py --problem tsptw --num_gpus 4 --zero_stage 3
    accelerate launch --num_processes 4 train.py --problem tsp cvrp tsptw tspdl --num_gpus 4 --zero_stage 3

注意：--num_gpus 必须与 --num_processes 保持一致，脚本不自动拉起多进程。
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.utils.checkpoint as torch_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from peft import LoraConfig, get_peft_model

# ── 诊断: 开启 checkpoint debug 模式,出错时给出更详细的 tensor 信息 ────────
# 包括: tensor 来源代码位置、变量名、forward call 路径
# 性能损耗: 几乎为零 (只在元数据对比那一步多记录信息)
torch_checkpoint.set_checkpoint_debug_enabled(True)

from config import config
from data.generate import build_dataset, build_mixed_dataset
from problems import get_problem, SUPPORTED_PROBLEMS
from pomo_prm import POMOPRM
from grpo_prm_trainer import GRPOPRMTrainer
from terminal_reward import compute_terminal_reward


# ── DeepSpeed ZeRO 配置生成 ──────────────────────────────────────────────────

def make_deepspeed_config(zero_stage: int) -> dict | None:
    """
    根据 zero_stage 生成 DeepSpeed 配置字典，传给 GRPOConfig(deepspeed=...)。
    返回 None 表示不启用 DeepSpeed（单卡模式）。

    ZeRO-2：优化器状态 + 梯度分片，模型权重不拆分，通信开销低，适合 1.5B 多卡。
    ZeRO-3：完全分片（权重 + 梯度 + 优化器），每卡只保存 1/N 权重，7B 多卡必须用。
    """
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
            "stage3_param_persistence_threshold": 1_000_000,
            "stage3_max_live_parameters":     1e9,
            "stage3_max_reuse_distance":      1e9,
            "gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {
                "device":     "cpu",
                "pin_memory": True,
            },
        }

    return base


# ── 问题实例缓存 ─────────────────────────────────────────────────────────────
# get_problem() 每次调用都构造新对象，训练循环中频繁调用时产生无意义的重复开销。
# 问题类无内部状态（rng 由外部传入），同一类型复用同一实例完全安全。

_PROBLEM_CACHE: dict = {}

def _get_problem(problem_type: str):
    if problem_type not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[problem_type] = get_problem(problem_type)
    return _PROBLEM_CACHE[problem_type]


# ── Placeholder reward（GRPOTrainer 强制要求至少一个 reward_func） ──────────
# 实际 reward 由 GRPOPRMTrainer 内部用 terminal_reward + POMO PRM 接管。
# 此函数仅为满足 trl 接口签名，返回值不被使用。

def _placeholder_reward_fn(completions, **kwargs):
    return [0.0] * len(completions)


# ── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",      type=str, nargs="+", default=[config.problem_type],
                        choices=SUPPORTED_PROBLEMS,
                        help="一个或多个问题类型，多个时自动切换为混合训练模式")
    parser.add_argument("--problem_size", type=int, default=config.problem_size)
    parser.add_argument("--model",        type=str, default=config.model_name)
    parser.add_argument("--num_train",    type=int, default=config.num_train)
    parser.add_argument("--output_dir",   type=str, default=config.output_dir)
    parser.add_argument("--no_lora",      action="store_true")
    parser.add_argument("--num_gpus",     type=int, default=config.num_gpus,
                        help="使用的 GPU 数量，需与 accelerate launch --num_processes 一致")
    parser.add_argument("--zero_stage",   type=int, default=config.zero_stage,
                        choices=[0, 2, 3],
                        help="DeepSpeed ZeRO stage：0=关闭 | 2=梯度分片(1.5B) | 3=完全分片(7B)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="开启梯度重计算节省显存，ZeRO-3 + 7B 时建议加上")
    parser.add_argument("--no_repeat_ngram_size", type=int,
                        default=config.no_repeat_ngram_size,
                        help="GRPO 生成时的 n-gram 硬禁，0=关闭")
    parser.add_argument("--clip_epsilon_low", type=float,
                        default=config.clip_epsilon_low,
                        help="GRPO ratio clip 下界 ε_low (默认 0.20)")
    parser.add_argument("--clip_epsilon_high", type=float,
                        default=config.clip_epsilon_high,
                        help="GRPO ratio clip 上界 ε_high，> ε_low 启用 DAPO Clip-Higher (默认 0.28)")
    # POMO PRM (always enabled, vanilla reward 路径已删除)
    parser.add_argument("--pomo_ckpt_dir", type=str, default=config.pomo_ckpt_dir,
                        help="POMO checkpoint 根目录，子目录: {type}_n{size}/MODEL_BEST.pt")
    parser.add_argument("--pomo_baseline_dir", type=str, default=config.pomo_baseline_dir,
                        help="POMO-Baseline 项目根目录")
    parser.add_argument("--pipd_ckpt_dir", type=str, default=config.pipd_ckpt_dir,
                        help="PIP-D TSPTW checkpoint 根目录 (POMO+PIP/pretrained/TSPTW)")
    parser.add_argument("--pipd_dir", type=str, default=config.pipd_dir,
                        help="PIP-D 代码目录 (POMO+PIP),用于 sys.path 注入")
    parser.add_argument("--vllm_server_host", type=str, default="localhost",
                        help="vLLM server 主机，server 模式 rollout 加速用")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                        help="vLLM server 端口，需与 trl vllm-serve 启动端口一致")
    args = parser.parse_args()

    problem_types = args.problem           # 始终为 list[str]
    is_mixed      = len(problem_types) > 1
    run_tag       = "mixed" if is_mixed else problem_types[0]

    config.problem_type  = problem_types[0]   # 保持向后兼容（evaluate 等处读单值）
    config.problem_size  = args.problem_size
    config.model_name    = args.model
    config.num_train     = args.num_train
    config.output_dir    = os.path.join(args.output_dir, f"{run_tag}_n{args.problem_size}")
    if args.no_lora:
        config.use_lora = False
    config.num_gpus               = args.num_gpus
    config.zero_stage             = args.zero_stage
    config.gradient_checkpointing = args.gradient_checkpointing or config.gradient_checkpointing
    config.no_repeat_ngram_size   = args.no_repeat_ngram_size
    config.clip_epsilon_low       = args.clip_epsilon_low
    config.clip_epsilon_high      = args.clip_epsilon_high
    config.pomo_ckpt_dir          = args.pomo_ckpt_dir
    config.pomo_baseline_dir      = args.pomo_baseline_dir
    config.pipd_ckpt_dir          = args.pipd_ckpt_dir
    config.pipd_dir               = args.pipd_dir

    # ── 早期检查：所有问题类型必须在 POMO PRM 支持列表内 ───────────
    unsupported = [pt for pt in problem_types if pt not in POMOPRM.SUPPORTED]
    if unsupported:
        raise ValueError(
            f"以下问题类型不在 POMO PRM 支持列表 {sorted(POMOPRM.SUPPORTED)}: "
            f"{unsupported}。vanilla reward 模式已删除，请仅使用 POMO 支持的类型。"
        )

    print(f"问题类型:  {problem_types}{'（混合模式）' if is_mixed else ''}")
    print(f"问题规模:  n={config.problem_size}")
    print(f"模型:      {config.model_name}")
    print(f"训练样本:  {config.num_train}")
    print(f"输出路径:  {config.output_dir}")
    print(f"GPU 数量:  {config.num_gpus}  ZeRO stage: {config.zero_stage}"
          f"  梯度重计算: {config.gradient_checkpointing}")
    print(f"no_repeat_ngram_size: {config.no_repeat_ngram_size}"
          f"{'（已关闭）' if config.no_repeat_ngram_size == 0 else ''}")
    _clip_mode = ("asymmetric (Clip-Higher)"
                  if config.clip_epsilon_high > config.clip_epsilon_low
                  else "symmetric")
    print(f"Ratio clip:  ε_low={config.clip_epsilon_low}, "
          f"ε_high={config.clip_epsilon_high}  [{_clip_mode}]")

    # ── 加载模型 ────────────────────────────────────────────────────────
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    # ⚠️ pad_token 处理 (对齐 SFT 的 safe 逻辑):
    # GRPO 下 completion_mask 基于 attention_mask 识别,SFT 那种"EOS 被 mask
    # 不学"的致命 bug 在 GRPO 里 不直接发生。但仍然避免 pad=eos,
    # 保证 attention_mask/padding 逻辑清晰,和 SFT 阶段一致。
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        _pad_set = False
        for cand in ["<｜▁pad▁｜>", "<|▁pad▁|>", "<|PAD_TOKEN|>"]:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
                tokenizer.pad_token = cand
                _pad_set = True
                break
        if not _pad_set:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # ⚠️ 不要用 tokenizer.model_max_length 限制长度。
    # 原配置 `tokenizer.model_max_length = config.max_prompt_length (=768)` 会触发
    # "Token indices sequence length is longer than the specified maximum sequence length"
    # 警告 (超过 768 就 warn),但实际 prompt+completion 总长常到 5000 token。
    # 正确做法: 保留 tokenizer 从 model config 读到的上限 (Qwen2.5 是 131K),
    # prompt 长度上限由 GRPOConfig(max_prompt_length=...) 独立控制,见下面 grpo_config。

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 如果前面 add_special_tokens 新加了 pad token (vocab size +1),
    # 必须在 LoRA wrap 之前 resize embedding,否则 pad_token_id 会超出 embedding 范围
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        print(f"  resize_token_embeddings {model.get_input_embeddings().num_embeddings} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if config.use_lora:
        print(f"启用 LoRA (rank={config.lora_rank})")
        lora_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # gradient_checkpointing + LoRA + ZeRO-3 三件套必需:
        # base 是 frozen 的, embedding 输出默认无 grad, checkpoint 在 backward
        # 时算不出反传路径,触发 "Recomputed values shape [0]" 错。
        # 这一行强制让 input embeddings 的输出 requires_grad=True,
        # 让 LoRA 反向传播能正确回到 input。
        if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            print("✓ enable_input_require_grads() 已开启 (LoRA + gc 必需)")

        # ── 诊断: 显式调 gradient_checkpointing_enable,确保 use_reentrant 设置落实 ─
        # TRL 1.1 内部也会调一次,这里我们先调,后面 trainer 看到已经开了就不再覆盖
        # 关键: 顺序必须是先 gc_enable,再 input_require_grads
        if config.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": True}
                )
                print("✓ gradient_checkpointing_enable() with use_reentrant=True")
                # 再 enable 一次 input_require_grads 以防顺序不对
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception as e:
                print(f"⚠️ gradient_checkpointing_enable failed: {e}")

        # ── 诊断: 打印前 5 个 frozen base param 的状态 ─────────────────────────
        print("\n=== 模型加载后的 frozen param 抽样 (DeepSpeed wrap 之前) ===")
        frozen_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad and frozen_count < 5:
                print(f"  [frozen] {name}: shape={tuple(param.shape)}, "
                      f"numel={param.numel()}, dtype={param.dtype}")
                frozen_count += 1
        print(f"  (注: ZeRO-3 wrap 后 < stage3_param_persistence_threshold "
              f"的小 param 应保持完整不分片)\n")

    # ── 数据集 ──────────────────────────────────────────────────────────
    print("\n生成训练数据...")
    if is_mixed:
        num_each = config.num_train // len(problem_types)
        train_dataset = build_mixed_dataset(
            problem_types=problem_types,
            num_samples_each=num_each,
            seed=config.data_seed,
            n=config.problem_size,
        )
    else:
        train_dataset = build_dataset(
            problem_type=problem_types[0],
            num_samples=config.num_train,
            seed=config.data_seed,
            n=config.problem_size,
        )
    print(f"训练集大小: {len(train_dataset)}")

    # ── GRPO 配置 ────────────────────────────────────────────────────────
    ds_config = make_deepspeed_config(config.zero_stage)

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        beta=config.kl_coef,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="no",
        save_strategy="steps",
        report_to="wandb" if config.use_wandb else "none",
        bf16=True,
        remove_unused_columns=False,
        deepspeed=ds_config,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        epsilon=config.clip_epsilon_low,
        epsilon_high=config.clip_epsilon_high,
        max_prompt_length=config.max_prompt_length,
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
    )

    # ── 初始化 POMO PRM + 训练 ─────────────────────────────────────────
    pomo_prm = POMOPRM(
        pomo_ckpt_dir=config.pomo_ckpt_dir,
        pomo_baseline_dir=config.pomo_baseline_dir,
        device=config.pomo_device,
        pipd_ckpt_dir=config.pipd_ckpt_dir or None,
        pipd_dir=config.pipd_dir or None,
    )
    pomo_prm.check_checkpoints(problem_types, [config.problem_size])

    trainer = GRPOPRMTrainer(
        pomo_prm=pomo_prm,
        problem_types=problem_types,
        model=model,
        reward_funcs=[_placeholder_reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\n开始 GRPO 训练...")

    # ── 诊断: 在 trainer.train() 之前打印 DeepSpeed 实际生效的 ZeRO 配置 ────
    # 让 trainer 先做完 DeepSpeed init, 但不真的开始训练
    # 这里通过 _wrap_model 触发 (transformers 内部入口)
    try:
        if hasattr(trainer, "deepspeed_plugin") and trainer.deepspeed_plugin is not None:
            print("\n=== DeepSpeed 实际生效配置 (zero_optimization 部分) ===")
            ds_cfg = trainer.deepspeed_plugin.deepspeed_config or {}
            zo = ds_cfg.get("zero_optimization", {})
            for k in ("stage", "stage3_param_persistence_threshold",
                      "stage3_max_live_parameters", "stage3_max_reuse_distance",
                      "stage3_prefetch_bucket_size", "reduce_bucket_size"):
                print(f"  {k}: {zo.get(k, '(not set)')}")
            print()
    except Exception as e:
        print(f"⚠️ 读取 DeepSpeed 配置失败: {e}")

    trainer.train()

    # ── 保存 ─────────────────────────────────────────────────────────────
    # trainer.save_model 内部已有 rank 守卫（ZeRO-3 gather + 主 rank 写）；
    # tokenizer.save_pretrained 没有，多 rank 同时写会竞争，必须显式守卫。
    save_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(save_path)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(save_path)
    trainer.accelerator.wait_for_everyone()
    print(f"\n模型已保存到: {save_path}")

    # _save_examples 已禁用（耗时 + 占显存）；推理样例由 evaluate.py 统一处理


def _save_examples(model, tokenizer, problem_types, n, save_dir, num_examples=3):
    """
    对每种问题类型各生成 num_examples 个推理样例并保存到 save_dir/examples.json。
    每条样例记录：
        - problem_type, instance_id
        - prompt_tokens:      prompt 的 token 数（与 max_prompt_length 比较判断是否截断）
        - completion_tokens:  生成的 token 数（与 max_completion_length 比较判断是否截断）
        - truncated:          completion 是否触达 max_completion_length 上限（疑似截断）
        - prompt_text:        完整 prompt 文本
        - completion_text:    模型生成的完整文本（含 <think> 链）
        - terminal_reward:    terminal reward 值（4 维加权和，∈ [0, 4]）
        - is_feasible:        严格可行性判断
    """
    print("\n生成推理样例...")
    model.eval()
    rng = np.random.default_rng(seed=2025)
    examples = []

    for pt in problem_types:
        prob = _get_problem(pt)
        for i in range(num_examples):
            instance   = prob.generate_instance(n, rng)
            prompt     = prob.build_prompt(instance)
            chat_text  = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            inputs     = tokenizer(chat_text, return_tensors="pt").to(model.device)
            prompt_tokens = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_completion_length,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completion_ids    = outputs[0][prompt_tokens:]
            completion_tokens = len(completion_ids)
            completion_text   = tokenizer.decode(completion_ids, skip_special_tokens=True)
            truncated         = (completion_tokens >= config.max_completion_length)

            instance_for_eval = prob.from_json(prob.to_json(instance))
            is_feasible = prob.is_feasible(completion_text, instance_for_eval)
            tour_dist   = prob.get_tour_distance(completion_text, instance_for_eval)
            term_reward = compute_terminal_reward(completion_text, instance_for_eval, pt)

            examples.append({
                "problem_type":      pt,
                "instance_id":       i,
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "truncated":         truncated,
                "is_feasible":       is_feasible,
                "terminal_reward":   round(term_reward, 4),
                "tour_distance":     round(tour_dist, 4) if tour_dist is not None else None,
                "prompt_text":       chat_text,
                "completion_text":   completion_text,
            })

            status = "✓可行" if is_feasible else "✗不可行"
            trunc  = " ⚠️截断" if truncated else ""
            print(f"  [{pt}] 样例{i+1}: terminal={term_reward:.3f} {status}"
                  f"  prompt={prompt_tokens}tok  completion={completion_tokens}tok{trunc}")

    out_path = os.path.join(save_dir, "examples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"样例已保存到: {out_path}")


if __name__ == "__main__":
    main()
