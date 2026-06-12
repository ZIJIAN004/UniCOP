#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_grpo_foarl.py — FOARL CVRP RL (Stage-2), GRPO + 规则奖励 (可行性+最优性)。

复现 FOARL (LLMCoSolver, NeurIPS 2025, arXiv:2509.16865) 的 Stage-2:
  在 Stage-1 SFT 模型基础上做 GRPO, 用 foarl_reward_cvrp 的规则奖励
  (R^P = R_f + R_o) 显式压低约束违反、精炼解质量。无 reward model / 无 PRM,
  纯规则奖励 —— 这正是论文 FOARL 的形态。

栈与 train_sft_foarl.py 一致 (transformers + TRL + LoRA + DeepSpeed ZeRO-3),
仅把 SFTTrainer 换成 trl.GRPOTrainer。GRPOConfig 字段名对齐本团队
UniCOP-Reason/train.py 在用的 TRL 版本 (num_generations/beta/epsilon/epsilon_high)。

数据 (build_foarl_cvrp_data.py 产出的 foarl_cvrp20.jsonl) 每条:
  {instruction, input, output, instance:[coords,demands,capacity]}
  - prompt   : 与 SFT 完全相同的 chat 渲染 (system=preamble, user=instruction\\n\\ninput),
               add_generation_prompt=True, 但不含 answer (RL 自己采样补全)。
  - ref_dist : 从 output 的 "Objective: X" 解析出的 solver 参考目标值, 供 R_o 的 gap。
  - instance : 存成 JSON 字符串列, 规避 Arrow 对 [coords,demands,cap] 混合嵌套的 schema 报错。

启动 (从 merged SFT 模型开始; LoRA adapter 须先 merge 回基座):
  单卡:  python train_grpo_foarl.py --model <MERGED_SFT_DIR> --data data/foarl_cvrp20.jsonl
  多卡:  accelerate launch --num_processes 4 train_grpo_foarl.py --zero_stage 3 \
            --gradient_checkpointing --model <MERGED_SFT_DIR> --data data/foarl_cvrp20.jsonl

⚠️ num_generations(组大小 S) 必须整除 全局生成批 = per_device_train_batch_size ×
   num_processes × grad_accum。默认 pdtb=8, S=8, ga=4, 4 卡 → 128 % 8 == 0, OK。
⚠️ 默认 use_vllm=False (HF 原生生成, 无需另起 vLLM server, 与 ZeRO-3 兼容)。
   CVRP n=20 补全短(~数百 tok), 原生生成可接受; 需更快可 --use_vllm 走 server 模式。
"""
import argparse
import json
import os
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from foarl_reward_cvrp import compute_foarl_reward_cvrp

# 与 SFT 完全一致的开场白 (放进 system role), 保证 RL 的 prompt 分布 == SFT 训练分布
FOARL_PREAMBLE = (
    "Below is an instruction describing a combinatorial optimization problem. "
    "It is paired with an input that provides the data of the instance. "
    "Your task is to produce a feasible solution that optimizes (minimizes or maximizes) "
    "the given objective."
)

_RE_OBJ = re.compile(r"Objective:\s*([-\d.]+)")


# ── DeepSpeed 配置 (与 train_sft_foarl.py 一致) ────────────────────────────────
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


def load_foarl_rl_dataset(data_paths, tokenizer, max_prompt_length, max_samples=0):
    """加载 FOARL jsonl, 渲染成 GRPO 需要的 {prompt, instance_json, ref_distance, id}。"""
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    # 探针: generation prompt 不应含 <think> (非思维 Instruct)
    probe = tokenizer.apply_chat_template(
        [{"role": "system", "content": "probe"}, {"role": "user", "content": "probe"}],
        tokenize=False, add_generation_prompt=True)
    if "<think>" in probe[-40:]:
        print(f"  [WARN] chat_template 末尾出现 <think>! 基座可能选错 (末尾: {probe[-40:]!r})")
    else:
        print("  [OK  ] chat_template generation prompt 不含 <think> (符合非思维 Instruct)")

    prompts, instances, refs, ids = [], [], [], []
    skipped_no_inst = skipped_no_ref = skipped_long = 0
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
                instance = r.get("instance")
                if instance is None:
                    skipped_no_inst += 1
                    continue
                # ref 目标值: 优先 solver_distance, 否则从 output 的 "Objective: X" 解析
                ref = r.get("solver_distance")
                if ref is None:
                    m = _RE_OBJ.search(r.get("output", ""))
                    ref = float(m.group(1)) if m else None
                if ref is None:
                    skipped_no_ref += 1
                    continue

                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": FOARL_PREAMBLE},
                     {"role": "user", "content": f"{instruction}\n\n{inp}"}],
                    tokenize=False, add_generation_prompt=True)
                if len(tokenizer.encode(prompt_text)) > max_prompt_length:
                    skipped_long += 1
                    continue

                prompts.append(prompt_text)
                instances.append(json.dumps(instance))   # 存 JSON 字符串, 规避 Arrow schema
                refs.append(float(ref))
                ids.append(r.get("id"))
                loaded += 1
        print(f"  [{path}] 加载 {loaded} 条")

    if skipped_no_inst:
        print(f"  跳过 {skipped_no_inst} 条无 instance 字段 (需 build_foarl_cvrp_data.py 重建)")
    if skipped_no_ref:
        print(f"  跳过 {skipped_no_ref} 条无参考目标值 (R_o 的 gap 无从计算)")
    if skipped_long:
        print(f"  过滤 {skipped_long} 条 prompt token > max_prompt_length={max_prompt_length}")
    if not prompts:
        raise RuntimeError("没有可用样本, 检查数据路径 / instance 字段 / max_prompt_length")

    if max_samples > 0 and len(prompts) > max_samples:
        prompts, instances, refs, ids = (prompts[:max_samples], instances[:max_samples],
                                         refs[:max_samples], ids[:max_samples])
        print(f"  [sanity] 截取前 {max_samples} 条")

    # 首条样本验证
    print(f"  [首条样本验证] prompt 末尾 60 字: {prompts[0][-60:]!r}")
    print(f"                  ref_distance[0]={refs[0]:.4f}  instance keys={len(json.loads(instances[0]))} 段")
    print(f"  最终 {len(prompts)} 条")
    return Dataset.from_dict({"prompt": prompts, "instance_json": instances,
                              "ref_distance": refs, "id": ids})


def build_reward_func(reward_kwargs, log_every=50):
    """闭包出 TRL GRPOTrainer 需要的 reward 函数, 并周期性打印组件均值便于盯训练。

    TRL 约定: reward_func(prompts, completions, **kwargs), kwargs 含 dataset 各列
    (instance_json / ref_distance), 按 num_generations 展开后与 completions 等长。
    """
    state = {"calls": 0}

    def reward_func(prompts, completions, instance_json=None, ref_distance=None, **kwargs):
        rewards, comps = [], []
        for i, comp in enumerate(completions):
            inst = json.loads(instance_json[i])
            ref = ref_distance[i] if ref_distance is not None else None
            r, c = compute_foarl_reward_cvrp(comp, inst, ref, **reward_kwargs)
            rewards.append(r)
            comps.append(c)

        state["calls"] += 1
        if state["calls"] % log_every == 1 and comps:
            def _mean(k):
                vals = [c[k] for c in comps if c.get(k) is not None]
                return sum(vals) / len(vals) if vals else float("nan")
            print(f"  [reward#{state['calls']}] R̄={sum(rewards)/len(rewards):.3f} "
                  f"ζ={_mean('parse'):.2f} depot={_mean('depot'):.2f} cov={_mean('coverage'):.2f} "
                  f"cap={_mean('capacity'):.2f} feas={_mean('feasible'):.2f} "
                  f"R_o={_mean('R_o'):.3f} gap={_mean('gap'):.3f}")
        return rewards

    reward_func.__name__ = "foarl_reward"
    return reward_func


def main():
    ap = argparse.ArgumentParser(description="FOARL CVRP Stage-2 GRPO RL (规则奖励)")
    ap.add_argument("--model", required=True, help="merged SFT 模型路径 (Stage-1 产物, LoRA 须先 merge)")
    ap.add_argument("--data", nargs="+", default=["data/foarl_cvrp20.jsonl"])
    ap.add_argument("--output_dir", default="./output_grpo_foarl_cvrp20")
    # LoRA (RL 也用 LoRA, 与 SFT 同规格)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    # ── GRPO 超参 ────────────────────────────────────────────────────────
    # ── GRPO 超参: 默认对齐官方 rl_train.py (Summer142857/LLMCoSolver), 仅 lr 按用户改 2e-5 ──
    ap.add_argument("--num_generations", type=int, default=8, help="组大小 S (官方默认 8)")
    ap.add_argument("--lr", type=float, default=2e-5, help="RL 学习率 (用户指定 2e-5; 官方默认是 1e-6)")
    ap.add_argument("--epochs", type=int, default=1, help="官方 num_train_epochs=1")
    ap.add_argument("--max_steps", type=int, default=-1, help=">0 时按步数停, 覆盖 epochs")
    ap.add_argument("--batch_size", type=int, default=8, help="per-device prompt 数 (官方 8); 须使全局批被 S 整除")
    ap.add_argument("--grad_accum", type=int, default=8, help="官方 gradient_accumulation_steps=8")
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="官方 warmup_ratio=0.05")
    ap.add_argument("--beta", type=float, default=0.05, help="KL 系数 β (官方 0.05)")
    ap.add_argument("--epsilon", type=float, default=0.1, help="GRPO 裁剪下界 ε (官方 0.1)")
    ap.add_argument("--epsilon_high", type=float, default=0.28, help="裁剪上界 (官方 0.28, DAPO 非对称)")
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout 温度 (官方 RL 未设→TRL 默认; BoN 推理才用 0.7)")
    ap.add_argument("--top_p", type=float, default=1.0, help="rollout top_p (官方 RL 未设)")
    ap.add_argument("--max_prompt_length", type=int, default=20000, help="官方 20000 (CVRP20 实际 prompt 很短, 仅作上限)")
    ap.add_argument("--max_completion_length", type=int, default=1000, help="官方 1000")
    # ── FOARL 奖励权重: 对齐官方 rewards.py 的 CVRP weights (论文附录 A.3.3 同值) ──
    ap.add_argument("--alpha", type=float, default=1.0, help="最优性奖励权重 α (官方 1/(1+gap) → α=1.0)")
    ap.add_argument("--omega_parse", type=float, default=0.2, help="官方 parse=0.2")
    ap.add_argument("--omega_depot", type=float, default=0.1, help="官方 depot_constraint=0.1")
    ap.add_argument("--omega_coverage", type=float, default=0.1, help="官方 coverage=0.1")
    ap.add_argument("--omega_capacity", type=float, default=0.6, help="官方 capacity=0.6")
    # 生成后端
    ap.add_argument("--use_vllm", action="store_true", help="走 TRL vLLM server 模式 (需另起 trl vllm-serve)")
    ap.add_argument("--vllm_server_host", default="0.0.0.0")
    ap.add_argument("--vllm_server_port", type=int, default=8000)
    # 多卡 / 通用
    ap.add_argument("--zero_stage", type=int, default=0, choices=[0, 2, 3])
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None, help="路径或 'auto'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0, help="只取前 N 条 sanity, 0=全量")
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--use_wandb", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print("  FOARL CVRP Stage-2 GRPO RL (规则奖励 R_f + R_o)")
    print(f"  模型(merged SFT): {args.model}")
    print(f"  数据: {args.data}")
    print(f"  GRPO: S={args.num_generations} lr={args.lr} β={args.beta} ε=[{args.epsilon},{args.epsilon_high}] "
          f"ZeRO={args.zero_stage} vLLM={args.use_vllm}")
    print(f"  奖励: α={args.alpha} ω(parse={args.omega_parse},depot={args.omega_depot},"
          f"cov={args.omega_coverage},cap={args.omega_capacity})")
    print(f"  输出: {args.output_dir}")
    print("=" * 60)

    # ── tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if getattr(tokenizer, "add_bos_token", False):
        tokenizer.add_bos_token = False
        print("  ✓ add_bos_token = False")
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # GRPO 生成需要可用且独立的 pad; merged SFT 通常已带, 这里兜底
        for cand in ["<|pad|>", "<|PAD_TOKEN|>"]:
            tid = tokenizer.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.eos_token_id:
                tokenizer.pad_token = cand
                print(f"  ✓ pad_token = {cand!r}")
                break
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print(f"  ⚠️ 新建 pad_token <|pad|> (id={tokenizer.pad_token_id})")
    tokenizer.padding_side = "left"  # 生成阶段必须 left pad
    print(f"  padding_side = {tokenizer.padding_side}")

    # ── 数据 ─────────────────────────────────────────────────────────────
    dataset = load_foarl_rl_dataset(args.data, tokenizer, args.max_prompt_length, args.max_samples)

    # ── 全局批可整除性检查 (TRL 硬约束: 全局生成批 % num_generations == 0) ──
    world = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = args.batch_size * world * args.grad_accum
    if global_batch % args.num_generations != 0:
        raise ValueError(
            f"全局批 {global_batch} (pdtb={args.batch_size}×world={world}×ga={args.grad_accum}) "
            f"不能被 num_generations={args.num_generations} 整除; 调整 batch_size/grad_accum/S。")
    print(f"  [批检查] 全局生成批={global_batch}, S={args.num_generations} → OK")

    # ── 模型 ─────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        print(f"  resize embedding → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    # ── GRPOConfig (字段名对齐 UniCOP-Reason/train.py 在用的 TRL 版) ──────
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=3,
        save_strategy="steps", eval_strategy="no",
        bf16=True, seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,           # 保留 instance_json / ref_distance 给 reward
        deepspeed=make_deepspeed_config(args.zero_stage),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": args.zero_stage == 3},
        # 官方: lr_scheduler='linear', weight_decay=0.02 (官方 optim='adamw_8bit' 是 Unsloth 专用,
        # 本栈走 DeepSpeed ZeRO-3, 优化器由 ds_config 的 AdamW 接管, weight_decay 经 'auto' 透传)
        lr_scheduler_type="linear", weight_decay=0.02,
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
    )

    reward_kwargs = dict(alpha=args.alpha, omega_parse=args.omega_parse, omega_depot=args.omega_depot,
                         omega_coverage=args.omega_coverage, omega_capacity=args.omega_capacity)
    reward_func = build_reward_func(reward_kwargs, log_every=args.logging_steps)

    # ── reward 自检 (rank0): 用首条样本的金标准答案喂 reward, 应拿满 R_f 且 gap≈0 ──
    if os.environ.get("LOCAL_RANK", "0") == "0":
        first = dataset[0]
        inst0 = json.loads(first["instance_json"])
        gold = f"Routes: [[0, 0]], Objective: {first['ref_distance']:.2f}"  # 占位格式探针
        _r, _c = compute_foarl_reward_cvrp(gold, inst0, first["ref_distance"], **reward_kwargs)
        print(f"  [reward 自检] 占位补全 → R={_r:.3f} ζ={_c['parse']} (ζ=1 说明 Routes/Objective 正则可命中)")
        if _c["parse"] == 0.0:
            print("  [FAIL] 占位补全 ζ=0, 正则没命中, 检查 foarl_reward_cvrp 的解析或输出格式!")

    trainer = GRPOTrainer(
        model=model, args=grpo_config,
        reward_funcs=reward_func, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )

    resume = args.resume_from_checkpoint
    if resume == "auto":
        ckpts = [(int(d.split("-")[1]), os.path.join(args.output_dir, d))
                 for d in (os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else [])
                 if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
        resume = max(ckpts)[1] if ckpts else None
        print(f"  resume=auto → {resume}")

    print("\n开始 FOARL GRPO...")
    trainer.train(resume_from_checkpoint=resume) if resume else trainer.train()

    save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存到: {save_path}")
    if peft_config is not None and trainer.accelerator.is_main_process:
        print("LoRA adapter 已存; eval 前需 merge")


if __name__ == "__main__":
    main()
