"""
Stage 2 SFT：在 Stage 1 ckpt 上用 Gemini 推理链教 <think>...</think> 推理格式。

基座模型：Stage 1 输出的 Qwen2.5-Instruct ckpt（已学会生成可行解）
数据来源：generate_chains.py 生成的 chains.jsonl
训练目标：模型根据原始问题描述，生成 <think>...</think> 推理链 + 格式化答案

⚠️ Qwen2.5-Instruct 的 chat_template 末尾不带 <think>，
    脚本会自动检测并在 prompt 末尾手动补 <think>\n，无需手动处理。
    （与 R1-Distill 的 chat_template 吃 think 链问题不同，Qwen2.5 无此坑。）

单卡运行：
    python stage2_reasoning/train_sft_stage2.py
    python stage2_reasoning/train_sft_stage2.py --model ./output_sft_stage1/final_model

多卡运行：
    accelerate launch --num_processes 3 stage2_reasoning/train_sft_stage2.py --zero_stage 2 --gradient_checkpointing
"""

import argparse
import hashlib
import json
import os

import torch

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from peft import LoraConfig


# ── 默认参数 ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "./output_sft_stage1/final_model"  # Stage 1 输出，或原始 base model 路径
DEFAULT_DATA  = ["data/chains_v3_clean.jsonl"]

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
            "reduce_bucket_size":   5e7,      # LoRA 梯度仅 ~5MB, 小 bucket 最大化通信-计算 overlap
            "allgather_bucket_size": 5e7,
            "reduce_scatter":       True,
            "round_robin_gradients": True,   # 梯度分片负载均衡
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


def _detect_response_template(tokenizer) -> str:
    """从 chat_template 探测 assistant 标记，适配 ChatML / DeepSeek / Llama 3 等格式。

    Stage 2 训练数据中 prompt 末尾保留 chat_template 自动 prepend 的 <think>\n
    （Qwen3-Thinking 强制如此；R1-Distill 同样），因此 response_template
    必须包含完整 generation_prompt 末尾的 <think>\n。
    若剥掉 <think>，DataCollatorForCompletionOnlyLM 会把模板自动生成的
    <think>\n 误判为 completion，造成 loss mask 错位。
    """
    msgs = [{"role": "user", "content": "Hi"}]
    without_gp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    with_gp    = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gen_prompt = with_gp[len(without_gp):]
    return gen_prompt


def _dataset_cache_key(data_path, tokenizer_name, max_length, max_output_length,
                       filter_problems, filter_sizes) -> str:
    """处理后数据集的缓存 key: 数据文件 mtime/size + 所有影响产物的参数 + tokenizer。

    任一数据文件被重新生成 (mtime/size 变) 或参数变更 → key 变 → 自动失效重建,
    不会用到过期缓存。
    """
    paths = [data_path] if isinstance(data_path, str) else list(data_path)
    parts = []
    for p in sorted(paths):
        try:
            st = os.stat(p)
            parts.append(f"{os.path.abspath(p)}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{os.path.abspath(p)}:MISSING")
    blob = "|".join(parts) + (
        f"||ml={max_length}|mol={max_output_length}"
        f"|fp={sorted(filter_problems) if filter_problems else None}"
        f"|fs={sorted(filter_sizes) if filter_sizes else None}"
        f"|tok={tokenizer_name}"
    )
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:16]


def load_sft_dataset(data_path, tokenizer, max_length: int,
                     max_output_length: int = 4096,
                     filter_problems=None, filter_sizes=None) -> Dataset:
    """
    从 chains.jsonl 加载数据，构建 Stage 2 SFT 训练集。

    `data_path` 可以是单个文件路径 (str) 或多个文件路径 (list[str])。
    多文件时会按顺序读入并拼接，过滤、长度统计、首条样本验证均针对合集执行。

    基座模型为 Qwen2.5-Instruct（经 Stage 1 微调），其 chat_template 末尾
    不带 <think>，需手动在 prompt 末尾补 <think>\\n。
    Qwen2.5 不会吃 assistant 消息里的 <think>...</think>（无 R1-Distill 那个坑），
    但仍用 prompt+completion 手动拼接方式，与 completion_only_loss=True 配合。

    每条样本渲染后的训练 text 结构:
        <|im_start|>system\\n...\\n<|im_end|>
        <|im_start|>user\\nPlan route...\\n<|im_end|>
        <|im_start|>assistant\\n<think>\\n
        I need to find the shortest route...
        </think>
        Route: 0 -> ... -> 0
        <|im_end|>
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

    data_paths = [data_path] if isinstance(data_path, str) else list(data_path)

    for path in data_paths:
        per_file_loaded = 0
        with open(path, "r", encoding="utf-8") as f:
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

                # 2. 探针兜底: Qwen2.5-Instruct 的 chat_template 末尾无 <think>,手动补
                #    注意不能 rstrip()，否则会吃掉 ChatML 格式中 assistant 后的换行符
                if not probe_ends_with_think:
                    prompt_text += "<think>\n"

                # 3. Gemini output 开头必有 <think> (generate_chains.py 强制),
                #    和 prompt 末尾的 <think> 重复,剥掉
                output_stripped = output.lstrip()
                if output_stripped.startswith("<think>"):
                    output_stripped = output_stripped[len("<think>"):].lstrip("\n")

                # 4. completion = thinking + </think> + answer + eos
                #    (eos_token 显式加, 让模型学会"完整答案结束就停止")
                completion_text = output_stripped + tokenizer.eos_token

                # 长度过滤推迟到循环外批量编码 (见下方),
                # 单条 tokenizer.encode 是串行的且有 per-call 开销,
                # 批量 tokenizer(list) 能用上 fast tokenizer 的 Rust 多线程。
                records.append({
                    "prompt":     prompt_text,
                    "completion": completion_text,
                    "problem_type": r.get("problem_type", "unknown"),
                    "n": r.get("n", 0),
                })
                per_file_loaded += 1
        print(f"  [{path}] 加载 {per_file_loaded} 条")

    # ── 批量编码算 token 长度 (fast tokenizer Rust 多线程, 远快于循环里逐条 encode) ──
    #    分块 1000 条只取 len, 避免一次性持有 5 万条 input_ids 的内存。
    #    语义与单条 tokenizer.encode 一致 (都默认 add_special_tokens=True)。
    def _batch_token_lens(texts, chunk=1000):
        lens = []
        for i in range(0, len(texts), chunk):
            enc = tokenizer(texts[i:i + chunk], add_special_tokens=True)["input_ids"]
            lens.extend(len(x) for x in enc)
        return lens

    if records:
        clens = _batch_token_lens([r["completion"] for r in records])
        plens = _batch_token_lens([r["prompt"] for r in records])
        kept = []
        for r, cl, pl in zip(records, clens, plens):
            # 阶段 1: completion (output) 超长过滤
            if cl > max_output_length:
                skipped_too_long += 1
                continue
            r["_clen"] = cl
            r["_plen"] = pl
            kept.append(r)
        records = kept

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

    before_filter = len(records)
    # 阶段 2: prompt+completion 总长过滤, 复用上面批量算好的 token 长度, 不再二次编码
    records = [r for r in records if r["_plen"] + r["_clen"] <= max_length]
    skipped_total_len = before_filter - len(records)
    if skipped_total_len:
        print(f"  过滤 {skipped_total_len} 条 prompt+completion 超过 max_length={max_length} 的样本")

    return Dataset.from_dict({
        "text": [r["prompt"] + r["completion"] for r in records],
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
    parser.add_argument("--data",         type=str, nargs="+", default=DEFAULT_DATA,
                        help="chains.jsonl 文件路径，可传多个（按顺序拼接训练集）")
    parser.add_argument("--filter_problems", type=str, nargs="+", default=None,
                        help="只训练指定问题类型，如 tsp cvrp")
    parser.add_argument("--filter_sizes", type=int, nargs="+", default=None,
                        help="只训练指定规模，如 20 50")
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
    parser.add_argument("--max_samples",  type=int,   default=0,
                        help="只取前 N 条样本(sanity test 用), 0=全量")
    parser.add_argument("--dataset_num_proc", type=int, default=0,
                        help="TRL 内部 tokenize 的并行进程数(0=自动取 min(8,CPU))。"
                             "大数据集长序列时单核 tokenize 极慢, 多核可线性加速。"
                             "也可用环境变量 SFT_NUM_PROC 覆盖。")
    parser.add_argument("--no_data_cache", action="store_true",
                        help="禁用处理后数据集的磁盘缓存(默认开启)。缓存按数据文件"
                             "mtime/size + 长度/filter/tokenizer 做 key, 命中则跳过"
                             "整个加载+编码循环, 重跑从 ~30min 降到秒级。"
                             "也可用环境变量 SFT_DATA_CACHE=0 关闭。")

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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从 ckpt 恢复, 可传具体路径或 'auto' 自动找 output_dir 内最新 checkpoint-*")

    # 输出
    parser.add_argument("--output_dir",   type=str, default="./output_sft")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps",   type=int,  default=100)
    parser.add_argument("--use_wandb",    action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    # TF32: Ampere+ GPU 上 matmul 加速 ~1.3×，精度损失可忽略
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"{'='*60}")
    print(f"  SFT 蒸馏训练")
    print(f"  模型:       {args.model}")
    _data_show = args.data if isinstance(args.data, str) else ", ".join(args.data)
    print(f"  数据:       {_data_show}")
    print(f"  LoRA:       {'关闭' if args.no_lora else f'rank={args.lora_rank}'}")
    print(f"  最大序列长: {args.max_length}")
    print(f"  最大output: {args.max_output_length}")
    print(f"  ZeRO:       {args.zero_stage}")
    print(f"  输出:       {args.output_dir}")
    print(f"{'='*60}\n")

    # ── 加载 tokenizer ───────────────────────────────────────────────────
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # chat_template 已在文本开头渲染 BOS，禁止 tokenizer 再自动加一个
    if getattr(tokenizer, "add_bos_token", False):
        tokenizer.add_bos_token = False
        print("  ✓ 已设置 add_bos_token = False (防止双 BOS)")

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
    _use_cache = (not args.no_data_cache) and os.environ.get("SFT_DATA_CACHE", "1") != "0"

    def _build_dataset():
        return load_sft_dataset(args.data, tokenizer, args.max_length, args.max_output_length,
                                filter_problems=args.filter_problems,
                                filter_sizes=args.filter_sizes)

    if _use_cache:
        # 多 rank 协调: 用文件系统而非 NCCL barrier。
        # ⚠️ 不能用 accelerate main_process_first(): 它在入口 torch.distributed.barrier()
        #    阻塞非主进程等 rank0 跑完 body(=30min 构建), 而 barrier 是首个 collective,
        #    NCCL 默认超时仅 600s → rank0 构建超 10min 时非主进程在 NCCL 初始化处超时崩溃
        #    (DistBackendError: wait timeout after 600000ms / broadcastUniqueNCCLID)。
        # 改为: rank0 构建+落盘+写 .done 标记; 其余 rank 轮询文件系统(无 collective)等标记,
        #    NCCL 初始化自然推迟到 SFTTrainer (所有 rank 到齐时), 不再被构建阻塞。
        from accelerate import PartialState
        _state = PartialState()
        _key = _dataset_cache_key(args.data, args.model, args.max_length,
                                  args.max_output_length, args.filter_problems, args.filter_sizes)
        _cache_root = os.environ.get(
            "SFT_CACHE_DIR",
            os.path.join(os.path.dirname(os.path.abspath(
                args.data[0] if isinstance(args.data, list) else args.data)), ".sft_cache"))
        _cache_path = os.path.join(_cache_root, f"sft_{_key}")
        _done_marker = _cache_path + ".done"   # 完整性标记(落盘成功后才写), sibling 不污染 dataset 目录

        if os.path.isfile(_done_marker):
            # 已有完整缓存: 所有 rank 直接 load, 无需任何协调
            print(f"  [cache] 命中处理后数据集, 跳过加载+编码: {_cache_path}")
            dataset = load_from_disk(_cache_path)
        elif _state.is_main_process:
            # 仅主进程构建; 先清理上次未完成的残留(有目录但无 .done), 避免 save_to_disk 撞已存在目录
            import shutil
            if os.path.isdir(_cache_path):
                shutil.rmtree(_cache_path, ignore_errors=True)
            dataset = _build_dataset()
            os.makedirs(_cache_root, exist_ok=True)
            dataset.save_to_disk(_cache_path)
            with open(_done_marker, "w") as _f:
                _f.write("ok")
            print(f"  [cache] 已落盘处理后数据集: {_cache_path}")
        else:
            # 非主进程: 轮询文件系统等 rank0 写出 .done (不走 NCCL), 带超时防 rank0 崩溃后死等
            import time
            _t0 = time.time()
            _timeout_s = int(os.environ.get("SFT_CACHE_BUILD_TIMEOUT", "7200"))
            while not os.path.isfile(_done_marker):
                if time.time() - _t0 > _timeout_s:
                    raise TimeoutError(
                        f"等待 rank0 构建数据集缓存超时 ({_timeout_s}s): {_done_marker}。"
                        f" rank0 可能已崩溃, 请检查 rank0 日志。")
                time.sleep(10)
            print(f"  [cache] rank{_state.process_index}: rank0 已完成构建, 加载 {_cache_path}")
            dataset = load_from_disk(_cache_path)
    else:
        dataset = _build_dataset()

    # sanity 模式: 只取前 N 条
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
        print(f"  [sanity] 截取前 {args.max_samples} 条样本")

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
    # attn_implementation 默认 flash_attention_2: O(n²) attention 激活不再 materialize,
    # 长序列 (cvrp100 ~13.5k token) 省大量显存 + 提速。transformers 4.57 默认走 sdpa,
    # 不会自动用 flash-attn,必须显式指定。可用 SFT_ATTN_IMPL 覆盖 (如 sdpa/eager 调试)。
    _attn_impl = os.environ.get("SFT_ATTN_IMPL", "flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=_attn_impl,
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

    # TRL 内部 tokenize 的并行度: 优先级 CLI --dataset_num_proc > 环境变量 SFT_NUM_PROC
    # > 自动 min(8, CPU)。单核 tokenize 50k×13k token 极慢, 多核近线性加速。
    _num_proc = args.dataset_num_proc or int(os.environ.get("SFT_NUM_PROC", "0")) \
        or min(8, os.cpu_count() or 1)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        dataset_num_proc=_num_proc,
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
        gradient_checkpointing_kwargs={"use_reentrant": args.zero_stage == 3},
        # Liger fused linear CE: 不 materialize logits (vocab 151936 × seq 13.5k × bf16
        # ≈ 4GB,CE 内部 fp32 upcast 再翻倍,ZeRO-3 不分片这块,gc 也救不了),
        # 直接 hidden→loss 融合,峰值显存大降。与 LoRA/PEFT/DeepSpeed ZeRO-3 兼容
        # (layer 级 patch,不碰 adapter 权重)。注意: 我们的 completion-only 走
        # DataCollatorForCompletionOnlyLM (collator 内 label=-100),非 TRL 的
        # assistant_only_loss 路径,liger 的 fused CE 尊重 ignore_index=-100,mask 保留
        # (trl#3781 的静默丢 mask 坑只影响 assistant_only_loss,不影响我们)。
        # 可用 SFT_USE_LIGER=0 关闭。
        use_liger_kernel=os.environ.get("SFT_USE_LIGER", "1") != "0",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
    )

    # ── Completion-only loss（TRL 0.16 不支持 SFTConfig.completion_only_loss）──
    response_template_str = _detect_response_template(tokenizer)
    response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    print(f"  response_template: {response_template_str!r} → IDs: {response_template_ids}")
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ── 训练 ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    print("\n开始 SFT 训练...")
    # ── resume 处理: 'auto' 自动找 output_dir 下最新 checkpoint-*, 否则传具体路径 ──
    resume = args.resume_from_checkpoint
    if resume == "auto":
        ckpts = []
        if os.path.isdir(args.output_dir):
            for d in os.listdir(args.output_dir):
                if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit():
                    ckpts.append((int(d[len("checkpoint-"):]), os.path.join(args.output_dir, d)))
        if ckpts:
            ckpts.sort()
            resume = ckpts[-1][1]
            print(f"  ✓ resume_from_checkpoint=auto → 选最新 ckpt: {resume}")
        else:
            print(f"  [INFO] resume=auto 但 {args.output_dir} 下无 checkpoint-*, 从头训")
            resume = None
    if resume:
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    # ── 保存 ─────────────────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    if peft_config is not None:
        if trainer.accelerator.is_main_process:
            print(f"LoRA adapter 已保存到: {save_path}")
            print("请在训练结束后单独运行 merge_adapter.py 合并到基座模型")
        trainer.accelerator.wait_for_everyone()

    print(f"\n模型已保存到: {save_path}")



if __name__ == "__main__":
    main()
