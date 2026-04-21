"""
评估脚本：测试训练后的模型在各类 COP 问题上的性能。

支持两种推理后端：
  - local: 本地加载 HuggingFace 模型（默认，需要 GPU）
  - api:   通过 Vertex AI Gemini API 调用云端模型

指标：
  1. 全局可行率   = 可行解数 / (实例数 × 采样次数)
  2. 实例级可行率 = 至少有一个可行解的实例数 / 实例数
  3. 最优距离均值 = 每个实例中可行解的最短距离，取所有实例均值（仅统计有可行解的实例）
  4. 推理链长度   = completion 的平均 / 最小 / 最大 token 数

运行示例：
    # ── 本地模型 ──────────────────────────────────────────────────────
    python evaluate.py --model_path ./output/tsptw_n10/final_model \
        --problem tsp tsptw vrptw cvrp --problem_size 20 50 100 \
        --prompt_mode think --batch_size 4

    # ── Vertex AI Gemini ─────────────────────────────────────────────
    python evaluate.py --backend api \
        --api_model gemini-2.5-flash \
        --problem tsp tsptw --problem_size 20 50 \
        --prompt_mode think --num_samples 5 --temperature 0.7

    # ── 指定 credentials 和 project ──────────────────────────────────
    python evaluate.py --backend api \
        --api_model gemini-2.5-pro \
        --gcp_credentials /path/to/credentials.json \
        --gcp_project my-project --gcp_location us-central1 \
        --problem tsptw --problem_size 10
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from config import config
from problems import get_problem, SUPPORTED_PROBLEMS


# ── 结构化提示词（仅 evaluate 使用，不影响训练） ──────────────────────────────

_STRUCTURED_SYSTEM = {
    "tsp": """You are a route planning expert solving the Travelling Salesman Problem (TSP).
Rules: Starting from node 0, visit all customer nodes exactly once and return to node 0, minimizing total distance.
Before answering, think through the problem step by step in <think>...</think>. Inside <think>, you MUST follow this structured format:

Strategy: [1-2 sentences]
Step 1: → [node_id] ([reason in a few words])
Step 2: → [node_id] ([reason])
...
Step 10: → [node_id] ([reason])
  Remaining: [list of unvisited node ids]
(continue; output Remaining every 10 steps)

After </think>, output ONLY the final route:
Route: 0 -> A -> B -> C -> ... -> 0""",

    "tsptw": """You are a route planning expert solving the Travelling Salesman Problem with Time Windows (TSPTW).
Rules:
- Start from node 0 (depot), visit all customer nodes exactly once, and return to node 0
- Travel time between nodes = Euclidean distance
- Each customer node has a time window [earliest, latest]: arrival time must be <= latest
- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue
- Objective: minimize total travel distance
Before answering, think through the problem step by step in <think>...</think>. Inside <think>, you MUST follow this structured format:

Strategy: [1-2 sentences]
Step 1: → [node_id] ([reason]) | arr=[value] tw=[earliest,latest]
Step 2: → [node_id] ([reason]) | arr=[value] tw=[earliest,latest]
...
Step 10: → [node_id] ([reason]) | arr=[value] tw=[earliest,latest]
  Remaining: [list of unvisited node ids]
(continue; output Remaining every 10 steps)

After </think>, output ONLY the final route:
Route: 0 -> A -> B -> C -> ... -> 0""",

    "tspdl": """You are a logistics route planning expert solving the Travelling Salesman Problem with Draft Limits (TSPDL).
Rules:
- Start from node 0 (depot), visit all customer nodes exactly once, and return to node 0
- Travel time between nodes = Euclidean distance
- The vehicle departs fully loaded (initial load = total capacity); load decreases as cargo is unloaded at each customer
- Upon arriving at a node (before unloading), current load must be <= that node's draft_limit
- Objective: minimize total travel distance
Before answering, think through the problem step by step in <think>...</think>. Inside <think>, you MUST follow this structured format:

Strategy: [1-2 sentences]
Step 1: → [node_id] ([reason]) | load=[value] dlimit=[value]
Step 2: → [node_id] ([reason]) | load=[value] dlimit=[value]
...
Step 10: → [node_id] ([reason]) | load=[value] dlimit=[value]
  Remaining: [list of unvisited node ids]
(continue; output Remaining every 10 steps)

After </think>, output ONLY the final route:
Route: 0 -> A -> B -> C -> ... -> 0""",

    "cvrp": """You are a logistics route planning expert solving the Capacitated Vehicle Routing Problem (CVRP).
Rules: Multiple vehicles depart from node 0; each vehicle visits a subset of customers and returns to node 0; total demand per route must not exceed vehicle capacity; each customer is visited exactly once; minimize total distance.
Before answering, think through the problem step by step in <think>...</think>. Inside <think>, you MUST follow this structured format, building one route at a time:

Strategy: [1-2 sentences]
Route 1:
  Step 1: → [node_id] ([reason]) | dem=[value]/[capacity]
  Step 2: → [node_id] ([reason]) | dem=[value]/[capacity]
  ...
  (output Remaining every 10 steps within each route)
Route 2:
  Step 1: → [node_id] ([reason]) | dem=[value]/[capacity]
  ...

After </think>, output ONLY the final routes (one per line):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0""",

    "vrptw": """You are a logistics scheduling expert solving the Vehicle Routing Problem with Time Windows (VRPTW).
Rules:
- Multiple vehicles depart from node 0 (depot); each vehicle visits a subset of customers and returns to node 0
- All customer nodes are visited exactly once
- Travel time between nodes = Euclidean distance
- Each customer node has a time window [earliest, latest]: arrival time must be <= latest
- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue
- Objective: minimize total travel distance across all routes
Before answering, think through the problem step by step in <think>...</think>. Inside <think>, you MUST follow this structured format, building one route at a time:

Strategy: [1-2 sentences]
Route 1:
  Step 1: → [node_id] ([reason]) | arr=[value] tw=[earliest,latest]
  Step 2: → [node_id] ([reason]) | arr=[value] tw=[earliest,latest]
  ...
  (output Remaining every 10 steps within each route)
Route 2:
  Step 1: → [node_id] ([reason]) | arr=[value] tw=[earliest,latest]
  ...

After </think>, output ONLY the final routes (one per line):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0""",

    "cvrptw": """You are a logistics scheduling expert solving the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
Rules: Multiple vehicles depart from node 0; total demand per route <= vehicle capacity; each node has a time window [earliest, latest]; travel time = Euclidean distance; early arrival allows waiting, late arrival is a violation; each customer is visited exactly once; minimize total distance.
Before answering, think through the problem step by step in <think>...</think>. Inside <think>, you MUST follow this structured format, building one route at a time:

Strategy: [1-2 sentences]
Route 1:
  Step 1: → [node_id] ([reason]) | dem=[value]/[capacity] arr=[value] tw=[earliest,latest]
  Step 2: → [node_id] ([reason]) | dem=[value]/[capacity] arr=[value] tw=[earliest,latest]
  ...
  (output Remaining every 10 steps within each route)
Route 2:
  Step 1: → [node_id] ([reason]) | dem=[value]/[capacity] arr=[value] tw=[earliest,latest]
  ...

After </think>, output ONLY the final routes (one per line):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0""",
}

# 单路线问题：structured 模式下 user prompt 末尾的输出提示
_STRUCTURED_USER_SUFFIX_SINGLE = (
    "\nSolve step by step, then output the final route:"
    "\nRoute: 0 -> A -> B -> C -> ... -> 0"
)
# 多路线问题
_STRUCTURED_USER_SUFFIX_MULTI = (
    "\nSolve step by step, then output the final routes (one per line):"
    "\nRoute 1: 0 -> node -> ... -> 0\nRoute 2: 0 -> node -> ... -> 0\n..."
)


def _apply_structured_prompt(prompt: list[dict], problem_type: str) -> list[dict]:
    """
    将 prob.build_prompt() 生成的原始 prompt 替换为结构化版本。
    仅替换 system message 和 user message 末尾的输出格式提示。
    """
    structured_sys = _STRUCTURED_SYSTEM.get(problem_type)
    if structured_sys is None:
        return prompt  # 不支持的问题类型，原样返回

    new_prompt = []
    for msg in prompt:
        if msg["role"] == "system":
            new_prompt.append({"role": "system", "content": structured_sys})
        elif msg["role"] == "user":
            # 替换 user message 末尾的输出格式提示
            content = msg["content"]
            # 截掉原始 prompt 末尾的输出格式说明（及 TSPDL 的 Note 提示）
            for marker in ["\nNote:", "\nOutput format", "\nSolve step by step"]:
                idx = content.find(marker)
                if idx != -1:
                    content = content[:idx]
                    break

            is_multi = problem_type in ("cvrp", "vrptw", "cvrptw")
            content += _STRUCTURED_USER_SUFFIX_MULTI if is_multi else _STRUCTURED_USER_SUFFIX_SINGLE
            new_prompt.append({"role": "user", "content": content})
        else:
            new_prompt.append(msg)

    return new_prompt


# ── 推理后端：本地模型 ──────────────────────────────────────────────────────────

def _load_local_model(model_path: str):
    """加载本地 HuggingFace 模型和 tokenizer。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载本地模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # 生成任务 padding 在左侧

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _generate_local(model, tokenizer, prompts: list[list[dict]],
                    num_samples: int, temperature: float,
                    max_completion_length: int, batch_size: int,
                    repetition_penalty: float = 1.0,
                    no_repeat_ngram_size: int = 0) -> list[list[str]]:
    """
    本地模型批量推理。

    Args:
        prompts: 每个元素是一个 chat 格式的 prompt（list[dict]）
        num_samples: 每个 prompt 的采样次数
        temperature: 采样温度
        max_completion_length: 最大生成 token 数
        batch_size: batch 大小

    Returns:
        completions: len(prompts) × num_samples 的二维列表，每个元素是 completion 文本
    """
    import torch

    # 先用 tokenizer 把 chat prompt 转为 text
    chat_texts = []
    for prompt in prompts:
        chat_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        chat_texts.append(chat_text)

    all_completions = [[] for _ in range(len(prompts))]

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_texts = chat_texts[batch_start:batch_end]
        cur_batch_size = len(batch_texts)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_completion_length,
            do_sample=(num_samples > 1),
            temperature=temperature if num_samples > 1 else 1.0,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()

        for i in range(cur_batch_size):
            input_len = input_lens[i]
            for s in range(num_samples):
                idx = i * num_samples + s
                output_ids = outputs[idx]
                completion_ids = output_ids[input_len:]
                mask = completion_ids != tokenizer.pad_token_id
                completion_ids = completion_ids[mask]
                # 标记是否被截断（token 数达到 max_completion_length）
                num_tokens = len(completion_ids)
                is_truncated = (num_tokens >= max_completion_length)
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                all_completions[batch_start + i].append((completion, is_truncated, num_tokens))

    return all_completions


# ── 推理后端：vLLM ─────────────────────────────────────────────────────────────

def _load_vllm_model(model_path: str, tensor_parallel_size: int = 1):
    """
    加载 vLLM 模型。
    如果 model_path 是 LoRA adapter 目录（含 adapter_config.json 但无 config.json），
    则自动合并 LoRA 到基座模型后再加载。
    """
    import os
    from vllm import LLM

    actual_path = model_path

    # 检测是否为 LoRA adapter（有 adapter_config.json 但没有 config.json）
    is_lora = (
        os.path.isfile(os.path.join(model_path, "adapter_config.json"))
        and not os.path.isfile(os.path.join(model_path, "config.json"))
    )

    if is_lora:
        import json
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        merged_path = os.path.join(os.path.dirname(model_path), "merged_model")

        if os.path.isdir(merged_path) and os.path.isfile(os.path.join(merged_path, "config.json")):
            print(f"已存在合并模型: {merged_path}，跳过合并")
        else:
            # 读取 adapter_config 获取基座模型路径
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_cfg = json.load(f)
            base_model_path = adapter_cfg.get("base_model_name_or_path", "")
            print(f"检测到 LoRA adapter，基座模型: {base_model_path}")
            print(f"合并 LoRA → {merged_path} ...")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
                device_map="cpu",
            )
            lora_model = PeftModel.from_pretrained(base_model, model_path)
            merged = lora_model.merge_and_unload()
            merged.save_pretrained(merged_path)
            AutoTokenizer.from_pretrained(model_path).save_pretrained(merged_path)
            del base_model, lora_model, merged
            torch.cuda.empty_cache()
            print("LoRA 合并完成")

        actual_path = merged_path

    print(f"加载 vLLM 模型: {actual_path}  (tp={tensor_parallel_size})")
    model = LLM(
        model=actual_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=True,
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def _generate_vllm(model, tokenizer, prompts: list[list[dict]],
                   num_samples: int, temperature: float,
                   max_completion_length: int, batch_size: int) -> list[list]:
    """
    vLLM 批量推理（continuous batching，忽略 batch_size 参数）。
    """
    from vllm import SamplingParams

    # 把 chat prompt 转为 text
    chat_texts = []
    for prompt in prompts:
        chat_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        chat_texts.append(chat_text)

    sampling_params = SamplingParams(
        max_tokens=max_completion_length,
        temperature=temperature if num_samples > 1 else 0,
        n=num_samples,
    )

    outputs = model.generate(chat_texts, sampling_params)

    all_completions = [[] for _ in range(len(prompts))]
    for i, output in enumerate(outputs):
        for sample in output.outputs:
            completion = sample.text
            num_tokens = len(sample.token_ids)
            is_truncated = (num_tokens >= max_completion_length)
            all_completions[i].append((completion, is_truncated, num_tokens))

    return all_completions


# ── 推理后端：Vertex AI Gemini ───────────────────────────────────────────────

def _create_gemini_client(credentials_path: str, project: str, location: str):
    """创建 Vertex AI Gemini 客户端。"""
    from google import genai

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )
    return client


def _chat_to_gemini_contents(prompt: list[dict]) -> tuple[str | None, list]:
    """
    将 chat 格式 [{"role": "system", ...}, {"role": "user", ...}]
    转换为 Gemini API 的 (system_instruction, contents) 格式。

    Gemini 不支持 system role 在 contents 中，需要单独传 system_instruction。
    """
    from google.genai.types import Content, Part

    system_text = None
    contents = []

    for msg in prompt:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            contents.append(Content(
                role="user",
                parts=[Part.from_text(text=msg["content"])],
            ))
        elif msg["role"] == "assistant":
            contents.append(Content(
                role="model",
                parts=[Part.from_text(text=msg["content"])],
            ))

    return system_text, contents


def _call_gemini_single(client, model: str, prompt: list[dict],
                        temperature: float,
                        max_output_tokens: int | None = None) -> str:
    """单次 Gemini API 调用，返回 completion 文本。"""
    from google.genai.types import GenerateContentConfig

    system_text, contents = _chat_to_gemini_contents(prompt)

    config_kwargs = {"temperature": temperature}
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = max_output_tokens

    gen_config = GenerateContentConfig(**config_kwargs)
    if system_text:
        gen_config.system_instruction = system_text

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=gen_config,
    )
    return response.text or ""


def _generate_gemini(client, model: str, prompts: list[list[dict]],
                     num_samples: int, temperature: float,
                     max_concurrency: int,
                     max_output_tokens: int | None = None) -> list[list[str]]:
    """
    Gemini API 并发推理（使用线程池）。

    google-genai SDK 是同步的，通过 ThreadPoolExecutor 实现并发。
    限流时指数退避重试，上限 300s，加随机抖动避免惊群。

    Returns:
        completions: len(prompts) × num_samples 的二维列表
    """
    import concurrent.futures
    import threading

    all_completions = [[] for _ in range(len(prompts))]
    lock = threading.Lock()

    def _task(prompt_idx: int, sample_idx: int):
        max_retries = 10
        base_wait = 30
        max_wait = 300
        for attempt in range(max_retries):
            try:
                text = _call_gemini_single(
                    client, model, prompts[prompt_idx],
                    temperature, max_output_tokens,
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in [
                    "429", "rate limit", "resource exhausted",
                    "quota", "too many requests",
                ])
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = min(base_wait * (2 ** attempt), max_wait)
                    sleep_time += random.uniform(0, sleep_time * 0.3)
                    print(f"  RATE LIMIT: prompt={prompt_idx} sample={sample_idx}, "
                          f"等待 {sleep_time:.0f}s 后重试 ({attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    print(f"  WARNING: Gemini 调用失败 (prompt={prompt_idx}, "
                          f"sample={sample_idx}): {e}")
                    text = ""
                    break
        with lock:
            all_completions[prompt_idx].append(text)

    tasks = []
    for i in range(len(prompts)):
        for s in range(num_samples):
            tasks.append((i, s))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(_task, pi, si) for pi, si in tasks]
        for f in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures), desc="Gemini 推理"):
            f.result()

    return all_completions


# ── 评估核心逻辑 ─────────────────────────────────────────────────────────────

def evaluate_single(generate_fn, problem_type: str, num_test: int,
                    problem_size: int, num_samples: int, temperature: float,
                    max_completion_length: int, batch_size: int = 1,
                    save_dir: str | None = None,
                    prompt_mode: str = "think"):
    """
    评估单个 (problem_type, problem_size) 组合。

    Args:
        generate_fn: 推理函数，签名 (prompts, num_samples, temperature,
                     max_completion_length, batch_size) -> list[list[str]]
                     返回 len(prompts) × num_samples 的 completion 列表
    """
    prob = get_problem(problem_type)
    rng  = np.random.default_rng(seed=9999)

    total_samples    = 0
    total_parsed     = 0
    total_feasible   = 0
    instance_has_feas = 0
    best_dists       = []
    completion_lens  = []

    # 收集示例：一个可行、一个不可行（用于结果展示）
    example_feasible   = None   # (instance_idx, completion_text)
    example_infeasible = None

    # 预生成所有实例和 prompt
    instances = []
    prompts = []
    for _ in range(num_test):
        instance = prob.generate_instance(problem_size, rng)
        prompt   = prob.build_prompt(instance)

        # structured 模式：替换 system prompt 和 user 末尾输出格式
        if prompt_mode == "structured":
            prompt = _apply_structured_prompt(prompt, problem_type)

        instances.append(instance)
        prompts.append(prompt)

    # 调用推理后端
    print(f"[{problem_type.upper()} n={problem_size}] 生成 {num_test} 实例 × {num_samples} 采样 ...")
    all_completions = generate_fn(
        prompts, num_samples, temperature, max_completion_length, batch_size,
    )

    # 评估每个实例的所有 completion
    total_truncated = 0
    for i in range(num_test):
        instance = instances[i]
        instance_best = None

        for item in all_completions[i]:
            # 兼容三种格式：
            #   (completion, is_truncated, num_tokens)  — local 后端
            #   (completion, is_truncated)              — 旧格式兼容
            #   纯 str                                  — API 后端
            if isinstance(item, tuple):
                completion = item[0]
                is_truncated = item[1]
                num_tokens = item[2] if len(item) > 2 else None
            else:
                completion, is_truncated, num_tokens = item, False, None

            total_samples += 1
            if is_truncated:
                total_truncated += 1
            comp_len = num_tokens if num_tokens is not None else len(completion)
            completion_lens.append(comp_len)

            dist = prob.get_tour_distance(completion, instance)
            if dist is not None:
                total_parsed += 1

            feasible = prob.is_feasible(completion, instance)
            if feasible:
                total_feasible += 1
                if dist is not None:
                    if instance_best is None or dist < instance_best:
                        instance_best = dist
                # 收集可行示例
                if example_feasible is None:
                    example_feasible = (i, completion)
            else:
                # 收集不可行示例
                if example_infeasible is None:
                    example_infeasible = (i, completion)

        if instance_best is not None:
            instance_has_feas += 1
            best_dists.append(instance_best)

    # ── 计算指标 ──────────────────────────────────────────────────────
    parse_rate         = total_parsed / total_samples if total_samples else 0
    global_feas_rate   = total_feasible / total_samples if total_samples else 0
    instance_feas_rate = instance_has_feas / num_test
    truncation_rate    = total_truncated / total_samples if total_samples else 0
    avg_best_dist      = float(np.mean(best_dists)) if best_dists else float("nan")
    avg_comp_len       = float(np.mean(completion_lens)) if completion_lens else 0.0
    max_comp_len       = int(np.max(completion_lens)) if completion_lens else 0
    min_comp_len       = int(np.min(completion_lens)) if completion_lens else 0

    # ── 打印结果 ──────────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  {problem_type.upper()}  n={problem_size}  |  {num_test} 实例 × {num_samples} 采样 = {total_samples} 次")
    print(f"  推理链长度:   avg={avg_comp_len:.0f}  min={min_comp_len}  max={max_comp_len} tokens")
    print(f"  截断率:       {truncation_rate:.2%}  ({total_truncated}/{total_samples})")
    print(f"  格式匹配率:   {parse_rate:.2%}  ({total_parsed}/{total_samples})")
    print(f"  全局可行率:   {global_feas_rate:.2%}  ({total_feasible}/{total_samples})")
    print(f"  实例可行率:   {instance_feas_rate:.2%}  ({instance_has_feas}/{num_test})")
    print(f"  最优距离均值: {avg_best_dist:.4f}  ({len(best_dists)} 个可行实例)")
    print(f"  {'─'*55}")

    # ── 输出示例 ──────────────────────────────────────────────────────
    # 优先一个可行 + 一个不可行；全可行则 2 个可行，全不可行则 2 个不可行
    examples = []
    if example_feasible and example_infeasible:
        examples = [
            ("FEASIBLE", example_feasible),
            ("INFEASIBLE", example_infeasible),
        ]
    elif example_feasible:
        second = None
        for j in range(num_test):
            if j != example_feasible[0]:
                for item in all_completions[j]:
                    comp = item[0] if isinstance(item, tuple) else item
                    if prob.is_feasible(comp, instances[j]):
                        second = (j, comp)
                        break
            if second:
                break
        examples = [("FEASIBLE #1", example_feasible)]
        if second:
            examples.append(("FEASIBLE #2", second))
    elif example_infeasible:
        second = None
        for j in range(num_test):
            if j != example_infeasible[0]:
                for item in all_completions[j]:
                    comp = item[0] if isinstance(item, tuple) else item
                    if not prob.is_feasible(comp, instances[j]):
                        second = (j, comp)
                        break
            if second:
                break
        examples = [("INFEASIBLE #1", example_infeasible)]
        if second:
            examples.append(("INFEASIBLE #2", second))

    for label, (inst_idx, comp_text) in examples:
        preview = comp_text[:500] + ("..." if len(comp_text) > 500 else "")
        print(f"\n  >>> 示例 [{label}]  (实例 #{inst_idx})")
        print(f"    {preview}")

    # 将示例也写入结果 JSON
    example_records = []
    for label, (inst_idx, comp_text) in examples:
        example_records.append({
            "label": label,
            "instance_idx": inst_idx,
            "completion": comp_text,
        })

    results = {
        "problem_type":         problem_type,
        "problem_size":         problem_size,
        "num_test":             num_test,
        "num_samples":          num_samples,
        "temperature":          temperature,
        "max_completion_length": max_completion_length,
        "batch_size":           batch_size,
        "prompt_mode":          prompt_mode,
        "avg_completion_tokens": round(avg_comp_len, 1),
        "min_completion_tokens": min_comp_len,
        "max_completion_tokens": max_comp_len,
        "truncation_rate":      round(truncation_rate, 4),
        "format_match_rate":    round(parse_rate, 4),
        "global_feasibility_rate":   round(global_feas_rate, 4),
        "instance_feasibility_rate": round(instance_feas_rate, 4),
        "avg_best_dist":        round(avg_best_dist, 4) if not np.isnan(avg_best_dist) else None,
        "feasible_instances":   instance_has_feas,
        "examples":             example_records,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="UniCOP-Reason 评估脚本")

    # ── 推理后端 ──────────────────────────────────────────────────────
    parser.add_argument("--backend",     type=str, default=config.eval_backend,
                        choices=["local", "vllm", "api"],
                        help="推理后端: local=HF本地模型 | vllm=vLLM加速 | api=API调用")

    # ── 本地模型参数 ──────────────────────────────────────────────────
    parser.add_argument("--model_path",  type=str, default=None,
                        help="本地模型路径（backend=local 时必填）")

    # ── Vertex AI Gemini 参数 ─────────────────────────────────────────
    parser.add_argument("--gcp_project", type=str, default=config.gcp_project,
                        help="GCP 项目 ID")
    parser.add_argument("--gcp_location", type=str, default=config.gcp_location,
                        help="GCP 区域，如 us-central1")
    parser.add_argument("--gcp_credentials", type=str, default=config.gcp_credentials,
                        help="服务账号 JSON 密钥文件路径")
    parser.add_argument("--api_model",   type=str, default=config.api_model,
                        help="Gemini 模型名称，如 gemini-2.5-flash / gemini-2.5-pro")
    parser.add_argument("--api_max_concurrency", type=int,
                        default=config.api_max_concurrency,
                        help="API 最大并发请求数")

    # ── 评估参数 ──────────────────────────────────────────────────────
    parser.add_argument("--problem",      type=str,   nargs="+",
                        default=[config.problem_type],
                        choices=SUPPORTED_PROBLEMS,
                        help="一个或多个问题类型")
    parser.add_argument("--problem_size", type=int,   nargs="+",
                        default=[config.problem_size],
                        help="一个或多个节点规模")
    parser.add_argument("--num_test",     type=int,   default=config.num_test,
                        help="每个 (problem, size) 组合的测试实例数")
    parser.add_argument("--num_samples",  type=int,   default=1,
                        help="每个实例的采样次数，>1 时启用随机采样")
    parser.add_argument("--temperature",  type=float, default=1.0,
                        help="采样温度，仅 num_samples>1 时生效")
    parser.add_argument("--model_type",   type=str,   default="reasoning",
                        choices=["reasoning", "instruct"],
                        help="reasoning=推理模型(10000 tokens)，instruct=指令模型(512 tokens)")
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="手动指定生成长度上限，不填则由 model_type 自动决定")
    parser.add_argument("--batch_size",   type=int,   default=1,
                        help="batch 推理大小（仅 local 模式有效）")
    parser.add_argument("--tp_size",      type=int,   default=1,
                        help="vLLM tensor parallel 卡数（仅 vllm 模式有效）")
    parser.add_argument("--prompt_mode",  type=str,   default="think",
                        choices=["think", "structured"],
                        help="提示词模式：think=自由推理 | structured=结构化逐步输出")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="重复惩罚系数，1.0=无惩罚，1.2-1.5=常用范围（仅 local 模式）")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="n-gram 硬禁：任意长度 n 的 n-gram 只要出现过就禁止其下一个 token。"
                             "0=关闭；推荐 5-7；全局生效（含 Route 输出），不需要 exempt 列表。")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="结果保存目录，不填则保存在 model_path 目录下（local）或当前目录（api）")
    args = parser.parse_args()

    # ── 参数校验 ──────────────────────────────────────────────────────
    if args.backend in ("local", "vllm") and not args.model_path:
        parser.error("backend=local/vllm 时必须通过 --model_path 指定模型路径")
    if args.backend == "api" and not args.api_model:
        parser.error("backend=api 时必须通过 --api_model 指定模型名称")
    if args.backend == "api" and not args.gcp_credentials:
        parser.error("backend=api 时必须通过 --gcp_credentials 指定服务账号密钥文件")

    # 确定 max_completion_length
    if args.max_completion_length is not None:
        max_completion_length = args.max_completion_length
    elif args.model_type == "reasoning":
        max_completion_length = 10000
    else:
        max_completion_length = 512

    prompt_mode = args.prompt_mode

    # ── 初始化推理后端 ────────────────────────────────────────────────
    if args.backend == "local":
        model, tokenizer = _load_local_model(args.model_path)

        rep_penalty = args.repetition_penalty
        no_repeat_ngram = args.no_repeat_ngram_size

        def generate_fn(prompts, num_samples, temperature, max_length, batch_size):
            return _generate_local(
                model, tokenizer, prompts, num_samples,
                temperature, max_length, batch_size, rep_penalty,
                no_repeat_ngram_size=no_repeat_ngram,
            )

        ngram_tag = f" | no_repeat_ngram={no_repeat_ngram}" if no_repeat_ngram else ""
        backend_info = f"local | {args.model_path} | rep_penalty={rep_penalty}{ngram_tag}"
    elif args.backend == "vllm":
        model, tokenizer = _load_vllm_model(args.model_path, args.tp_size)

        def generate_fn(prompts, num_samples, temperature, max_length, batch_size):
            return _generate_vllm(
                model, tokenizer, prompts, num_samples,
                temperature, max_length, batch_size,
            )

        backend_info = f"vllm | {args.model_path} (tp={args.tp_size})"
    else:
        client = _create_gemini_client(
            args.gcp_credentials, args.gcp_project, args.gcp_location,
        )

        def generate_fn(prompts, num_samples, temperature, max_length, batch_size):
            return _generate_gemini(
                client, args.api_model, prompts, num_samples,
                temperature, args.api_max_concurrency,
            )

        backend_info = f"api | Vertex AI ({args.gcp_project}/{args.gcp_location}) | model={args.api_model}"

    print(f"推理后端:  {backend_info}")
    print(f"模型类型:  {args.model_type}  提示词模式: {prompt_mode}  "
          f"max_completion_length: {max_completion_length}  batch_size: {args.batch_size}")

    # 遍历所有 (problem, size) 组合
    combos = [(p, n) for p in args.problem for n in args.problem_size]
    print(f"\n评估组合: {len(combos)} 个  {combos}")
    print(f"每组合: {args.num_test} 实例 × {args.num_samples} 采样\n")

    # ── 构建全局超参数记录 ─────────────────────────────────────────────
    # 确定模型名（用于文件命名和记录）
    if args.backend in ("local", "vllm"):
        model_label = os.path.basename(args.model_path.rstrip("/\\"))
    else:
        model_label = args.api_model

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hyperparams = {
        "backend":              args.backend,
        "model_label":          model_label,
        "prompt_mode":          prompt_mode,
        "model_type":           args.model_type,
        "max_completion_length": max_completion_length,
        "problems":             args.problem,
        "problem_sizes":        args.problem_size,
        "num_test":             args.num_test,
        "num_samples":          args.num_samples,
        "temperature":          args.temperature,
        "batch_size":           args.batch_size,
    }
    if args.backend == "local":
        hyperparams["model_path"] = args.model_path
    else:
        hyperparams["gcp_project"]        = args.gcp_project
        hyperparams["gcp_location"]       = args.gcp_location
        hyperparams["gcp_credentials"]    = args.gcp_credentials
        hyperparams["api_model"]          = args.api_model
        hyperparams["api_max_concurrency"] = args.api_max_concurrency

    all_results = []
    for problem_type, problem_size in combos:
        results = evaluate_single(
            generate_fn, problem_type, args.num_test,
            problem_size, args.num_samples, args.temperature,
            max_completion_length, args.batch_size, args.save_dir,
            prompt_mode,
        )
        results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_results.append(results)

    # ── 汇总表格 ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*98}")
        print(f"{'Problem':<10} {'n':>4}  {'AvgTok':>7}  {'Trunc%':>7}  "
              f"{'Parse%':>7}  {'GFeas%':>7}  {'IFeas%':>7}  {'AvgDist':>9}")
        print(f"{'─'*98}")
        for r in all_results:
            dist_str = f"{r['avg_best_dist']:.4f}" if r['avg_best_dist'] is not None else "N/A"
            print(f"{r['problem_type']:<10} {r['problem_size']:>4}  "
                  f"{r['avg_completion_tokens']:>7.0f}  "
                  f"{r['truncation_rate']:>7.2%}  "
                  f"{r['format_match_rate']:>7.2%}  "
                  f"{r['global_feasibility_rate']:>7.2%}  {r['instance_feasibility_rate']:>7.2%}  "
                  f"{dist_str:>9}")
        print(f"{'='*98}")

    # ── 保存结果 ──────────────────────────────────────────────────────
    if args.save_dir:
        out_dir = args.save_dir
    elif args.backend == "local":
        out_dir = args.model_path
    else:
        out_dir = "./eval_results"
    os.makedirs(out_dir, exist_ok=True)

    # 输出文件：{模型名}_{时间戳}.json，内含全部超参数 + 各组合结果
    fname = f"{model_label}_{run_timestamp}.json"
    out_path = os.path.join(out_dir, fname)

    output = {
        "hyperparams": hyperparams,
        "results":     all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
