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
import re
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from config import config
from problems import get_problem, SUPPORTED_PROBLEMS
from terminal_reward import compute_terminal_components
from utils.parse import parse_multi_route, parse_single_route


# ── Retry-until-feasible helpers ────────────────────────────────────────────
# 与 UniCOP-Reason-Mask/evaluate.py 同设计 (统一框架).
# 目的: 第一轮回答缺/重 customer 时, 把缺/重信息回灌给模型再生成一轮,
# 循环到 cov=1+无重复 或达 max_retry_rounds. 只针对 missing+duplicate
# (cap/TW/DL 违例 retry 救不了, 早停).

def _diagnose_routes(completion: str, instance: dict, prob) -> dict:
    """解析 completion 返回 cov 诊断 + strict feasibility."""
    n = instance["n"]
    if prob.multi_route:
        routes = parse_multi_route(completion, n)
    else:
        single = parse_single_route(completion, n)
        routes = [single] if single is not None else None

    if routes is None:
        return dict(parse_ok=False, missing=list(range(1, n + 1)),
                    duplicates=[], feasible_strict=False)

    seen = {}
    for r in routes:
        for v in r:
            if v == 0:
                continue
            seen[v] = seen.get(v, 0) + 1
    expected = set(range(1, n + 1))
    missing = sorted(expected - set(seen))
    duplicates = sorted(c for c, k in seen.items() if k > 1)
    feasible = prob.is_feasible(completion, instance)
    return dict(parse_ok=True, missing=missing, duplicates=duplicates,
                feasible_strict=feasible)


_OTHER_CONSTRAINT_BULLET = {
    "tsp":    None,
    "tsptw":  "All customer time windows [earliest, latest] must be respected.",
    "tspdl":  "All customer draft limits must be respected "
              "(vehicle load at arrival ≤ node draft limit).",
    "cvrp":   "Vehicle capacity must be respected "
              "(total demand per route ≤ capacity).",
    "vrptw":  "Vehicle capacity AND all customer time windows must be respected.",
    "cvrptw": "Vehicle capacity AND all customer time windows must be respected.",
}


def _build_retry_feedback(diag: dict, multi_route: bool, n: int,
                          problem_type: str) -> str:
    """
    统一框架 retry feedback (适用于 TSP/TSPTW/TSPDL/CVRP/VRPTW/CVRPTW).

    结构 (固定 4 段):
      [1] 第一句直陈: missed [...] / repeated [...] / both / parse fail
      [2] "Re-plan ... so that:" 引言
      [3] 固定 4 类 bullet: cov / depot 规则 / 其它约束 / 最小化距离
      [4] 输出格式 (multi/single 不同)
    """
    if not diag["parse_ok"]:
        problem_stmt = (
            "Your previous answer could not be parsed as routes."
            if multi_route else
            "Your previous answer could not be parsed as a route."
        )
    else:
        missed_part = (f"missed customers {diag['missing']}"
                       if diag["missing"] else None)
        dup_part = (f"visited customers {diag['duplicates']} more than once"
                    if diag["duplicates"] else None)
        if missed_part and dup_part:
            problem_stmt = f"You {missed_part} and {dup_part} in your previous answer."
        elif missed_part:
            problem_stmt = f"You {missed_part} in your previous answer."
        elif dup_part:
            problem_stmt = f"You {dup_part} in your previous answer."
        else:
            problem_stmt = "Your previous answer needs revision."

    if multi_route:
        depot_bullet = (
            "The depot 0 must be the first and last node of every route "
            "(and must not appear in the middle)."
        )
        fmt_hint = "Route 1: 0 -> ... -> 0\nRoute 2: 0 -> ... -> 0\n..."
        re_plan_verb = "Re-plan the routes"
    else:
        depot_bullet = (
            "The depot 0 must appear exactly twice: once at the start and once "
            "at the end (it must not appear in the middle of the route)."
        )
        fmt_hint = "Route: 0 -> ... -> 0"
        re_plan_verb = "Re-plan the route"

    bullets = [
        f"- Every customer in 1..{n} must be visited exactly once.",
        f"- {depot_bullet}",
    ]
    other_constraint = _OTHER_CONSTRAINT_BULLET.get(problem_type)
    if other_constraint:
        bullets.append(f"- {other_constraint}")
    bullets.append("- Total travel distance must be minimized.")

    return "\n".join([
        problem_stmt,
        "",
        f"{re_plan_verb} so that ALL of the following hold:",
        *bullets,
        "",
        "Output:",
        fmt_hint,
    ])


def _unpack_completion_item(item):
    """generate_fn 返回项归一化成 (text, truncated, num_tokens)."""
    if isinstance(item, tuple):
        text = item[0]
        truncated = item[1] if len(item) > 1 else False
        num_tokens = item[2] if len(item) > 2 else None
        return text, truncated, num_tokens
    return item, False, None


def _strip_think_for_history(text: str) -> str:
    """
    剥 <think>...</think>, 只留最终答案. 用于多轮 retry 拼 chat history.
    Qwen3-Thinking 的 chat_template 有 rolling checkpoint, 主动剥让所有 backend 一致.
    truncated 没出 </think> 时给占位避免空 message.
    """
    idx = text.rfind("</think>")
    if idx == -1:
        return "[Previous answer was truncated and did not produce a final route.]"
    tail = text[idx + len("</think>"):].lstrip("\n")
    if not tail.strip():
        return "[Previous answer reached </think> but produced no final route.]"
    return tail


def _retry_loop_one(
    generate_fn, prompt, instance, prob, problem_type: str,
    initial_item, max_completion_length, temperature,
    max_rounds: int,
):
    """单 sample retry loop. 详见 UniCOP-Reason-Mask/evaluate.py 同名函数."""
    cur_item = initial_item
    cur_text, cur_truncated, cur_tokens = _unpack_completion_item(cur_item)
    cur_prompt = list(prompt)
    rounds_used = 0
    cumulative_tokens = cur_tokens or 0

    last_diag = _diagnose_routes(cur_text, instance, prob)
    converged = last_diag["feasible_strict"]
    stopped_reason = "feasible" if converged else None

    for r in range(max_rounds):
        if last_diag["feasible_strict"]:
            stopped_reason = "feasible"
            converged = True
            break
        if (last_diag["parse_ok"]
                and not last_diag["missing"]
                and not last_diag["duplicates"]):
            stopped_reason = "cov_ok_other_violation"
            break

        feedback = _build_retry_feedback(
            last_diag, prob.multi_route,
            n=instance["n"], problem_type=problem_type,
        )
        prev_answer = _strip_think_for_history(cur_text)
        cur_prompt = cur_prompt + [
            {"role": "assistant", "content": prev_answer},
            {"role": "user",      "content": feedback},
        ]
        new_outs = generate_fn(
            [cur_prompt], 1, temperature, max_completion_length, 1,
        )
        cur_item = new_outs[0][0]
        cur_text, cur_truncated, cur_tokens = _unpack_completion_item(cur_item)
        cumulative_tokens += (cur_tokens or 0)
        rounds_used = r + 1
        last_diag = _diagnose_routes(cur_text, instance, prob)

    if last_diag["feasible_strict"]:
        converged = True
        stopped_reason = "feasible"
    elif stopped_reason is None:
        stopped_reason = "max_rounds"

    retry_info = {
        "rounds_used":      rounds_used,
        "converged":        converged,
        "stopped_reason":   stopped_reason,
        "cumulative_tokens": cumulative_tokens,
        "final_missing":    last_diag["missing"],
        "final_duplicates": last_diag["duplicates"],
        "final_parse_ok":   last_diag["parse_ok"],
    }
    return cur_item, retry_info


def _strip_think_instructions(system: str) -> str:
    """从 system prompt 中剥离 <think> 推理指令（instruct 模型不需要）。"""
    system = re.sub(
        r'Before answering, think through the problem in <think>\.\.\.</think>\.[^\n]*\n?',
        '', system,
    )
    system = system.replace("After completing your analysis, output", "Output")
    system = re.sub(r'\n{3,}', '\n\n', system).strip()
    return system


# ── 结构化提示词（仅 evaluate 使用，不影响训练） ──────────────────────────────

_STRUCTURED_SYSTEM = {
    "tsp": """You are a route planning expert solving the Travelling Salesman Problem (TSP).
Rules: Starting from node 0, visit all customer nodes exactly once and return to node 0, minimizing total distance.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these three sections in order:

1. **Strategy**: Analyze the node distribution and state your approach. Examples: "Angular sweep counterclockwise from depot", "Convex hull first then insert interior nodes", "Divide into 3 geographic segments and connect". Reference specific node groups or spatial features.

2. **Step-by-step construction**: Build the route one node at a time. Each step format:
   [step] at N → M (d=X.XXX, total=X.XX) | alt: A(X.XX), B(X.XX)
   - d = distance from current to chosen node
   - total = cumulative route distance so far
   - alt = 2-3 nearest unvisited alternatives with distances
   Every 10 steps, insert a line: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.

3. **Final route**: Write the complete route in "Route: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final route (copied from think):
Route: 0 -> A -> B -> C -> ... -> 0""",

    "tsptw": """You are a route planning expert solving the Travelling Salesman Problem with Time Windows (TSPTW).
Rules:
- Start from node 0 (depot), visit all customer nodes exactly once, and return to node 0
- Travel time between nodes = Euclidean distance
- Each customer node has a time window [earliest, latest]: arrival time must be <= latest
- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue
- Objective: minimize total travel distance

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these three sections in order:

1. **Strategy**: Analyze time windows and node positions. Identify urgent nodes (tight deadlines that must be visited early), flexible nodes (late deadlines), and the overall visit ordering principle (e.g., "deadline-driven with geographic continuity", "sweep with urgency priority"). Reference specific node IDs.

2. **Step-by-step construction**: Build the route one node at a time. Each step format:
   [step] at N → M (d=X.XXX, t=X.XX, arr=X.XX, slack=X.XX) | alt: A(X.XX,slack=X.XX), B(X.XX,slack=X.XX)
   - d = distance from current to chosen node
   - t = current time (before departing from N)
   - arr = arrival time at M
   - slack = deadline of M minus arrival time (how much time margin remains)
   - alt = 2-3 nearest feasible alternatives with distance and slack
   If arrival < earliest of M, mark "wait" and set current time to earliest.
   Every 10 steps, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.

3. **Final route**: Write the complete route in "Route: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final route (copied from think):
Route: 0 -> A -> B -> C -> ... -> 0""",

    "tspdl": """You are a logistics route planning expert solving the Travelling Salesman Problem with Draft Limits (TSPDL).
Rules:
- Start from node 0 (depot), visit all customer nodes exactly once, and return to node 0
- Travel time between nodes = Euclidean distance
- The vehicle departs fully loaded (initial load = total capacity); load decreases as cargo is unloaded at each customer
- Upon arriving at a node (before unloading), current load must be <= that node's draft_limit
- Objective: minimize total travel distance
Key insight: A node with a smaller draft_limit requires a lower load upon arrival, so it must be visited after enough cargo has already been unloaded.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these three sections in order:

1. **Strategy**: Analyze draft limits and node positions. Identify which nodes have tight draft limits (must be visited late) vs loose limits (can be visited early), and how you will balance draft-limit ordering with distance minimization. Reference specific node IDs.

2. **Step-by-step construction**: Build the route one node at a time. Each step format:
   [step] at N → M (d=X.XXX, unload=X, load:X→X, dl=X, #feas=X) | alt: A(X.XX,dl=X), B(X.XX,dl=X)
   - d = distance from current to chosen node
   - unload = cargo unloaded at M
   - load = current load before→after unloading at M
   - dl = draft_limit of M (must satisfy: load_before <= dl)
   - #feas = number of remaining unvisited nodes feasible at current load
   - alt = 2-3 nearest feasible alternatives with distance and draft_limit
   Every 10 steps, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.

3. **Final route**: Write the complete route in "Route: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final route (copied from think):
Route: 0 -> A -> B -> C -> ... -> 0""",

    "cvrp": """You are a logistics route planning expert solving the Capacitated Vehicle Routing Problem (CVRP).
Rules: Multiple vehicles depart from node 0; each vehicle visits a subset of customers and returns to node 0; total demand per route must not exceed vehicle capacity; each customer is visited exactly once; minimize total distance.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these three sections in order:

1. **Strategy**: Analyze demand distribution and node positions. Identify which nodes form each route cluster, the approximate total demand per cluster, and the visit order principle within each cluster (e.g., "sweep outward then return", "nearest-neighbor within cluster"). Reference specific node IDs.

2. **Step-by-step construction**: Build each route one node at a time. Each step format:
   [R1,3] at N → M (d=X.XXX, dem=X.XXXX) cap:X.XX→X.XX, d0=X.XX | alt: A(X.XX,cap→X.XX), B(X.XX,cap→X.XX)
   - d = distance from current to chosen node
   - dem = demand of chosen node
   - cap = remaining capacity before→after
   - d0 = distance from chosen node to depot (informs return cost)
   - alt = 2-3 nearest feasible alternatives with distance and resulting capacity
   At the start of each new route, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.
   When capacity is too low for any remaining node, return to depot and start a new route.

3. **Final routes**: Write all complete routes in "Route N: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final routes (copied from think):
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

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these three sections in order:

1. **Strategy**: Analyze time windows and node positions. Group nodes into routes by time-window compatibility and geographic proximity. For each planned route, state which nodes belong to it and the time-window range of that group. Reference specific node IDs.

2. **Step-by-step construction**: Build each route one node at a time. Each step format:
   [R1,3] at N → M (d=X.XXX, t=X.XX, arr=X.XX, slack=X.XX) | alt: A(X.XX,slack=X.XX), B(X.XX,slack=X.XX)
   - d = distance from current to chosen node
   - t = current time (before departing from N)
   - arr = arrival time at M
   - slack = deadline of M minus arrival time (how much time margin remains)
   - alt = 2-3 nearest feasible alternatives with distance and slack
   If arrival < earliest of M, mark "wait" and set current time to earliest.
   At the start of each new route, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.
   When no feasible next node exists within current route's time constraints, return to depot and start a new route.

3. **Final routes**: Write all complete routes in "Route N: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final routes (copied from think):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0""",

    "cvrptw": """You are a logistics scheduling expert solving the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
Rules:
- Multiple vehicles depart from node 0 (depot); each vehicle visits a subset of customers and returns to node 0
- All customer nodes are visited exactly once
- Travel time between nodes = Euclidean distance
- Total demand per route must not exceed vehicle capacity
- Each customer node has a time window [earliest, latest]: arrival time must be <= latest
- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue
- Objective: minimize total travel distance across all routes

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these three sections in order:

1. **Strategy**: Analyze constraints and node positions. Group nodes into routes by capacity feasibility, time-window compatibility, and geographic proximity. For each planned route, state which nodes belong to it, approximate demand sum, and time-window range. Reference specific node IDs.

2. **Step-by-step construction**: Build each route one node at a time. Each step format:
   [R1,3] at N → M (d=X.XXX, dem=X.XXXX) cap:X.XX→X.XX, arr=X.XX, slack=X.XX | alt: A(X.XX,cap→X.XX,slack=X.XX), B(X.XX,cap→X.XX,slack=X.XX)
   - d = distance from current to chosen node
   - dem = demand of chosen node
   - cap = remaining capacity before→after
   - arr = arrival time at M
   - slack = deadline of M minus arrival time
   - alt = 2-3 nearest feasible alternatives with distance, resulting capacity, and slack
   If arrival < earliest of M, mark "wait" and set current time to earliest.
   At the start of each new route, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.
   When capacity is too low or no feasible time-window node exists, return to depot and start a new route.

3. **Final routes**: Write all complete routes in "Route N: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final routes (copied from think):
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


# ── stride>1 思维链: 用与 build_think_chains 同源的 system (problems_prompt.get_system_prompt)
#    覆盖 prob.build_prompt 的 stride=1 _SYSTEM, 保证 eval prompt 与 SFT 训练数据逐字一致。 ──
_GET_SYS_PROMPT = None
def _stride_system(problem_type: str, stride: int) -> str:
    global _GET_SYS_PROMPT
    if _GET_SYS_PROMPT is None:
        import sys as _sys, os as _os
        _d = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "UniCOP-Distill"))
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        from problems_prompt import get_system_prompt as _g
        _GET_SYS_PROMPT = _g
    return _GET_SYS_PROMPT(problem_type, stride)


def _apply_stride_system(prompt: list[dict], problem_type: str, stride: int) -> list[dict]:
    """stride>1: 仅把 system 换成 SFT 训练同源的 stride 版 (问题描述/user 不变, 只改决策粒度说明)。"""
    if stride <= 1:
        return prompt
    new_sys = _stride_system(problem_type, stride)
    return [{"role": "system", "content": new_sys} if m["role"] == "system" else m
            for m in prompt]


# ── 推理后端：本地模型 ──────────────────────────────────────────────────────────

def _load_local_model(model_path: str):
    """加载本地 HuggingFace 模型和 tokenizer。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载本地模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 尊重训练阶段的 safe pad 配置 (见 train.py / train_sft.py: pad != eos),
    # 只在 tokenizer 完全没 pad_token 时才 fallback 到 eos。无条件覆盖会抹掉
    # SFT/GRPO 训练端特意避开的 "pad==eos 致 EOS 被 mask" 配置。
    if tokenizer.pad_token_id is None:
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

        # temperature > 0 → 采样模式；temperature == 0 → 贪心
        # num_samples > 1 必须采样（贪心多次结果完全相同），上游 main() 已校验
        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_completion_length,
            do_sample=do_sample,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        else:
            # 贪心时显式置空 generation_config 自带的采样字段，抑制 HF "may be ignored" 警告
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # padding_side="left" → 短 prompt 左侧补 PAD, generate 输出保留完整
        # padded input 前缀. 必须用 padded 总长度切, 否则 completion 混入
        # 残留 PAD 和 real input tokens, 导致 token 计数膨胀 / 文本污染.
        padded_input_len = inputs["input_ids"].shape[1]

        for i in range(cur_batch_size):
            for s in range(num_samples):
                idx = i * num_samples + s
                output_ids = outputs[idx]
                completion_ids = output_ids[padded_input_len:]
                # 修正(2026-04-21): 原做法 mask = (ids != pad_token_id) + 删所有 pad
                #   在 R1-Distill 默认 pad == eos 情况下会把"真实生成的 EOS"误删,
                #   导致 num_tokens 少算 1, truncation 判定失准, decode 仍完整。
                # 新做法: pad != eos 时原逻辑(删所有 pad); pad == eos 时保留第一个 eos,
                #   仅 trim 第一个 eos 之后的连续填充。
                pad_id = tokenizer.pad_token_id
                eos_id = tokenizer.eos_token_id
                if pad_id != eos_id:
                    completion_ids = completion_ids[completion_ids != pad_id]
                else:
                    eos_positions = (completion_ids == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        first_eos = eos_positions[0].item()
                        completion_ids = completion_ids[:first_eos + 1]
                # 标记是否被截断（token 数达到 max_completion_length）
                num_tokens = len(completion_ids)
                is_truncated = (num_tokens >= max_completion_length)
                # skip_special_tokens=False: 关键! Qwen3-Thinking 把 <think>/</think>
                # 注册为 special token (id 151667/151668), True 会把它们剥掉, 后续
                # rfind("</think>") 失败导致 thinking 段被算进答案。R1-Distill 上
                # <think>/</think> 是普通 BPE, 不受 skip_special_tokens 影响, 两者兼容。
                # <|im_end|> / <|endoftext|> 等结构 token 通过下面的字符串 replace 去掉。
                completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
                # 去掉结构性 special token (保留 <think>/</think> 用于后续解析)
                for tok in ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>",
                            "<|begin_of_text|>", "<|eot_id|>"):
                    completion = completion.replace(tok, "")
                all_completions[batch_start + i].append((completion, is_truncated, num_tokens))

    return all_completions


# ── 推理后端：vLLM ─────────────────────────────────────────────────────────────

def _load_vllm_model(model_path: str, tensor_parallel_size: int = 1,
                     gpu_mem_util: float = 0.9):
    """
    加载 vLLM 模型。
    如果 model_path 是 LoRA adapter 目录（含 adapter_config.json 但无 config.json），
    则自动合并 LoRA 到基座模型后再加载。
    gpu_mem_util: vLLM 预留显存比例。同 GPU 还要跑 POMO PRM(--wave) 时调低(如 0.8)留地方。
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
        gpu_memory_utilization=gpu_mem_util,
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

    # temperature > 0 → 采样；temperature == 0 → 贪心（vLLM 原生语义）
    # skip_special_tokens=False: 关键! 与 _generate_local 对齐。Qwen3-Thinking 把
    # <think>/</think> 注册为 special token(id 151667/151668), vLLM 默认 True 会剥掉,
    # 导致下游 rfind("</think>") 失败、think 段被算进答案、解析与 local 后端不一致。
    sampling_params = SamplingParams(
        max_tokens=max_completion_length,
        temperature=temperature,
        n=num_samples,
        skip_special_tokens=False,
    )

    outputs = model.generate(chat_texts, sampling_params)

    all_completions = [[] for _ in range(len(prompts))]
    for i, output in enumerate(outputs):
        for sample in output.outputs:
            completion = sample.text
            # 与 _generate_local 一致: 去掉结构性 special token, 保留 <think>/</think>
            for tok in ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>",
                        "<|begin_of_text|>", "<|eot_id|>"):
                completion = completion.replace(tok, "")
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
                    prompt_mode: str = "think",
                    model_type: str = "reasoning",
                    stride: int = 1,
                    retry_until_feasible: bool = False,
                    max_retry_rounds: int = 3,
                    wave_cfg=None, prm=None, wave_tokenizer=None,
                    run_bestofn: bool = False):
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
    all_coverage     = []
    all_constraint   = []

    # 收集示例:按 "解析成功 / 解析失败" 各多条备选,最终各挑 3 个,
    # 不够的一类用另一类补,总计 6 个。便于定位 parse 逻辑是否出问题。
    MAX_COLLECT_PER_CLASS = 12       # 每类最多收集多少条候选(比最终 3 个多,给挑选留空间)
    parsed_samples   = []            # list of (instance_idx, completion_text)
    unparsed_samples = []

    # 预生成所有实例和 prompt
    instances = []
    prompts = []
    for _ in range(num_test):
        instance = prob.generate_instance(problem_size, rng)
        prompt   = prob.build_prompt(instance)

        # instruct 模型：剥离 system prompt 中的 <think> 指令
        if model_type == "instruct":
            for msg in prompt:
                if msg["role"] == "system":
                    msg["content"] = _strip_think_instructions(msg["content"])

        # structured 模式：替换 system prompt 和 user 末尾输出格式
        if prompt_mode == "structured":
            prompt = _apply_structured_prompt(prompt, problem_type)

        # stride>1: 用 SFT 训练同源的 stride 版 system 覆盖 (与训练数据逐字一致)
        if stride > 1:
            prompt = _apply_stride_system(prompt, problem_type, stride)

        instances.append(instance)
        prompts.append(prompt)

    # 调用推理后端
    print(f"[{problem_type.upper()} n={problem_size}] 生成 {num_test} 实例 × {num_samples} 采样 ...")
    all_completions = generate_fn(
        prompts, num_samples, temperature, max_completion_length, batch_size,
    )

    # 评估每个实例的所有 completion
    total_truncated = 0
    hlr_extras: list[dict] = []   # backend=hlr 时, 收集每条 sample 的 extra info
    retry_records: list[dict] = []   # retry_until_feasible 时收集
    for i in range(num_test):
        instance = instances[i]
        instance_best = None

        for s_idx, item in enumerate(all_completions[i]):
            # ── Retry loop: 第一轮诊断, 缺/重时回灌 missing/dup 再生 ──
            if retry_until_feasible:
                item, retry_info = _retry_loop_one(
                    generate_fn, prompts[i], instance, prob, problem_type,
                    initial_item=item,
                    max_completion_length=max_completion_length,
                    temperature=temperature,
                    max_rounds=max_retry_rounds,
                )
                retry_records.append(retry_info)

            # 兼容四种格式:
            #   (completion, is_truncated, num_tokens, extra_info)  — hlr 后端 (4 元组)
            #   (completion, is_truncated, num_tokens)              — local / vllm 后端
            #   (completion, is_truncated)                          — 旧格式兼容
            #   纯 str                                              — API 后端
            if isinstance(item, tuple):
                completion = item[0]
                is_truncated = item[1]
                num_tokens = item[2] if len(item) > 2 else None
                if len(item) > 3 and item[3] is not None:
                    hlr_extras.append(item[3])
            else:
                completion, is_truncated, num_tokens = item, False, None

            total_samples += 1
            if is_truncated:
                total_truncated += 1
            comp_len = num_tokens if num_tokens is not None else len(completion)
            completion_lens.append(comp_len)

            tc = compute_terminal_components(completion, instance, problem_type)
            all_coverage.append(tc["coverage"])
            all_constraint.append(tc["constraint"])

            dist = prob.get_tour_distance(completion, instance)
            parsed = dist is not None
            if parsed:
                total_parsed += 1
                if len(parsed_samples) < MAX_COLLECT_PER_CLASS:
                    parsed_samples.append((i, completion, is_truncated, comp_len))
            else:
                if len(unparsed_samples) < MAX_COLLECT_PER_CLASS:
                    unparsed_samples.append((i, completion, is_truncated, comp_len))

            feasible = prob.is_feasible(completion, instance)
            if feasible:
                total_feasible += 1
                if dist is not None:
                    if instance_best is None or dist < instance_best:
                        instance_best = dist

        if instance_best is not None:
            instance_has_feas += 1
            best_dists.append(instance_best)

    # ── 计算指标 ──────────────────────────────────────────────────────
    parse_rate         = total_parsed / total_samples if total_samples else 0
    coverage_rate      = float(np.mean(all_coverage)) if all_coverage else 0.0
    constraint_rate    = float(np.mean(all_constraint)) if all_constraint else 0.0
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
    print(f"  覆盖率:       {coverage_rate:.2%}")
    print(f"  约束满足率:   {constraint_rate:.4f}")
    print(f"  全局可行率:   {global_feas_rate:.2%}  ({total_feasible}/{total_samples})")
    print(f"  实例可行率:   {instance_feas_rate:.2%}  ({instance_has_feas}/{num_test})")
    print(f"  最优距离均值: {avg_best_dist:.4f}  ({len(best_dists)} 个可行实例)")
    print(f"  {'─'*55}")

    # ── 覆盖完整样本的约束满足分布分析 ────────────────────────────────
    cov_arr = np.array(all_coverage)
    con_arr = np.array(all_constraint)
    covered_mask = cov_arr == 1.0
    n_covered = int(covered_mask.sum())
    if n_covered > 0:
        con_of_covered = con_arr[covered_mask]
        print(f"\n  {'─'*55}")
        print(f"  覆盖完整样本的约束满足分布 ({n_covered} 个样本):")
        print(f"    约束满足率 = 1.0 (全部路线合法): {int((con_of_covered == 1.0).sum())} 个")
        for lo, hi, label in [
            (0.8, 1.0, "[0.8, 1.0)  差1-2条路线"),
            (0.6, 0.8, "[0.6, 0.8)  约2/5路线违约"),
            (0.4, 0.6, "[0.4, 0.6)  约半数路线违约"),
            (0.0, 0.4, "[0.0, 0.4)  大部分路线违约"),
        ]:
            cnt = int(((con_of_covered >= lo) & (con_of_covered < hi)).sum())
            print(f"    约束满足率 ∈ {label}: {cnt} 个")
        print(f"    均值: {con_of_covered.mean():.4f}  中位数: {np.median(con_of_covered):.4f}")
        print(f"    最小值: {con_of_covered.min():.4f}  最大值: {con_of_covered.max():.4f}")
        # 按路线数细分：约束=valid/total，反推违约路线数
        print(f"\n    违约路线数分布 (覆盖完整的 {n_covered} 个样本):")
        violation_counts = {}
        for c in con_of_covered:
            if c == 1.0:
                n_violated = 0
            elif c == 0.0:
                n_violated = -1  # 无法推断
            else:
                # c = valid/total → total = round(1/(1-c)) 近似不准，直接用分数反推
                # 尝试 total_routes = 2..10，找最接近整数的
                best_total, best_violated = None, None
                for t in range(2, 15):
                    valid = c * t
                    if abs(valid - round(valid)) < 0.01:
                        best_total = t
                        best_violated = t - int(round(valid))
                        break
                n_violated = best_violated if best_violated is not None else -1
            violation_counts[n_violated] = violation_counts.get(n_violated, 0) + 1
        for k in sorted(violation_counts.keys()):
            label = f"{k} 条路线超载" if k >= 0 else "无法推断"
            print(f"      {label}: {violation_counts[k]} 个样本")
        print(f"  {'─'*55}")

    # ── 输出示例: parse 成功 / 失败 各 3 个, 不够用另一边补 ─────────
    TARGET_EACH = 3
    TOTAL_TARGET = TARGET_EACH * 2

    n_parsed_pick   = min(TARGET_EACH, len(parsed_samples))
    n_unparsed_pick = min(TARGET_EACH, len(unparsed_samples))
    # 差多少从另一边补
    deficit = TOTAL_TARGET - n_parsed_pick - n_unparsed_pick
    if deficit > 0:
        if n_parsed_pick < TARGET_EACH and len(unparsed_samples) > n_unparsed_pick:
            add = min(deficit, len(unparsed_samples) - n_unparsed_pick)
            n_unparsed_pick += add
        elif n_unparsed_pick < TARGET_EACH and len(parsed_samples) > n_parsed_pick:
            add = min(deficit, len(parsed_samples) - n_parsed_pick)
            n_parsed_pick += add

    examples = []
    for k in range(n_parsed_pick):
        examples.append((f"PARSED #{k + 1}", parsed_samples[k]))
    for k in range(n_unparsed_pick):
        examples.append((f"UNPARSED #{k + 1}", unparsed_samples[k]))

    FULL_DISPLAY_CHAR_LIMIT = 8000

    def _focused_preview(comp_text: str, truncated: bool) -> str:
        if not truncated and len(comp_text) <= FULL_DISPLAY_CHAR_LIMIT:
            return comp_text
        head_max = 300
        tail_max = 1500
        think_end_idx = comp_text.rfind("</think>")
        if think_end_idx == -1:
            if len(comp_text) <= head_max + tail_max:
                return comp_text
            return comp_text[:head_max] + "\n    ...[middle omitted]...\n" + comp_text[-tail_max:]
        after_think = comp_text[think_end_idx:]
        if think_end_idx <= head_max:
            return comp_text[:think_end_idx + len(after_think)]
        return (comp_text[:head_max]
                + f"\n    ...[think middle omitted, {think_end_idx - head_max} chars]...\n"
                + after_think)

    for label, (inst_idx, comp_text, trunc, n_tok) in examples:
        trunc_tag = "TRUNCATED" if trunc else "COMPLETE"
        preview = _focused_preview(comp_text, trunc)
        print(f"\n  >>> 示例 [{label}]  (实例 #{inst_idx}, {n_tok} tokens, {trunc_tag})")
        print(f"    {preview}")

    example_records = []
    for label, (inst_idx, comp_text, trunc, n_tok) in examples:
        example_records.append({
            "label": label,
            "instance_idx": inst_idx,
            "num_tokens": n_tok,
            "is_truncated": trunc,
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
        "coverage_rate":        round(coverage_rate, 4),
        "constraint_rate":      round(constraint_rate, 4),
        "global_feasibility_rate":   round(global_feas_rate, 4),
        "instance_feasibility_rate": round(instance_feas_rate, 4),
        "avg_best_dist":        round(avg_best_dist, 4) if not np.isnan(avg_best_dist) else None,
        "feasible_instances":   instance_has_feas,
        "examples":             example_records,
    }

    # HLR backend 汇总 (per-sample extras → per-combo summary).
    # 注: 真正的"节省"指标要跟 baseline 比, 见 Latent-SFT/eval_hlr_compare.py;
    # 这里只汇报 HLR 本身的计数, 不算"节省比例"避免自比误导.
    if hlr_extras:
        sum_explicit = sum(e.get("explicit_tokens", 0) for e in hlr_extras)
        sum_latent = sum(e.get("latent_steps", 0) for e in hlr_extras)
        sum_segs = sum(len(e.get("latent_segments", [])) for e in hlr_extras)
        sum_wall = sum(e.get("wall_time_sec", 0.0) for e in hlr_extras)
        n = len(hlr_extras)
        results["hlr_summary"] = {
            "samples":                  n,
            "avg_explicit_tokens":      round(sum_explicit / n, 2),
            "avg_latent_steps":         round(sum_latent / n, 2),
            "avg_latent_segments":      round(sum_segs / n, 2),
            "avg_wall_time_sec":        round(sum_wall / n, 3),
            "total_wall_time_sec":      round(sum_wall, 2),
        }

    # ── Retry-until-feasible 汇总 ─────────────────────────────────────
    if retry_until_feasible and retry_records:
        rounds_arr = [r["rounds_used"] for r in retry_records]
        converged = sum(1 for r in retry_records if r["converged"])
        n_r = len(retry_records)
        rounds_dist = {k: rounds_arr.count(k) for k in sorted(set(rounds_arr))}
        cumulative_tokens = [r["cumulative_tokens"] for r in retry_records]
        avg_cum = sum(cumulative_tokens) / n_r if n_r else 0.0
        reasons = {}
        for r in retry_records:
            reasons[r["stopped_reason"]] = reasons.get(r["stopped_reason"], 0) + 1

        results["retry_summary"] = {
            "enabled":              True,
            "max_retry_rounds":     max_retry_rounds,
            "samples":              n_r,
            "converged":            converged,
            "converged_rate":       round(converged / n_r, 4) if n_r else 0.0,
            "avg_rounds_used":      round(sum(rounds_arr) / n_r, 3) if n_r else 0.0,
            "max_rounds_used":      max(rounds_arr) if rounds_arr else 0,
            "rounds_distribution":  rounds_dist,
            "stopped_reasons":      reasons,
            "avg_cumulative_tokens": round(avg_cum, 1),
        }
        print(f"\n  {'─'*55}")
        print(f"  Retry-until-feasible 汇总 (max_rounds={max_retry_rounds}):")
        print(f"    收敛比例:       {converged}/{n_r} = {100*converged/n_r:.2f}%")
        print(f"    平均 retry 轮:  {sum(rounds_arr)/n_r:.2f} (max={max(rounds_arr)})")
        print(f"    轮数分布:       {rounds_dist}")
        print(f"    终止原因分布:   {reasons}")
        print(f"    平均累计 token: {avg_cum:.0f}")
        print(f"  {'─'*55}")
    elif retry_until_feasible:
        results["retry_summary"] = {"enabled": True, "samples": 0}

    # ── 朴素 best-of-N scaling 曲线 (POMO-free 对照基线) ──────────────────
    # 详见 bestofn_eval.py. 给出 best-of-k (k=1..N) 的 (算力, 质量) 曲线, 作为
    # wave 要打赢的基线. 不依赖 POMO, 仅需 tokenizer + prob.
    if run_bestofn and wave_tokenizer is not None:
        from bestofn_eval import bestofn_replay

        def _ctext_b(it):
            return it[0] if isinstance(it, tuple) else it

        bon_completions = [[_ctext_b(it) for it in all_completions[i]]
                           for i in range(num_test)]
        bon = bestofn_replay(bon_completions, instances, prob, wave_tokenizer)
        results["bestofn"] = bon

        def _fb(x):
            return f"{x:.4f}" if x is not None else "N/A"

        print(f"\n  {'─'*55}")
        print(f"  best-of-N scaling 曲线 (N={bon['N']}, "
              f"~{bon['mean_tokens_per_sample']:.0f} tok/样本, 全量算力={bon['total_tokens']}):")
        for pt in bon["scaling_curve"]:
            if pt["k"] in (1, 2, 4, 8, 16, 32, 64, bon["N"]):
                print(f"    k={pt['k']:>3}  算力={pt['compute']:>11.0f}  "
                      f"best_dist={_fb(pt['avg_best_dist'])}  feas={pt['feas_rate']:.2%}")
        print(f"  {'─'*55}")

    # ── 波次式 (successive-halving) best-of-N 离线回放 ────────────────────
    # 用 POMO PRM 在 1/4 客户检查点剪枝, 与朴素 best-of-N 在【同算力(token)】下对比.
    # 详见 wave_replay.py 模块 docstring. 仅 local/vllm + 提供 prm/tokenizer 时启用.
    if wave_cfg is not None and prm is not None and wave_tokenizer is not None:
        from wave_replay import wave_replay

        def _ctext(it):
            return it[0] if isinstance(it, tuple) else it

        wave_completions = [[_ctext(it) for it in all_completions[i]]
                            for i in range(num_test)]
        wave_out = wave_replay(
            wave_completions, instances, prob, prm, wave_tokenizer,
            problem_type, wave_cfg,
        )
        results["wave"] = {
            "n_instances":            wave_out["n_instances"],
            "checkpoint_fracs":       list(wave_cfg.checkpoint_fracs),
            "halve_fracs":            list(wave_cfg.halve_fracs),
            "keep_fraction":          wave_cfg.keep_fraction,
            "wave_C_total":           wave_out["wave_C_total"],
            "baseline_C_total":       wave_out["baseline_C_total"],
            "compute_saving_ratio":   wave_out["compute_saving_ratio"],
            "wave_avg_best_dist":     wave_out["wave_avg_best_dist"],
            "baseline_avg_best_dist": wave_out["baseline_avg_best_dist"],
            "baseline_avg_best_dist_at_wave_C": wave_out["baseline_avg_best_dist_at_wave_C"],
            "per_instance":           wave_out["per_instance"],
        }
        w = results["wave"]
        save_str = (f"{w['compute_saving_ratio']:.1%}"
                    if w["compute_saving_ratio"] is not None else "N/A")

        def _f(x):
            return f"{x:.4f}" if x is not None else "N/A"

        print(f"\n  {'─'*55}")
        print(f"  波次式回放 (检查点={list(wave_cfg.checkpoint_fracs)}, "
              f"halve@{list(wave_cfg.halve_fracs)}, keep={wave_cfg.keep_fraction}):")
        print(f"    算力(token): wave={w['wave_C_total']}  "
              f"baseline={w['baseline_C_total']}  省={save_str}")
        print(f"    最优距离:    wave={_f(w['wave_avg_best_dist'])}  "
              f"baseline(全量)={_f(w['baseline_avg_best_dist'])}  "
              f"baseline@同算力={_f(w['baseline_avg_best_dist_at_wave_C'])}")
        print(f"    → 同算力对比: wave 的距离应 ≤ baseline@同算力 才算赢")
        print(f"  {'─'*55}")

    return results


def main():
    parser = argparse.ArgumentParser(description="UniCOP-Reason 评估脚本")

    # ── 推理后端 ──────────────────────────────────────────────────────
    parser.add_argument("--backend",     type=str, default=config.eval_backend,
                        choices=["local", "vllm", "api", "hlr"],
                        help="推理后端: local=HF本地模型 | vllm=vLLM加速 | "
                             "api=API调用 | hlr=Latent-SFT HLR 引擎")

    # ── 本地模型参数 ──────────────────────────────────────────────────
    parser.add_argument("--model_path",  type=str, default=None,
                        help="本地模型路径（backend=local/vllm 时必填）")

    # ── HLR backend 参数 ─────────────────────────────────────────────
    parser.add_argument("--hlr_checkpoint", type=str, default=None,
                        help="HLR checkpoint 目录 (含 latent_reasoner.pt + adapter_config.json), "
                             "backend=hlr 时必填")
    parser.add_argument("--hlr_base_model", type=str, default=None,
                        help="HLR base 模型路径 (不传则从 adapter_config.json 读)")
    parser.add_argument("--hlr_merge_lora", action="store_true",
                        help="HLR 加载时 merge LoRA, 推理稍快但显存稍多")

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
                        help="每个实例的采样次数；>1 时必须配合 temperature>0")
    parser.add_argument("--temperature",  type=float, default=0.0,
                        help="采样温度。>0 → 采样（推荐 reasoning 模型用 0.6）；"
                             "=0 → 贪心解码（deterministic）。"
                             "num_samples>1 时必须 >0。")
    parser.add_argument("--model_type",   type=str,   default="reasoning",
                        choices=["reasoning", "instruct"],
                        help="reasoning=推理模型(10000 tokens)，instruct=指令模型(512 tokens)")
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="手动指定生成长度上限，不填则由 model_type 自动决定")
    parser.add_argument("--batch_size",   type=int,   default=1,
                        help="batch 推理大小（仅 local 模式有效）")
    parser.add_argument("--tp_size",      type=int,   default=1,
                        help="vLLM tensor parallel 卡数（仅 vllm 模式有效）")
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.9,
                        help="vLLM 显存预留比例（仅 vllm）。同 GPU 还要跑 POMO PRM(--wave) 时调低(如 0.8)。")
    parser.add_argument("--prompt_mode",  type=str,   default="think",
                        choices=["think", "structured"],
                        help="提示词模式：think=自由推理 | structured=结构化逐步输出")
    parser.add_argument("--stride", type=int, default=1,
                        help="思维链决策粒度 (1=逐点, 5=每5点一决策)。必须与 SFT 训练数据的 stride 一致, "
                             "否则 eval prompt 的 system 与训练数据不符 (train/eval 失配)。")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="重复惩罚系数，1.0=无惩罚，1.2-1.5=常用范围（仅 local 模式）")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="n-gram 硬禁：任意长度 n 的 n-gram 只要出现过就禁止其下一个 token。"
                             "0=关闭；推荐 5-7；全局生效（含 Route 输出），不需要 exempt 列表。")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="结果保存目录，不填则保存在 model_path 目录下（local）或当前目录（api）")
    # ── Retry-until-feasible ──────────────────────────────────────────
    parser.add_argument("--retry_until_feasible", action="store_true",
                        help="第一轮回答缺/重 customer 时, 把缺/重信息回灌给模型重新生成, "
                             "循环到 cov=1+无重复 或达 --max_retry_rounds. 只对 missing/dup 起效, "
                             "不修 cap/TW/DL 违例 (retry 救不了).")
    parser.add_argument("--max_retry_rounds", type=int, default=3,
                        help="--retry_until_feasible 时最多 retry 几轮 (第一轮不算 retry). 默认 3, "
                             "总轮数 = 1 + max_retry_rounds.")
    # ── 波次式 best-of-N 离线回放 (用 POMO PRM 在 1/4 检查点剪枝) ───────
    parser.add_argument("--bestofn", action="store_true",
                        help="输出朴素 best-of-N scaling 曲线 (best-of-k, k=1..N 的算力-质量曲线), "
                             "作为 wave 的对照基线. 不依赖 POMO, 仅 local/vllm (需 tokenizer). "
                             "可与 --wave 同时开 (基线 + 剪枝同图对比).")
    parser.add_argument("--wave", action="store_true",
                        help="开启波次式 best-of-N 离线回放: 生成完整链后用 POMO PRM 在 1/4 "
                             "客户检查点回放剪枝, 与朴素 best-of-N 在同算力(token)下对比. "
                             "需 --num_samples>1 + --pomo_ckpt_dir/--pomo_baseline_dir, 仅 local/vllm.")
    parser.add_argument("--pomo_ckpt_dir", type=str, default="",
                        help="POMO checkpoint 根目录 (子目录 {ts}__POMO_{TYPE}_n{N}/MODEL_FINAL.pt). --wave 时必填.")
    parser.add_argument("--pomo_baseline_dir", type=str, default="",
                        help="POMO-Baseline 项目根目录 (导入模型/环境代码). --wave 时必填.")
    parser.add_argument("--wave_keep_frac", type=float, default=0.5,
                        help="波次式每个 halve 检查点保留比例 (默认 0.5 = 留一半).")
    parser.add_argument("--wave_checkpoint_fracs", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="检查点比例 (默认 1/4 网格). 最后一个应为 1.0 (终点选择).")
    parser.add_argument("--wave_halve_fracs", type=float, nargs="+",
                        default=[0.5, 0.75],
                        help="哪些检查点做 POMO 排名淘汰 (方案A: 仅 50%/75%, 25% 只硬过滤).")
    parser.add_argument("--wave_device", type=str, default="cuda",
                        help="POMO PRM 运行设备 (默认 cuda).")
    args = parser.parse_args()

    # ── 参数校验 ──────────────────────────────────────────────────────
    if args.backend in ("local", "vllm") and not args.model_path:
        parser.error("backend=local/vllm 时必须通过 --model_path 指定模型路径")
    if args.backend == "api" and not args.api_model:
        parser.error("backend=api 时必须通过 --api_model 指定模型名称")
    if args.backend == "api" and not args.gcp_credentials:
        parser.error("backend=api 时必须通过 --gcp_credentials 指定服务账号密钥文件")
    if args.backend == "hlr" and not args.hlr_checkpoint:
        parser.error("backend=hlr 时必须通过 --hlr_checkpoint 指定 HLR checkpoint 目录")
    if args.backend == "hlr" and args.num_samples != 1:
        parser.error(f"backend=hlr 当前只支持 num_samples=1, 当前 num_samples={args.num_samples}")
    if args.num_samples > 1 and args.temperature <= 0:
        parser.error(f"num_samples={args.num_samples}>1 但 temperature={args.temperature}<=0；"
                     "贪心解码多次结果完全相同，请设 --temperature 0.6 或类似值")
    if args.wave:
        if args.backend not in ("local", "vllm"):
            parser.error("--wave 仅支持 backend=local/vllm (需 POMO PRM + tokenizer)")
        if not args.pomo_ckpt_dir or not args.pomo_baseline_dir:
            parser.error("--wave 必须同时指定 --pomo_ckpt_dir 和 --pomo_baseline_dir")
        if args.num_samples <= 1:
            print("⚠️ --wave 但 num_samples<=1, 波次式无样本池可剪枝(退化). "
                  "建议 --num_samples 16 或更多.", flush=True)
    if args.bestofn and args.backend not in ("local", "vllm"):
        parser.error("--bestofn 仅支持 backend=local/vllm (需 tokenizer)")

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
        model, tokenizer = _load_vllm_model(args.model_path, args.tp_size,
                                            gpu_mem_util=args.vllm_gpu_mem_util)

        def generate_fn(prompts, num_samples, temperature, max_length, batch_size):
            return _generate_vllm(
                model, tokenizer, prompts, num_samples,
                temperature, max_length, batch_size,
            )

        backend_info = f"vllm | {args.model_path} (tp={args.tp_size})"
    elif args.backend == "hlr":
        # Lazy import: Latent-SFT 不在 UniCOP-Reason 的默认 sys.path.
        # 注意 module 名: 用 hlr_config (不是 config), 避免与 UniCOP-Reason/config
        # 的 sys.modules cache 冲突 (UniCOP-Reason/evaluate.py 顶部已 import config).
        import sys
        from pathlib import Path
        _repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(_repo_root / "Latent-SFT"))
        from inference import HLRInferenceEngine
        from hlr_config import HLRConfig

        hlr_engine = HLRInferenceEngine(
            checkpoint_dir=args.hlr_checkpoint,
            base_model_path=args.hlr_base_model,
            cfg=HLRConfig(),
            merge_lora=args.hlr_merge_lora,
        )

        def generate_fn(prompts, num_samples, temperature, max_length, batch_size):
            # HLR engine 支持 batched generate. num_samples=1 强制 (上游已校验).
            # batch_size 控制 chunk 大小, 与 baseline (local) batch_size 参数语义一致.
            all_out: list[list[tuple]] = []
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start:start + batch_size]
                batch_results = hlr_engine.generate_batch(
                    chunk,
                    max_new_tokens=max_length,
                    temperature=temperature,
                )
                for text, info in batch_results:
                    info["compression_ratio"] = hlr_engine.compression_ratio
                    # num_tokens 用 total_equivalent_tokens, 让 avg_completion_tokens
                    # 与 baseline (local) 公平对比
                    num_tokens = info["total_equivalent_tokens"]
                    all_out.append([(text, info["truncated"], num_tokens, info)])
            return all_out

        backend_info = (f"hlr | ckpt={args.hlr_checkpoint} "
                        f"(base={args.hlr_base_model or 'auto'}, merge={args.hlr_merge_lora})")
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

    # ── 波次式回放 PRM 初始化 (仅 --wave 时; tokenizer 来自 local/vllm 后端) ──
    wave_cfg = None
    wave_prm = None
    if args.wave:
        from pomo_prm import POMOPRM
        from wave_replay import WaveConfig
        wave_cfg = WaveConfig(
            checkpoint_fracs=tuple(args.wave_checkpoint_fracs),
            halve_fracs=tuple(args.wave_halve_fracs),
            keep_fraction=args.wave_keep_frac,
        )
        wave_prm = POMOPRM(
            pomo_ckpt_dir=args.pomo_ckpt_dir,
            pomo_baseline_dir=args.pomo_baseline_dir,
            device=args.wave_device,
        )
        print(f"波次式回放: 开启  checkpoints={list(wave_cfg.checkpoint_fracs)}  "
              f"halve@{list(wave_cfg.halve_fracs)}  keep={wave_cfg.keep_fraction}")

    # 遍历所有 (problem, size) 组合
    combos = [(p, n) for p in args.problem for n in args.problem_size]
    print(f"\n评估组合: {len(combos)} 个  {combos}")
    print(f"每组合: {args.num_test} 实例 × {args.num_samples} 采样\n")

    # ── 构建全局超参数记录 ─────────────────────────────────────────────
    # 确定模型名（用于文件命名和记录）
    if args.backend in ("local", "vllm"):
        model_label = os.path.basename(args.model_path.rstrip("/\\"))
    elif args.backend == "hlr":
        model_label = "hlr_" + os.path.basename(args.hlr_checkpoint.rstrip("/\\"))
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
        "retry_until_feasible": args.retry_until_feasible,
        "max_retry_rounds":     args.max_retry_rounds if args.retry_until_feasible else None,
    }
    if args.backend in ("local", "vllm"):
        hyperparams["model_path"] = args.model_path
    elif args.backend == "hlr":
        hyperparams["hlr_checkpoint"] = args.hlr_checkpoint
        hyperparams["hlr_base_model"] = args.hlr_base_model
        hyperparams["hlr_merge_lora"] = args.hlr_merge_lora
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
            prompt_mode, args.model_type, stride=args.stride,
            retry_until_feasible=args.retry_until_feasible,
            max_retry_rounds=args.max_retry_rounds,
            wave_cfg=wave_cfg, prm=wave_prm,
            wave_tokenizer=(tokenizer if (args.wave or args.bestofn) else None),
            run_bestofn=args.bestofn,
        )
        results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_results.append(results)

    # ── 汇总表格 ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*112}")
        print(f"{'Problem':<10} {'n':>4}  {'AvgTok':>7}  {'Trunc%':>7}  "
              f"{'Parse%':>7}  {'Cov%':>7}  {'Constr':>7}  "
              f"{'GFeas%':>7}  {'IFeas%':>7}  {'AvgDist':>9}")
        print(f"{'─'*112}")
        for r in all_results:
            dist_str = f"{r['avg_best_dist']:.4f}" if r['avg_best_dist'] is not None else "N/A"
            print(f"{r['problem_type']:<10} {r['problem_size']:>4}  "
                  f"{r['avg_completion_tokens']:>7.0f}  "
                  f"{r['truncation_rate']:>7.2%}  "
                  f"{r['format_match_rate']:>7.2%}  "
                  f"{r['coverage_rate']:>7.2%}  "
                  f"{r['constraint_rate']:>7.4f}  "
                  f"{r['global_feasibility_rate']:>7.2%}  {r['instance_feasibility_rate']:>7.2%}  "
                  f"{dist_str:>9}")
        print(f"{'='*112}")

    # ── 保存结果 ──────────────────────────────────────────────────────
    if args.save_dir:
        out_dir = args.save_dir
    elif args.backend == "local":
        out_dir = args.model_path
    elif args.backend == "hlr":
        out_dir = args.hlr_checkpoint
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
