"""检测 rationalization prompt 的 token 长度，确认 max-model-len 设置是否安全。

从 log 中提取的真实 CVRP20 prompt 模板构造样本，用 transformers tokenizer 精确计数。
如果本地没有 tokenizer，用字符数近似估算。

用法：python check_prompt_length.py
"""

import random

# ── 从 rationalize_solutions.py 复制的模板 ─────────────────────────────────
_POSTHOC_SUFFIX = (
    "\n\nYour output MUST start with <think> and follow this exact structure:\n\n"
    "<think>\n[your reasoning here]\n</think>\n[solution in required format]\n\n"
    "Rules:\n"
    "1. Your FIRST token MUST be '<think>'. Do NOT output anything before <think>.\n"
    "2. In <think>, show your step-by-step decision process for "
    "constructing the route from scratch. At each step, state where you are, "
    "which nearby nodes are candidates, and why you pick the next one "
    "(e.g. nearest distance, tightest time window, capacity constraint). "
    "Write as if you are solving this problem yourself for the first time.\n"
    "3. Keep <think> concise (a few hundred words at most). "
    "Do NOT mention that a solution was provided or given to you. "
    "Do NOT describe your task as 'reconstructing', 'explaining', or 'justifying' a solution. "
    "You are solving this problem from scratch — your reasoning should read as original problem-solving, "
    "not as post-hoc analysis of a known answer.\n"
    "4. After </think>, output the solution exactly in the required format.\n"
    "5. Do NOT output the solution before <think>. The solution ONLY appears after </think>."
)

# ── 从 log 中提取的真实 system prompt ──────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a logistics route planning expert solving the "
    "Capacitated Vehicle Routing Problem (CVRP).\n"
    "Rules: Multiple vehicles depart from node 0; each vehicle visits a subset "
    "of customers and returns to node 0; total demand per route must not exceed "
    "vehicle capacity; each customer is visited exactly once; minimize total distance.\n"
    "Output in the following format (one route per line, nodes in visit order):\n"
    "Route 1: 0 -> node -> ... -> 0\n"
    "Route 2: 0 -> node -> ... -> 0"
)


def fake_cvrp20_instance():
    """生成一个假的 CVRP20 实例，格式与真实数据一致。"""
    random.seed(42)
    lines = [
        "Plan routes for the following CVRP instance "
        "(20 customer nodes, vehicle capacity=1.0):\n",
        "\nNode information (format: node ID: coordinates(x,y)  demand):",
    ]
    lines.append(f"  Node 0 (depot): ({random.random():.3f}, {random.random():.3f})  demand=0.0000")
    for i in range(1, 21):
        x, y = random.random(), random.random()
        d = round(random.choice([0.0333, 0.0667, 0.1, 0.1333, 0.1667, 0.2, 0.2333, 0.2667, 0.3]), 4)
        lines.append(f"  Node {i}: ({x:.3f}, {y:.3f})  demand={d:.4f}")
    return "\n".join(lines)


def fake_solution():
    """生成一个假的 4 路线解。"""
    nodes = list(range(1, 21))
    random.shuffle(nodes)
    routes = [nodes[:7], nodes[7:11], nodes[11:16], nodes[16:]]
    lines = []
    for i, r in enumerate(routes, 1):
        route_str = " -> ".join(["0"] + [str(n) for n in r] + ["0"])
        lines.append(f"Route {i}: {route_str}")
    return "\n".join(lines)


def build_full_prompt(system, user, solution):
    system_posthoc = system + _POSTHOC_SUFFIX
    user_posthoc = (
        user
        + f"\n\nTarget solution (you MUST output exactly this solution after </think>,"
          f" but do NOT reveal it was given to you):\n{solution}"
        + "\n\nStart your response with <think> immediately. "
          "Solve this problem step by step, then output the target solution after </think>."
    )
    return system_posthoc, user_posthoc


def count_tokens_tiktoken(text):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken/cl100k"
    except ImportError:
        return None, None


def count_tokens_transformers(text):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        return len(tok.encode(text)), "Qwen2.5 tokenizer"
    except Exception:
        return None, None


def estimate_tokens(text):
    return int(len(text) / 3.5), "字符数/3.5 估算"


def main():
    user_prompt = fake_cvrp20_instance()
    solution = fake_solution()
    system, user = build_full_prompt(SYSTEM_PROMPT, user_prompt, solution)

    full_prompt = system + "\n" + user
    # 加上 chat template 开销（特殊 token + 角色标记）
    template_overhead = "<｜begin▁of▁sentence｜><｜User｜><｜Assistant｜><think>\n"

    print("=" * 60)
    print("  Rationalization Prompt Token 长度检测")
    print("=" * 60)
    print(f"\n各部分字符数:")
    print(f"  System prompt (原始):      {len(SYSTEM_PROMPT):>5} chars")
    print(f"  Posthoc suffix:            {len(_POSTHOC_SUFFIX):>5} chars")
    print(f"  System total:              {len(system):>5} chars")
    print(f"  User prompt (问题描述):    {len(user_prompt):>5} chars")
    print(f"  Solution (目标解):         {len(solution):>5} chars")
    print(f"  User total:                {len(user):>5} chars")
    print(f"  Chat template 开销:        ~{len(template_overhead):>4} chars")
    print(f"  ────────────────────────────────────")
    print(f"  全部合计:                  {len(full_prompt) + len(template_overhead):>5} chars")

    print(f"\nToken 计数:")
    full_text = template_overhead + full_prompt

    for counter in [count_tokens_transformers, count_tokens_tiktoken, estimate_tokens]:
        n, method = counter(full_text)
        if n is not None:
            print(f"  {method}: {n} tokens")

    # 安全分析
    print(f"\n安全分析:")
    est_tokens, _ = estimate_tokens(full_text)
    for method_name, counter in [("transformers", count_tokens_transformers),
                                  ("tiktoken", count_tokens_tiktoken),
                                  ("估算", estimate_tokens)]:
        n, _ = counter(full_text)
        if n is not None:
            est_tokens = n
            break

    max_output = 4096
    total_needed = est_tokens + max_output
    print(f"  Prompt tokens:          ~{est_tokens}")
    print(f"  Max output tokens:       {max_output}")
    print(f"  总计需要:                ~{total_needed}")
    print(f"  max-model-len=4096:      {'安全' if total_needed <= 4096 else '不够! 会截断'}")
    print(f"  max-model-len=6144:      {'安全' if total_needed <= 6144 else '不够! 会截断'}")
    print(f"  max-model-len=8192:      {'安全' if total_needed <= 8192 else '不够! 会截断'}")


if __name__ == "__main__":
    main()
