"""检测不同问题规模下 rationalization prompt 的 token 长度。

用法：python check_prompt_length.py
"""

import math
import random

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


def fake_cvrp_instance(n):
    random.seed(42)
    lines = [
        f"Plan routes for the following CVRP instance "
        f"({n} customer nodes, vehicle capacity=1.0):\n",
        "\nNode information (format: node ID: coordinates(x,y)  demand):",
    ]
    lines.append(f"  Node 0 (depot): ({random.random():.3f}, {random.random():.3f})  demand=0.0000")
    for i in range(1, n + 1):
        x, y = random.random(), random.random()
        d = round(random.choice([0.0333, 0.0667, 0.1, 0.1333, 0.1667, 0.2, 0.2333, 0.2667, 0.3]), 4)
        lines.append(f"  Node {i}: ({x:.3f}, {y:.3f})  demand={d:.4f}")
    return "\n".join(lines)


def fake_solution(n):
    random.seed(42)
    nodes = list(range(1, n + 1))
    random.shuffle(nodes)
    num_routes = max(2, n // 5)
    chunk = math.ceil(n / num_routes)
    routes = [nodes[i:i+chunk] for i in range(0, n, chunk)]
    lines = []
    for i, r in enumerate(routes, 1):
        route_str = " -> ".join(["0"] + [str(nd) for nd in r] + ["0"])
        lines.append(f"Route {i}: {route_str}")
    return "\n".join(lines)


def build_full_text(n):
    user = fake_cvrp_instance(n)
    solution = fake_solution(n)
    system = SYSTEM_PROMPT + _POSTHOC_SUFFIX
    user_full = (
        user
        + f"\n\nTarget solution (you MUST output exactly this solution after </think>,"
          f" but do NOT reveal it was given to you):\n{solution}"
        + "\n\nStart your response with <think> immediately. "
          "Solve this problem step by step, then output the target solution after </think>."
    )
    template = "<|begin_of_sentence|><|User|><|Assistant|><think>\n"
    return template + system + "\n" + user_full


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        return tok, "Qwen2.5"
    except Exception:
        pass
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return enc, "tiktoken"
    except ImportError:
        return None, None


def count(tokenizer, text):
    if tokenizer is None:
        return int(len(text) / 3.5)
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))
    return int(len(text) / 3.5)


def main():
    tokenizer, tok_name = get_tokenizer()
    if tokenizer:
        print(f"Tokenizer: {tok_name}\n")
    else:
        print("Tokenizer: 字符数/3.5 估算 (未安装 transformers/tiktoken)\n")

    sizes = [20, 50, 100]
    max_output = 4096

    print(f"{'Size':>5} | {'Prompt':>7} | {'+ Output':>8} | {'= Total':>7} | 建议 max-model-len")
    print("-" * 65)

    for n in sizes:
        text = build_full_text(n)
        prompt_tokens = count(tokenizer, text)
        total = prompt_tokens + max_output
        if total <= 4096:
            suggestion = "4096"
        elif total <= 6144:
            suggestion = "6144"
        elif total <= 8192:
            suggestion = "8192"
        elif total <= 12288:
            suggestion = "12288"
        elif total <= 16384:
            suggestion = "16384"
        else:
            suggestion = f"{((total // 1024) + 1) * 1024}"
        print(f"  n={n:<3} | {prompt_tokens:>6} | {'+' + str(max_output):>8} | {total:>6} | {suggestion}")

    print(f"\n结论: max-model-len 需要根据问题规模动态设置。")
    print(f"      当前 bash 脚本中 VLLM_MAX_MODEL_LEN 应按最大规模配置。")


if __name__ == "__main__":
    main()
