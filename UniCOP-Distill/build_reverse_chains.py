"""
反向思维链（reverse CoT）SFT 数据生成器 —— 任务 A：depot 补全 / 路线切分。

动机（reverse-thinking augmentation）：
  正向任务给实例 → 模型同时决定"访问顺序 + 切分"。
  反向任务给"固定访问顺序"（真解删掉所有 depot token 拼成的 giant tour），
  模型只需决定"在哪里返回 depot（切分成容量可行的多条路线）"。

** 格式对齐（关键）**：
  第 2/3/4 段直接复用 build_think_chains 的 build_steps_cvrp / _build_verification /
  format_route_answer，与正向 chains **逐字节同格式**——保留 Unvisited 选择池、
  cap=X-X=X 容量算术、feasible 候选池、→ select M、两种 return depot 原因。
  目的：不引入任何新步骤模式，避免破坏模型已从正向学到的格式。
  反向与正向的差异只在：① system/user prompt（给定固定顺序 + depot 插入任务框定）
  ② 第 1 段 Strategy（复述给定顺序，而非簇分析）。

  标签 = ground-truth 路线；纯脚本、不调 LLM、零噪声、一致性由构造保证。
  分布保持：反向样本从同一批真解派生，实例分布不变；输出仍在路由解空间。
  混合比例由下游 SFT 控制（建议正向主导，反向 ~20%）。

用法：
    python build_reverse_chains.py --input data/solutions_cvrp20.jsonl \
        --output data/chains_reverse_cvrp20.jsonl
    python build_reverse_chains.py --input data/solutions_cvrp20.jsonl \
        --output data/chains_reverse_cvrp20_sample.jsonl --limit 200
"""

import argparse
import json
import math
import random
from datetime import datetime

# 复用正向生成器：解析、逐步构建、验证、格式化 —— 保证格式逐字节一致
from build_think_chains import (
    parse_instance_from_prompt,
    parse_routes,
    format_route_answer,
    _rewrite_demand_precision,
    build_steps_cvrp,
    _build_verification,
)
from problems_prompt import get_system_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# 0. 反向任务 system prompt = 任务框定 preamble + 正向 CVRP prompt（原样）
#    第 2~4 段格式指令与正向逐字节相同，仅 preamble 框定任务 + 改写第 1 段要求。
# ═══════════════════════════════════════════════════════════════════════════════

_REVERSE_PREAMBLE_CVRP = (
    "TASK VARIANT — DEPOT INSERTION: You are given a FIXED visit order of all customers "
    "in the user message (the line 'Fixed visit order: ...'). Do NOT reorder customers. "
    "Follow that exact order; your only decisions are where each vehicle returns to depot 0 "
    "(i.e., where to split the order into routes) so that no route's total demand exceeds "
    "capacity and total distance is minimized.\n"
    "For section 1 (Strategy), state total demand, capacity and minimum vehicles, restate "
    "the given order, and describe the splitting principle (follow the order accumulating "
    "load; return to depot before the next customer's demand exceeds remaining capacity, or "
    "earlier when remaining nodes are better served by a new vehicle). Sections 2-4 follow "
    "the standard format below UNCHANGED; at each construction step the selected node is "
    "dictated by the given fixed order.\n\n"
)


def get_reverse_system_prompt_cvrp() -> str:
    return _REVERSE_PREAMBLE_CVRP + get_system_prompt("cvrp")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 第 1 段 Strategy：复述固定顺序 + 切分原则（其余段沿用正向生成器）
# ═══════════════════════════════════════════════════════════════════════════════

def build_reverse_strategy_cvrp(coords, demands, capacity, giant, n_routes) -> str:
    total_demand = float(sum(demands))
    n_customers = int(sum(1 for d in demands if d > 0))
    min_v = math.ceil(total_demand / capacity)
    giant_str = " -> ".join(str(v) for v in giant)
    lines = [
        f"Depot at ({coords[0][0]:.3f}, {coords[0][1]:.3f}). {n_customers} customers.",
        f"Total demand: {total_demand:.2f}. Capacity: {capacity:.2f}. "
        f"Min vehicles needed: ceil({total_demand:.2f}/{capacity:.2f})={min_v}. "
        f"Using {n_routes} routes.",
        f"Given visit order (fixed, do not reorder): {giant_str}.",
        "Plan: follow the fixed order accumulating load; return to depot and start a new "
        "vehicle before the next customer's demand would exceed remaining capacity, or "
        "earlier when the remaining nodes are better served by a new vehicle.",
    ]
    return "\n".join(lines)


def build_reverse_think_chain_cvrp(instance, routes) -> str:
    """组装 <think>...</think> + answer。第 2/3/4 段复用正向生成器，逐字节同格式。"""
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    giant = [v for r in routes for v in r if v != 0]

    strategy = build_reverse_strategy_cvrp(coords, demands, capacity, giant, len(routes))
    step_lines = build_steps_cvrp(routes, coords, demands, capacity, stride=1)  # 正向同款
    answer = format_route_answer(routes, multi_route=True)
    verify_lines = _build_verification(routes, instance["n"], multi_route=True)  # 正向同款

    think_parts = [
        f"1. **Strategy**: {strategy}",
        "",
        "2. **Step-by-step construction**:",
        *step_lines,
        "",
        "3. **Verification**:",
        *verify_lines,
        "",
        "4. **Final routes**:",
        answer,
    ]
    think_content = "\n".join(think_parts)
    return f"<think>\n{think_content}\n</think>\n{answer}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. user prompt 改造：保留节点信息块，前置任务说明 + 固定访问顺序
# ═══════════════════════════════════════════════════════════════════════════════

def build_reverse_user_prompt(original_user, giant, capacity, n) -> str:
    idx = original_user.find("Node information")
    node_block = original_user[idx:] if idx != -1 else original_user
    giant_str = " -> ".join(str(v) for v in giant)
    header = (
        f"Insert depot returns for the following CVRP instance "
        f"({n} customer nodes, vehicle capacity={capacity:.1f}). "
        f"The customer visit order is FIXED as given below; decide only where each vehicle "
        f"returns to depot 0 (splitting the order into routes), so that no route's total "
        f"demand exceeds capacity and total distance is minimized.\n\n"
        f"Fixed visit order: {giant_str}\n\n"
    )
    return header + node_block


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 单条记录处理
# ═══════════════════════════════════════════════════════════════════════════════

def process_record_reverse(record: dict) -> dict | None:
    """把一条 solutions 记录转成 depot 补全反向样本。仅处理 CVRP。"""
    pt = record["problem_type"]
    if pt != "cvrp":
        return None  # 本脚本聚焦 CVRP；VRPTW 同理可扩展（见文末注释）

    solution = record["solution"]
    user_prompt = record["prompt"]["user"]

    instance = parse_instance_from_prompt(user_prompt, pt)
    if not instance:
        return None

    routes = parse_routes(solution, multi_route=True)
    if not routes:
        return None

    n = instance["n"]
    all_customers = sorted(v for r in routes for v in r if v != 0)
    if all_customers != list(range(1, n + 1)):
        return None

    if len(routes) < 2:
        return None  # 单路线无可学切分

    giant = [v for r in routes for v in r if v != 0]
    user_prompt_rw = _rewrite_demand_precision(user_prompt)
    rev_user = build_reverse_user_prompt(user_prompt_rw, giant, instance["capacity"], n)
    output = build_reverse_think_chain_cvrp(instance, routes)

    return {
        "id": record["id"] + "_rev",
        "problem_type": pt,
        "task": "depot_insertion",
        "n": n,
        "prompt": {"system": get_reverse_system_prompt_cvrp(), "user": rev_user},
        "output": output,
        "solver_distance": record.get("solver_distance"),
        "method": "reverse_template",
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="生成 depot 补全反向思维链 SFT 数据（CVRP）")
    parser.add_argument("--input", nargs="+", required=True,
                        help="输入 solutions JSONL（与正向同源）")
    parser.add_argument("--output", required=True,
                        help="输出 reverse chains JSONL")
    parser.add_argument("--sample", type=int, default=0,
                        help="随机抽取多少条作为反向数据（0=不随机抽样），需配合 --seed")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机抽样种子（保证可复现）")
    parser.add_argument("--limit", type=int, default=0,
                        help="顺序取前 N 条（0=全部）；--sample>0 时忽略本项")
    args = parser.parse_args()

    # 读取全部输入记录
    records = []
    for path in args.input:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)
    print(f"读取 {len(records)} 条 solutions")

    # 随机抽样：按种子打乱顺序后，依次处理直到凑够 sample 条成功输出（跳过的不计入）
    if args.sample > 0:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        target = args.sample
        print(f"随机抽样模式: 目标 {target} 条 (seed={args.seed})")
    else:
        target = args.limit if args.limit > 0 else None

    success, fail = 0, 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for line in records:
            rec = json.loads(line)
            result = process_record_reverse(rec)
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                success += 1
            else:
                fail += 1
            if target is not None and success >= target:
                break

    print(f"完成: {success} 条 reverse chain")
    if fail:
        print(f"跳过/失败: {fail} 条（非 CVRP / 单路线 / 解析失败）")
    if args.sample > 0 and success < args.sample:
        print(f"⚠️ 有效记录不足，仅得 {success}/{args.sample} 条")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# 扩展备忘（VRPTW depot 补全 / TSPTW 右窗口反推）
# ═══════════════════════════════════════════════════════════════════════════════
# - VRPTW：同样删 depot 拼 giant tour，第 2 段复用 build_steps_vrptw（时序+可行性同款），
#   第 1 段复述顺序，system prompt = 反向 preamble + get_system_prompt("vrptw")。
# - TSPTW 右窗口反推（输出标量，格式偏离更大，建议单独低比例混合）：
#   抹掉某节点 k 的 latest l_k，标签 = 沿真解时序传播算出的服务开始时刻 s_k。
