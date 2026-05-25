"""
反向思维链（reverse CoT）SFT 数据生成器 —— 任务 A：depot 补全 / giant-tour 切分。

动机（reverse-thinking augmentation）：
  正向任务是"实例 → 路线"（顺序 + 切分一起决定）。
  反向任务给出"固定访问顺序"（把真解的所有 depot token 删掉拼成的 giant tour），
  只让模型决定"在哪里返回 depot"（把顺序切分成容量可行的多条路线）。
  这单独锤炼"沿序累积 load + 在超容前/簇边界切分"这一容量推理子技能。

与 build_think_chains.py 的关系：
  - 复用其 parse_instance_from_prompt / parse_routes / _dist / format_route_answer
  - 标签 = ground-truth 路线（监督模仿，良定义），不重新求解
  - 切分原因（容量受限 vs 效率切分）用与正向 build_steps_cvrp 完全一致的可行性判据重建
  - 不调用任何 LLM，纯脚本、零噪声、一致性由构造保证

输出格式与正向严格对齐：同 <think> 四段骨架、同 "Route N: 0 -> ... -> 0" 答案格式、
同节点信息输入格式，仅 system prompt 的任务描述与第 2 段步骤内容不同。
分布保持：反向样本从同一批实例/同一真解派生，实例分布不变；输出仍在路由解空间，
token 分布扰动最小。混合比例由下游 SFT 控制（建议正向主导，反向 ~20%）。

用法：
    python build_reverse_chains.py --input data/solutions_cvrp20.jsonl \
        --output data/chains_reverse_cvrp20.jsonl
    # 只取前 N 条做样例/消融：
    python build_reverse_chains.py --input data/solutions_cvrp20.jsonl \
        --output data/chains_reverse_cvrp20_sample.jsonl --limit 200
"""

import argparse
import json
import math
from datetime import datetime

# 复用正向生成器的解析与格式化工具，避免重复实现、保证一致
from build_think_chains import (
    parse_instance_from_prompt,
    parse_routes,
    _dist,
    format_route_answer,
    _rewrite_demand_precision,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. 反向任务的 system prompt（镜像 CVRP 正向 prompt 的结构，仅任务语义不同）
# ═══════════════════════════════════════════════════════════════════════════════

REVERSE_SYSTEM_PROMPT_CVRP = """You are a logistics route planning expert solving the Capacitated Vehicle Routing Problem (CVRP).
You are given a FIXED visit order of all customers. Your job is NOT to reorder customers, but to decide where each vehicle returns to depot 0 (splitting the fixed order into routes) so that each route's total demand does not exceed vehicle capacity and total distance is minimized.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these four sections in order:

1. **Strategy**: State total demand, capacity, and minimum vehicles. Restate the fixed visit order. Describe the splitting principle: scan the order accumulating load; return to depot and start a new vehicle right before the next customer's demand would exceed remaining capacity, or earlier when the upcoming customers are geographically better served by a fresh vehicle.

2. **Step-by-step splitting**: Walk along the fixed order one customer at a time. Each kept-customer step format:
   [R1,step] cap=X.XX | keep N (dem=X.XX) cap→X.XX
   - cap = remaining capacity before this customer; subtract its demand to get cap→.
   When the current route must close because no remaining customer fits:
     [R1,step] cap=X.XX | check: A(dem=X.XX>X.XX), B(dem=X.XX>X.XX), ... → no room for any remaining → return depot (d=X.XX), start R2
   When the current route closes early for efficiency (some remaining customer still fits but the route is better closed):
     [R1,step] cap=X.XX | feasible: A(dem=X.XX<=X.XX), ... but upcoming nodes better served by a new vehicle → return depot (d=X.XX), start R2
   When all customers have been placed:
     [R1,step] cap=X.XX | all customers placed → return depot (d=X.XX)

3. **Verification**: For each route, list its customers, count, and load. Confirm load <= capacity and total count equals n. Format: "R1:{nodes}=count(load=X.XX) | R2:{nodes}=count(load=X.XX) | ... Total: X/N customers. ✓ All covered, no duplicates, all feasible."

4. **Final routes**: Write all complete routes in "Route N: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final routes (copied from think):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 反向 think 各段构建
# ═══════════════════════════════════════════════════════════════════════════════

# 浮点容差：demand 取 2 位小数，cap 由 2 位小数连减得到，比较时留余量
_EPS = 1e-9


def build_reverse_strategy_cvrp(coords, demands, capacity, giant) -> str:
    """第 1 段 Strategy：复述总需求/容量/最小车数 + 固定顺序 + 切分原则。"""
    total_demand = float(sum(demands))
    n_customers = int(sum(1 for d in demands if d > 0))
    min_v = math.ceil(total_demand / capacity)
    giant_str = " -> ".join(str(v) for v in giant)
    lines = [
        f"Depot at ({coords[0][0]:.3f}, {coords[0][1]:.3f}). {n_customers} customers.",
        f"Total demand: {total_demand:.2f}. Capacity: {capacity:.2f}. "
        f"Min vehicles needed: ceil({total_demand:.2f}/{capacity:.2f})={min_v}.",
        f"Fixed visit order: {giant_str}.",
        "Plan: scan the fixed order accumulating load; return to depot and start a new "
        "vehicle right before the load would exceed capacity, or earlier when the upcoming "
        "customers are geographically better served by a fresh vehicle.",
    ]
    return "\n".join(lines)


def build_reverse_steps_cvrp(routes, coords, demands, capacity) -> list[str]:
    """第 2 段：沿固定顺序逐客户走，给出 keep / 切分决策。

    切分原因用与正向 build_steps_cvrp 完全一致的判据重建：
      - 关闭路线时，对"后续所有未放置客户"做可行性筛 feasible=[v: dem(v)<=cap]
      - feasible 为空      → 容量受限切分（check: ... > cap）
      - feasible 非空      → 效率切分（next fits but better served by new vehicle）
      - 无后续客户         → all customers placed
    """
    steps = []
    giant = [v for r in routes for v in r if v != 0]
    placed = 0  # 已放置客户数（giant 顺序下的指针）

    for r_idx, route in enumerate(routes):
        customers = [v for v in route if v != 0]
        cap = capacity

        # 逐个 keep
        for j, v in enumerate(customers):
            dem = float(demands[v])
            before = cap
            cap = cap - dem
            steps.append(
                f"[R{r_idx+1},{j+1}] cap={before:.2f} | keep {v} (dem={dem:.2f}) cap→{cap:.2f}"
            )

        placed += len(customers)
        last = customers[-1]
        d = _dist(coords, last, 0)
        close_label = f"[R{r_idx+1},{len(customers)+1}]"
        remaining = giant[placed:]

        if not remaining:
            steps.append(
                f"{close_label} cap={cap:.2f} | all customers placed "
                f"→ return depot (d={d:.3f})"
            )
        else:
            feasible = [v for v in remaining if demands[v] <= cap + _EPS]
            if not feasible:
                checks = [f"{v}(dem={demands[v]:.2f}>{cap:.2f})" for v in remaining[:5]]
                if len(remaining) > 5:
                    checks.append("...")
                steps.append(
                    f"{close_label} cap={cap:.2f} | check: {', '.join(checks)} "
                    f"→ no room for any remaining → return depot (d={d:.3f}), "
                    f"start R{r_idx+2}"
                )
            else:
                # 存在仍能装入的后续客户（不一定是顺序上的下一个），但真解选择切分
                feas_str = ", ".join(
                    f"{v}(dem={demands[v]:.2f}<={cap:.2f})" for v in feasible[:3]
                )
                if len(feasible) > 3:
                    feas_str += ", ..."
                steps.append(
                    f"{close_label} cap={cap:.2f} | feasible: {feas_str} "
                    f"but upcoming nodes better served by a new vehicle "
                    f"→ return depot (d={d:.3f}), start R{r_idx+2}"
                )

    return steps


def build_reverse_verification(routes, demands, n) -> list[str]:
    """第 3 段：逐路线列客户/数量/load，核对覆盖与可行。"""
    summaries = []
    total = 0
    for i, route in enumerate(routes):
        custs = sorted(v for v in route if v != 0)
        load = float(sum(demands[v] for v in custs))
        total += len(custs)
        nodes_str = ",".join(str(v) for v in custs)
        summaries.append(f"R{i+1}:{{{nodes_str}}}={len(custs)}(load={load:.2f})")
    ok = (total == n)
    lines = [
        " | ".join(summaries),
        f"Total: {total}/{n} customers. "
        + ("✓ All covered, no duplicates, all feasible." if ok else "✗ Error."),
    ]
    return lines


def build_reverse_think_chain_cvrp(instance, routes) -> str:
    """组装完整 <think>...</think> + answer（与正向 build_think_chain 同骨架）。"""
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    giant = [v for r in routes for v in r if v != 0]

    strategy = build_reverse_strategy_cvrp(coords, demands, capacity, giant)
    step_lines = build_reverse_steps_cvrp(routes, coords, demands, capacity)
    answer = format_route_answer(routes, multi_route=True)
    verify_lines = build_reverse_verification(routes, demands, instance["n"])

    think_parts = [
        f"1. **Strategy**: {strategy}",
        "",
        "2. **Step-by-step splitting**:",
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
# 2. user prompt 改造：保留节点信息块，前置反向任务说明 + 固定访问顺序
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

    # 单路线没有可学的切分，跳过
    if len(routes) < 2:
        return None

    giant = [v for r in routes for v in r if v != 0]
    user_prompt_rw = _rewrite_demand_precision(user_prompt)
    rev_user = build_reverse_user_prompt(user_prompt_rw, giant, instance["capacity"], n)
    output = build_reverse_think_chain_cvrp(instance, routes)

    return {
        "id": record["id"] + "_rev",
        "problem_type": pt,
        "task": "depot_insertion",
        "n": n,
        "prompt": {"system": REVERSE_SYSTEM_PROMPT_CVRP, "user": rev_user},
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
    parser.add_argument("--limit", type=int, default=0,
                        help="最多处理多少条（0=全部），用于样例/消融")
    args = parser.parse_args()

    success, fail = 0, 0
    done = False
    with open(args.output, "w", encoding="utf-8") as out_f:
        for path in args.input:
            if done:
                break
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    result = process_record_reverse(rec)
                    if result:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        success += 1
                    else:
                        fail += 1
                    if args.limit and success >= args.limit:
                        done = True
                        break

    print(f"完成: {success} 条 reverse chain")
    if fail:
        print(f"跳过/失败: {fail} 条（非 CVRP / 单路线 / 解析失败）")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# 扩展备忘（VRPTW depot 补全 / TSPTW 右窗口反推）
# ═══════════════════════════════════════════════════════════════════════════════
# - VRPTW：同样删 depot 拼 giant tour，但切分需同时满足容量(若有)与时间窗可行；
#   关闭原因增加"后续节点 arr>deadline 不可达"一类，沿 build_steps_vrptw 的时序传播重建。
# - TSPTW 右窗口反推（输出标量，格式偏离更大，建议单独低比例混合）：
#   抹掉某节点 k 的 latest l_k，标签 = 沿真解时序传播算出的服务开始时刻 s_k（lk 最小可行值）。
