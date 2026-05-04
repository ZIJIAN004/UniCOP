"""
从 LKH/HGS 求解器的解 programmatically 构造思维链 SFT 数据。

与旧的 rationalize_solutions.py（后验推理，让 LLM 倒推理由）不同：
  - 不调用任何 LLM
  - 直接从解的访问顺序 + 实例数据计算每一步的决策信息
  - 产出因果一致的 think chain（推理步骤和最终路线完全对应）

输入：solutions_*.jsonl（由 stage1_solution/generate_solutions.py 生成）
输出：chains_template_*.jsonl（与 train_sft_stage2.py 兼容）

用法：
    python build_think_chains.py --input data/solutions_cvrp20.jsonl --output data/chains_template_cvrp20.jsonl
    python build_think_chains.py --input data/solutions_tsp50.jsonl data/solutions_cvrp50.jsonl --output data/chains_template_50.jsonl
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 实例解析：从 user prompt 文本中提取坐标、需求、容量、时间窗等
# ═══════════════════════════════════════════════════════════════════════════════

def parse_instance_from_prompt(user_prompt: str, problem_type: str) -> dict:
    """从 user prompt 文本中解析实例数据。"""
    lines = user_prompt.strip().split("\n")
    coords, demands, time_windows, draft_limits = [], [], [], []
    capacity = None
    n = 0

    cap_match = re.search(r"capacity[=:]?\s*([\d.]+)", user_prompt, re.IGNORECASE)
    if cap_match:
        capacity = float(cap_match.group(1))

    load_match = re.search(r"initial load[=:]?\s*([\d.]+)", user_prompt, re.IGNORECASE)
    initial_load = float(load_match.group(1)) if load_match else capacity

    node_pattern = re.compile(
        r"Node\s+(\d+).*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)"
    )
    demand_pattern = re.compile(r"demand[=:]\s*([\d.]+)")
    tw_pattern = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.inf]+)\s*\]")
    dl_pattern = re.compile(r"draft_limit[=:]\s*([\d.inf]+)")
    unload_pattern = re.compile(r"unload[=:]\s*(\d+)")

    node_data = {}
    for line in lines:
        nm = node_pattern.search(line)
        if not nm:
            continue
        nid = int(nm.group(1))
        x, y = float(nm.group(2)), float(nm.group(3))
        node_data[nid] = {"coord": (x, y)}

        dm = demand_pattern.search(line)
        if dm:
            node_data[nid]["demand"] = float(dm.group(1))

        tm = tw_pattern.search(line)
        if tm:
            earliest = float(tm.group(1))
            latest_s = tm.group(2)
            latest = 1e9 if "inf" in latest_s else float(latest_s)
            node_data[nid]["tw"] = (earliest, latest)

        dlm = dl_pattern.search(line)
        if dlm:
            dl_s = dlm.group(1)
            node_data[nid]["dl"] = 1e9 if "inf" in dl_s else float(dl_s)

        um = unload_pattern.search(line)
        if um:
            node_data[nid]["unload"] = float(um.group(1))

    if not node_data:
        return {}

    n = max(node_data.keys())
    coords = [(0.0, 0.0)] * (n + 1)
    for nid, nd in node_data.items():
        coords[nid] = nd["coord"]

    instance = {"n": n, "coords": np.array(coords)}

    if problem_type in ("cvrp",):
        dems = np.zeros(n + 1)
        for nid, nd in node_data.items():
            dems[nid] = nd.get("demand", 0.0)
        instance["demands"] = dems
        instance["capacity"] = capacity if capacity else 1.0

    if problem_type in ("tsptw", "vrptw"):
        tw = np.zeros((n + 1, 2))
        tw[0] = [0.0, 1e9]
        for nid, nd in node_data.items():
            if "tw" in nd:
                tw[nid] = nd["tw"]
        instance["time_windows"] = tw

    if problem_type == "tspdl":
        dl = np.full(n + 1, 1e9)
        dems = np.zeros(n + 1)
        for nid, nd in node_data.items():
            if "dl" in nd:
                dl[nid] = nd["dl"]
            if "unload" in nd:
                dems[nid] = nd["unload"]
            elif "demand" in nd:
                dems[nid] = nd["demand"]
        instance["draft_limits"] = dl
        instance["demands"] = dems
        instance["capacity"] = initial_load if initial_load else float(n)

    return instance


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 路线解析：从 solution 字符串中提取路线列表
# ═══════════════════════════════════════════════════════════════════════════════

def parse_routes(solution: str, multi_route: bool) -> list[list[int]]:
    """解析求解器输出的路线字符串为节点列表。"""
    routes = []
    for line in solution.strip().split("\n"):
        line = line.strip()
        m = re.match(r"Route\s*\d*\s*:\s*(.+)", line, re.IGNORECASE)
        if not m:
            continue
        nodes_str = m.group(1)
        nodes = [int(x.strip()) for x in re.split(r"\s*->\s*", nodes_str)]
        if nodes and nodes[0] == 0 and nodes[-1] == 0:
            routes.append(nodes)
    if not routes and not multi_route:
        m = re.match(r"Route\s*:\s*(.+)", solution.strip(), re.IGNORECASE)
        if m:
            nodes = [int(x.strip()) for x in re.split(r"\s*->\s*", m.group(1))]
            if nodes and nodes[0] == 0 and nodes[-1] == 0:
                routes.append(nodes)
    return routes


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 策略检测：分析路线，反推匹配度最高的启发式策略
# ═══════════════════════════════════════════════════════════════════════════════

def _angle_from_depot(coords, depot_idx, node_idx):
    dx = coords[node_idx][0] - coords[depot_idx][0]
    dy = coords[node_idx][1] - coords[depot_idx][1]
    return math.atan2(dy, dx)


def _angle_monotonicity(angles: list[float]) -> tuple[float, str]:
    """计算角度序列的单调性得分。返回 (score, direction)。"""
    if len(angles) < 2:
        return 0.0, "cw"
    n = len(angles) - 1
    cw_count = sum(1 for i in range(n)
                   if (angles[i + 1] - angles[i]) % (2 * math.pi) < math.pi)
    ccw_count = n - cw_count
    if cw_count >= ccw_count:
        return cw_count / n, "counterclockwise"
    return ccw_count / n, "clockwise"


def detect_strategy_tsp(route: list[int], coords: np.ndarray) -> str:
    """TSP 策略检测：显式展示特征计算过程。"""
    customers = route[1:-1]
    n = len(customers)
    angles = [_angle_from_depot(coords, 0, c) for c in customers]
    mono_score, direction = _angle_monotonicity(angles)

    depot = coords[0]
    quadrants = {"NE": [], "NW": [], "SE": [], "SW": []}
    for c in customers:
        dx, dy = coords[c][0] - depot[0], coords[c][1] - depot[1]
        key = ("N" if dy >= 0 else "S") + ("E" if dx >= 0 else "W")
        quadrants[key].append(c)
    non_empty = {k: v for k, v in quadrants.items() if v}

    nn_count = 0
    for i, node in enumerate(customers[:-1]):
        remaining = set(customers[i + 1:])
        if not remaining:
            break
        dists = {r: float(np.linalg.norm(coords[node] - coords[r])) for r in remaining}
        top3 = sorted(dists, key=dists.get)[:3]
        if customers[i + 1] in top3:
            nn_count += 1
    nn_score = nn_count / max(1, len(customers) - 1)

    # 构建特征分析段
    quad_str = ", ".join(f"{k}:{len(v)}" for k, v in sorted(non_empty.items()))
    feature_lines = [
        f"Depot at ({depot[0]:.3f}, {depot[1]:.3f}). {n} customer nodes.",
        f"Quadrant distribution: {quad_str}.",
        f"Angular monotonicity from depot: {mono_score:.2f} ({direction}).",
        f"Nearest-neighbor match rate: {nn_score:.2f} "
        f"({nn_count}/{max(1,n-1)} steps chose a top-3 nearest node).",
    ]

    if mono_score > 0.7:
        feature_lines.append(
            f"High angular monotonicity → angular sweep is effective."
        )
        feature_lines.append(
            f"Plan: sweep {direction} starting from nodes {customers[:3]}, "
            f"ending with {customers[-3:]}, then return to depot."
        )
    elif nn_score > 0.6:
        position = "centrally" if 0.3 < depot[0] < 0.7 and 0.3 < depot[1] < 0.7 else "at the edge"
        feature_lines.append(
            f"High NN match rate → nearest-neighbor heuristic is effective. "
            f"Depot positioned {position}."
        )
        feature_lines.append(
            f"Plan: greedily visit closest unvisited node at each step."
        )
    else:
        segments = sorted(non_empty.keys())
        feature_lines.append(
            f"No dominant single pattern. Nodes cluster by quadrant."
        )
        feature_lines.append(
            f"Plan: visit each geographic segment ({', '.join(segments)}) sequentially, "
            f"nearest-neighbor within segments, connect at boundary nodes."
        )

    return "\n".join(feature_lines)


def detect_strategy_cvrp(routes: list[list[int]], coords: np.ndarray,
                         demands: np.ndarray, capacity: float) -> str:
    """CVRP 策略检测：显式展示特征计算过程。"""
    n_routes = len(routes)
    depot = coords[0]
    n_customers = int(sum(1 for d in demands if d > 0))
    total_demand = float(sum(demands))
    min_vehicles = math.ceil(total_demand / capacity)

    # 分析各路线
    cluster_info = []
    for i, route in enumerate(routes):
        customers = [v for v in route if v != 0]
        total_dem = sum(demands[c] for c in customers)
        angles = [_angle_from_depot(coords, 0, c) for c in customers]
        mean_angle = math.degrees(sum(angles) / len(angles)) % 360 if angles else 0
        angle_span = (max(angles) - min(angles)) if len(angles) > 1 else 0
        avg_dist = float(np.mean([_dist(coords, 0, c) for c in customers])) if customers else 0
        cluster_info.append({
            "idx": i + 1, "nodes": customers,
            "demand": total_dem, "angle": mean_angle,
            "angle_span": math.degrees(angle_span),
            "avg_dist": avg_dist,
        })

    cluster_info.sort(key=lambda x: x["angle"])
    avg_util = total_demand / (n_routes * capacity) * 100

    # 构建特征分析段
    lines = [
        f"Depot at ({depot[0]:.3f}, {depot[1]:.3f}). {n_customers} customers.",
        f"Total demand: {total_demand:.2f}. Capacity: {capacity:.2f}.",
        f"Min vehicles needed: ceil({total_demand:.2f}/{capacity:.2f})={min_vehicles}. "
        f"Using {n_routes} routes (avg utilization={avg_util:.0f}%).",
        f"Cluster analysis:",
    ]
    for ci in cluster_info:
        nodes_str = ", ".join(str(n) for n in ci["nodes"])
        util = ci["demand"] / capacity * 100
        lines.append(
            f"  Route {ci['idx']}: [{nodes_str}] "
            f"sum_demand={ci['demand']:.2f} ({util:.0f}%), "
            f"mean_angle={ci['angle']:.0f}°, span={ci['angle_span']:.0f}°, "
            f"avg_d0={ci['avg_dist']:.3f}"
        )
    lines.append(
        f"Plan: serve each cluster as one route, visit by nearest-neighbor within cluster."
    )

    return "\n".join(lines)


def detect_strategy_tsptw(route: list[int], coords: np.ndarray,
                          tw: np.ndarray) -> str:
    """TSPTW 策略检测：显式展示特征计算过程。"""
    customers = route[1:-1]
    deadlines = [tw[c][1] for c in customers]
    earliests = [tw[c][0] for c in customers]
    n = len(customers)

    order_corr = float(np.corrcoef(range(n), deadlines)[0, 1]) if n > 2 else 0

    urgent_nodes = sorted(range(1, len(tw)), key=lambda i: tw[i][1])[:5]
    flexible_nodes = sorted(range(1, len(tw)), key=lambda i: -tw[i][1])[:5]

    depot = coords[0]
    avg_window_width = float(np.mean([tw[c][1] - tw[c][0] for c in customers]))
    min_deadline = min(deadlines)
    max_deadline = max(deadlines)

    angles = [_angle_from_depot(coords, 0, c) for c in customers]
    mono_score, direction = _angle_monotonicity(angles)

    # 构建特征分析段
    urgent_str = ", ".join(f"{u}(tw=[{tw[u][0]:.2f},{tw[u][1]:.2f}])" for u in urgent_nodes[:4])
    flex_str = ", ".join(f"{f}(tw=[{tw[f][0]:.2f},{tw[f][1]:.2f}])" for f in flexible_nodes[:3])

    lines = [
        f"Depot at ({depot[0]:.3f}, {depot[1]:.3f}). {n} customers.",
        f"Time window stats: avg width={avg_window_width:.2f}, "
        f"deadline range=[{min_deadline:.2f}, {max_deadline:.2f}].",
        f"Deadline-visit-order correlation: r={order_corr:.2f}.",
        f"Angular monotonicity from depot: {mono_score:.2f} ({direction}).",
        f"Urgent nodes (earliest deadlines): {urgent_str}.",
        f"Flexible nodes (latest deadlines): {flex_str}.",
    ]

    if order_corr > 0.5:
        lines.append(
            f"High deadline correlation (r={order_corr:.2f}) → deadline-driven ordering. "
            f"Visit urgent nodes first, progress to later deadlines. "
            f"Break ties by geographic proximity."
        )
    elif mono_score > 0.6:
        lines.append(
            f"Angular sweep viable (mono={mono_score:.2f}). "
            f"Use {direction} sweep as base, re-order locally for tight deadlines."
        )
    else:
        lines.append(
            f"No single principle dominates. Balance: visit urgent nodes early, "
            f"then nearest-feasible for remaining, checking deadlines at each step."
        )

    return "\n".join(lines)


def detect_strategy_vrptw(routes: list[list[int]], coords: np.ndarray,
                          tw: np.ndarray) -> str:
    """VRPTW 策略检测：显式展示特征计算过程。"""
    n_routes = len(routes)
    depot = coords[0]
    all_customers = [v for r in routes for v in r if v != 0]
    n_customers = len(all_customers)
    avg_window = float(np.mean([tw[c][1] - tw[c][0] for c in all_customers]))

    # 分析各路线
    route_info = []
    for i, route in enumerate(routes):
        customers = [v for v in route if v != 0]
        if not customers:
            continue
        tw_earliest = min(tw[c][0] for c in customers)
        tw_latest = max(tw[c][1] for c in customers)
        avg_dist = float(np.mean([_dist(coords, 0, c) for c in customers]))
        nodes_str = ", ".join(str(c) for c in customers)
        route_info.append(
            f"  Route {i+1}: [{nodes_str}] "
            f"tw=[{tw_earliest:.2f},{tw_latest:.2f}], "
            f"nodes={len(customers)}, avg_d0={avg_dist:.3f}"
        )

    # 检查路线间时间窗是否有明显分离
    lines = [
        f"Depot at ({depot[0]:.3f}, {depot[1]:.3f}). {n_customers} customers.",
        f"Average time-window width: {avg_window:.2f}.",
        f"Routes needed: {n_routes} (each serves a time-compatible cluster).",
        f"Route analysis:",
    ]
    lines.extend(route_info)
    lines.append(
        f"Plan: within each route, visit by earliest feasible arrival to minimize waiting."
    )

    return "\n".join(lines)


def _spearman_corr(x, y):
    """简单 Spearman 相关系数。"""
    n = len(x)
    if n < 3:
        return 0
    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def _rank(arr):
    order = sorted(range(len(arr)), key=lambda i: arr[i])
    ranks = [0] * len(arr)
    for i, idx in enumerate(order):
        ranks[idx] = i
    return ranks


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 逐步构建：按问题类型计算每一步的决策指标
# ═══════════════════════════════════════════════════════════════════════════════

def _dist(coords, a, b):
    return float(np.linalg.norm(coords[a] - coords[b]))


def _nearest_unvisited(coords, current, unvisited, k=3):
    """返回最近的 k 个未访问节点 [(node, dist), ...]。"""
    dists = [(n, _dist(coords, current, n)) for n in unvisited]
    dists.sort(key=lambda x: x[1])
    return dists[:k]


def build_steps_tsp(route: list[int], coords: np.ndarray) -> list[str]:
    """TSP: 显式计算 total=prev+d"""
    steps = []
    total_dist = 0.0
    unvisited = set(route[1:-1])

    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        d = _dist(coords, curr, nxt)
        prev_total = total_dist
        total_dist += d

        if nxt == 0 and i == len(route) - 2:
            steps.append(
                f"[{i}] at {curr} → 0 (d={d:.3f}, "
                f"total={prev_total:.2f}+{d:.3f}={total_dist:.2f}) "
                f"| return to depot"
            )
            break

        unvisited.discard(nxt)

        alts = _nearest_unvisited(coords, curr, unvisited, k=3)
        alt_str = ", ".join(f"{a}({ad:.3f})" for a, ad in alts) if alts else "none"

        steps.append(
            f"[{i}] at {curr} → {nxt} (d={d:.3f}, "
            f"total={prev_total:.2f}+{d:.3f}={total_dist:.2f}) "
            f"| alt: {alt_str}"
        )

        if i > 0 and i % 10 == 0 and unvisited:
            nodes_str = ", ".join(str(v) for v in sorted(unvisited))
            steps.append(f"Unvisited: {{{nodes_str}}}")

    return steps


def build_steps_cvrp(routes: list[list[int]], coords: np.ndarray,
                     demands: np.ndarray, capacity: float) -> list[str]:
    """CVRP: 显式计算 cap=X-Y=Z"""
    steps = []
    all_unvisited = set()
    for route in routes:
        all_unvisited.update(v for v in route if v != 0)

    for r_idx, route in enumerate(routes):
        nodes_str = ", ".join(str(v) for v in sorted(all_unvisited))
        steps.append(f"Unvisited: {{{nodes_str}}}")

        cap_remaining = capacity
        step_in_route = 0
        for i in range(len(route) - 1):
            curr, nxt = route[i], route[i + 1]
            if nxt == 0:
                d = _dist(coords, curr, 0)
                steps.append(
                    f"[R{r_idx+1},{step_in_route+1}] at {curr} → 0 "
                    f"(d={d:.3f}) return to depot"
                )
                break

            d = _dist(coords, curr, nxt)
            dem = demands[nxt]
            cap_before = cap_remaining
            cap_remaining -= dem
            d0 = _dist(coords, nxt, 0)
            all_unvisited.discard(nxt)
            step_in_route += 1

            feasible_unvisited = [
                v for v in all_unvisited if demands[v] <= cap_before
            ]
            alts = _nearest_unvisited(coords, curr, feasible_unvisited, k=2)
            alt_parts = []
            for a, ad in alts:
                a_cap = cap_before - demands[a]
                alt_parts.append(f"{a}({ad:.3f},cap={cap_before:.2f}-{demands[a]:.4f}={a_cap:.2f})")
            alt_str = ", ".join(alt_parts) if alt_parts else "none"

            steps.append(
                f"[R{r_idx+1},{step_in_route}] at {curr} → {nxt} "
                f"(d={d:.3f}, dem={dem:.4f}) "
                f"cap={cap_before:.2f}-{dem:.4f}={cap_remaining:.2f}, d0={d0:.2f} "
                f"| alt: {alt_str}"
            )

    return steps


def build_steps_tsptw(route: list[int], coords: np.ndarray,
                      tw: np.ndarray) -> list[str]:
    """TSPTW: 显式计算 arr=t+d, slack=dl-arr"""
    steps = []
    current_time = 0.0
    unvisited = set(route[1:-1])

    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        t_depart = current_time
        d = _dist(coords, curr, nxt)
        arr = t_depart + d

        if nxt == 0:
            steps.append(
                f"[{i}] at {curr} → 0 (d={d:.3f}, "
                f"arr={t_depart:.2f}+{d:.3f}={arr:.2f}) "
                f"| return to depot"
            )
            break

        unvisited.discard(nxt)
        deadline = tw[nxt][1]
        slack = deadline - arr
        wait_str = ""
        if arr < tw[nxt][0]:
            wait_str = f" wait until {tw[nxt][0]:.2f}"
            current_time = tw[nxt][0]
        else:
            current_time = arr

        alts = []
        for v in sorted(unvisited, key=lambda x: _dist(coords, curr, x))[:3]:
            ad = _dist(coords, curr, v)
            a_arr = t_depart + ad
            if a_arr <= tw[v][1]:
                a_slack = tw[v][1] - a_arr
                alts.append(f"{v}(d={ad:.3f},slack={tw[v][1]:.2f}-{a_arr:.2f}={a_slack:.2f})")
        alt_str = ", ".join(alts) if alts else "none"

        steps.append(
            f"[{i}] at {curr} → {nxt} (d={d:.3f}, "
            f"arr={t_depart:.2f}+{d:.3f}={arr:.2f}, "
            f"slack={deadline:.2f}-{arr:.2f}={slack:.2f}){wait_str} "
            f"| alt: {alt_str}"
        )

        if i > 0 and i % 10 == 0 and unvisited:
            nodes_str = ", ".join(str(v) for v in sorted(unvisited))
            steps.append(f"Unvisited: {{{nodes_str}}}")

    return steps


def build_steps_vrptw(routes: list[list[int]], coords: np.ndarray,
                      tw: np.ndarray) -> list[str]:
    """VRPTW: 显式计算 arr=t+d, slack=dl-arr"""
    steps = []
    all_unvisited = set()
    for route in routes:
        all_unvisited.update(v for v in route if v != 0)

    for r_idx, route in enumerate(routes):
        nodes_str = ", ".join(str(v) for v in sorted(all_unvisited))
        steps.append(f"Unvisited: {{{nodes_str}}}")

        current_time = 0.0
        step_in_route = 0
        for i in range(len(route) - 1):
            curr, nxt = route[i], route[i + 1]
            t_depart = current_time
            d = _dist(coords, curr, nxt)
            arr = t_depart + d

            if nxt == 0:
                steps.append(
                    f"[R{r_idx+1},{step_in_route+1}] at {curr} → 0 "
                    f"(d={d:.3f}, arr={t_depart:.2f}+{d:.3f}={arr:.2f}) "
                    f"return to depot"
                )
                break

            all_unvisited.discard(nxt)
            step_in_route += 1
            deadline = tw[nxt][1]
            slack = deadline - arr
            wait_str = ""
            if arr < tw[nxt][0]:
                wait_str = f" wait until {tw[nxt][0]:.2f}"
                current_time = tw[nxt][0]
            else:
                current_time = arr

            alts = []
            for v in sorted(all_unvisited, key=lambda x: _dist(coords, curr, x))[:2]:
                ad = _dist(coords, curr, v)
                a_arr = t_depart + ad
                if a_arr <= tw[v][1]:
                    a_slack = tw[v][1] - a_arr
                    alts.append(f"{v}(d={ad:.3f},slack={tw[v][1]:.2f}-{a_arr:.2f}={a_slack:.2f})")
            alt_str = ", ".join(alts) if alts else "none"

            steps.append(
                f"[R{r_idx+1},{step_in_route}] at {curr} → {nxt} "
                f"(d={d:.3f}, arr={t_depart:.2f}+{d:.3f}={arr:.2f}, "
                f"slack={deadline:.2f}-{arr:.2f}={slack:.2f}){wait_str} "
                f"| alt: {alt_str}"
            )

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 思维链组装：Strategy + Steps + Final route → <think>...</think> + answer
# ═══════════════════════════════════════════════════════════════════════════════

def format_route_answer(routes: list[list[int]], multi_route: bool) -> str:
    """格式化最终答案（</think> 之后的输出）。"""
    lines = []
    for i, route in enumerate(routes):
        route_str = " -> ".join(str(v) for v in route)
        if multi_route:
            lines.append(f"Route {i+1}: {route_str}")
        else:
            lines.append(f"Route: {route_str}")
    return "\n".join(lines)


def build_think_chain(problem_type: str, instance: dict,
                      routes: list[list[int]]) -> str:
    """组装完整的 <think>...</think> + answer。"""
    coords = instance["coords"]
    multi_route = problem_type in ("cvrp", "vrptw")

    # Strategy
    if problem_type == "tsp":
        strategy = detect_strategy_tsp(routes[0], coords)
    elif problem_type == "cvrp":
        strategy = detect_strategy_cvrp(
            routes, coords, instance["demands"], instance["capacity"])
    elif problem_type == "tsptw":
        strategy = detect_strategy_tsptw(routes[0], coords, instance["time_windows"])
    elif problem_type == "vrptw":
        strategy = detect_strategy_vrptw(routes, coords, instance["time_windows"])
    else:
        strategy = "Solve using a greedy heuristic approach."

    # Steps
    if problem_type == "tsp":
        step_lines = build_steps_tsp(routes[0], coords)
    elif problem_type == "cvrp":
        step_lines = build_steps_cvrp(
            routes, coords, instance["demands"], instance["capacity"])
    elif problem_type == "tsptw":
        step_lines = build_steps_tsptw(routes[0], coords, instance["time_windows"])
    elif problem_type == "vrptw":
        step_lines = build_steps_vrptw(routes, coords, instance["time_windows"])
    else:
        step_lines = []

    # Final route (inside think)
    answer = format_route_answer(routes, multi_route)

    # Assemble
    think_parts = [
        f"1. **Strategy**: {strategy}",
        "",
        "2. **Step-by-step construction**:",
        *step_lines,
        "",
        f"3. **Final {'routes' if multi_route else 'route'}**:",
        answer,
    ]

    think_content = "\n".join(think_parts)
    return f"<think>\n{think_content}\n</think>\n{answer}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def process_record(record: dict) -> dict | None:
    """处理单条 solutions 记录，返回 chains 格式的记录。"""
    pt = record["problem_type"]
    multi_route = pt in ("cvrp", "vrptw")
    solution = record["solution"]
    user_prompt = record["prompt"]["user"]
    system_prompt = record["prompt"]["system"]

    instance = parse_instance_from_prompt(user_prompt, pt)
    if not instance:
        return None

    routes = parse_routes(solution, multi_route)
    if not routes:
        return None

    # 验证：所有客户节点都被访问
    n = instance["n"]
    all_customers = sorted(v for r in routes for v in r if v != 0)
    if all_customers != list(range(1, n + 1)):
        return None

    output = build_think_chain(pt, instance, routes)

    # 用新版 system prompt（带思维链格式指导）
    from problems_prompt import get_system_prompt
    new_system = get_system_prompt(pt)

    return {
        "id": record["id"],
        "problem_type": pt,
        "n": n,
        "prompt": {"system": new_system, "user": user_prompt},
        "output": output,
        "solver_distance": record.get("solver_distance"),
        "method": "template",
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="从求解器的解 programmatically 构造思维链 SFT 数据")
    parser.add_argument("--input", nargs="+", required=True,
                        help="输入的 solutions JSONL 文件路径")
    parser.add_argument("--output", required=True,
                        help="输出的 chains JSONL 文件路径")
    args = parser.parse_args()

    records = []
    for path in args.input:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    print(f"读取 {len(records)} 条求解器解")

    success, fail = 0, 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in records:
            result = process_record(rec)
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                success += 1
            else:
                fail += 1

    print(f"完成: {success} 条成功, {fail} 条失败")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
