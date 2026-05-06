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
            dems[nid] = round(nd.get("demand", 0.0), 2)
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

    # 构建特征分析段
    lines = [
        f"Depot at ({depot[0]:.3f}, {depot[1]:.3f}). {n_customers} customers.",
        f"Total demand: {total_demand:.2f}. Capacity: {capacity:.2f}.",
        f"Min vehicles needed: ceil({total_demand:.2f}/{capacity:.2f})={min_vehicles}. "
        f"Using {n_routes} routes.",
        f"Cluster analysis:",
    ]
    for ci in cluster_info:
        nodes_str = ", ".join(str(n) for n in ci["nodes"])
        lines.append(
            f"  Route {ci['idx']}: [{nodes_str}] "
            f"{len(ci['nodes'])} nodes, "
            f"mean_angle={ci['angle']:.0f}°, span={ci['angle_span']:.0f}°, "
            f"avg_d0={ci['avg_dist']:.3f}"
        )
    lines.append(
        f"Plan: serve each cluster as one route, visit by nearest-neighbor within cluster. "
        f"End route and return to depot when remaining capacity cannot serve any unvisited node in cluster."
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


def _build_feasible_str(coords, curr, candidates, nxt, k=3):
    """构建 feasible 候选列表字符串，必含 nxt，最多 k 个，超出加省略号。"""
    all_feasible = [(n, _dist(coords, curr, n)) for n in candidates]
    all_feasible.sort(key=lambda x: x[1])
    total_feasible = len(all_feasible)

    # 确保 nxt 在列表中
    shown = []
    nxt_in_top = False
    for n, d in all_feasible[:k]:
        shown.append((n, d))
        if n == nxt:
            nxt_in_top = True
    if not nxt_in_top:
        nxt_d = _dist(coords, curr, nxt)
        if len(shown) >= k:
            shown[-1] = (nxt, nxt_d)
        else:
            shown.append((nxt, nxt_d))

    return shown, total_feasible


def build_steps_tsp(route: list[int], coords: np.ndarray) -> list[str]:
    """TSP: 先列 feasible 候选，最后 → select N"""
    steps = []
    total_dist = 0.0
    unvisited = set(route[1:-1])

    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        d = _dist(coords, curr, nxt)
        total_dist += d

        if nxt == 0 and i == len(route) - 2:
            steps.append(
                f"[{i}] from {curr}, total={total_dist - d:.2f} "
                f"→ return depot (d={d:.3f}, total={total_dist:.2f})"
            )
            break

        unvisited.discard(nxt)

        shown, total_feasible = _build_feasible_str(
            coords, curr, unvisited | {nxt}, nxt, k=3
        )
        parts = [f"{n}(d={nd:.3f})" for n, nd in shown]
        feasible_str = ", ".join(parts)
        if total_feasible > 3:
            feasible_str += ", ..."

        steps.append(
            f"[{i}] from {curr}, total={total_dist - d:.2f} "
            f"| feasible: {feasible_str} → select {nxt}"
        )

        if i > 0 and i % 10 == 0 and unvisited:
            nodes_str = ", ".join(str(v) for v in sorted(unvisited))
            steps.append(f"Unvisited: {{{nodes_str}}}")

    return steps


def build_steps_cvrp(routes: list[list[int]], coords: np.ndarray,
                     demands: np.ndarray, capacity: float) -> list[str]:
    """CVRP: 先算 cap，列 feasible 候选，最后 → select N"""
    steps = []
    all_unvisited = set()
    for route in routes:
        all_unvisited.update(v for v in route if v != 0)

    prev_dem = 0.0  # 上一步选择的 demand（用于本步开头的 cap 计算）
    for r_idx, route in enumerate(routes):
        nodes_str = ", ".join(str(v) for v in sorted(all_unvisited))
        steps.append(f"Unvisited: {{{nodes_str}}}")

        cap_remaining = capacity
        step_in_route = 0
        prev_dem = 0.0
        for i in range(len(route) - 1):
            curr, nxt = route[i], route[i + 1]
            step_in_route += 1

            # 构建 cap 状态字符串
            if i == 0:
                cap_str = f"cap={capacity:.2f}"
            else:
                cap_str = f"cap={cap_remaining + prev_dem:.2f}-{prev_dem:.2f}={cap_remaining:.2f}"

            if nxt == 0:
                d = _dist(coords, curr, 0)
                if not all_unvisited:
                    steps.append(
                        f"[R{r_idx+1},{step_in_route}] {cap_str} "
                        f"→ all customers served, return depot (d={d:.3f})"
                    )
                else:
                    # 检查是 Case A（被迫）还是 Case B（策略）
                    feasible_nodes = [
                        v for v in all_unvisited if demands[v] <= cap_remaining
                    ]
                    if not feasible_nodes:
                        # Case A: 无可行节点
                        check_parts = [
                            f"{v}(dem={demands[v]:.2f}>{cap_remaining:.2f})"
                            for v in sorted(all_unvisited)[:5]
                        ]
                        if len(all_unvisited) > 5:
                            check_parts.append("...")
                        steps.append(
                            f"[R{r_idx+1},{step_in_route}] {cap_str} "
                            f"| check: {', '.join(check_parts)} "
                            f"→ no feasible → return depot (d={d:.3f})"
                        )
                    else:
                        # Case B: 策略性回depot
                        shown, total_f = _build_feasible_str(
                            coords, curr, feasible_nodes, feasible_nodes[0], k=3
                        )
                        parts = []
                        for n, nd in shown:
                            nc = cap_remaining - demands[n]
                            parts.append(f"{n}(d={nd:.3f},dem={demands[n]:.2f},cap→{nc:.2f})")
                        feasible_str = ", ".join(parts)
                        if total_f > 3:
                            feasible_str += ", ..."
                        steps.append(
                            f"[R{r_idx+1},{step_in_route}] {cap_str} "
                            f"| feasible: {feasible_str} "
                            f"→ remaining nodes better served by new route, return depot (d={d:.3f})"
                        )
                break

            d = _dist(coords, curr, nxt)
            dem = demands[nxt]
            prev_dem = dem

            # 构建 feasible 候选
            feasible_candidates = [
                v for v in (all_unvisited | {nxt}) if demands[v] <= cap_remaining
            ]
            shown, total_f = _build_feasible_str(
                coords, curr, feasible_candidates, nxt, k=3
            )
            parts = []
            for n, nd in shown:
                nc = cap_remaining - demands[n]
                parts.append(f"{n}(d={nd:.3f},dem={demands[n]:.2f},cap→{nc:.2f})")
            feasible_str = ", ".join(parts)
            if total_f > 3:
                feasible_str += ", ..."

            cap_remaining -= dem
            all_unvisited.discard(nxt)

            steps.append(
                f"[R{r_idx+1},{step_in_route}] {cap_str} "
                f"| feasible: {feasible_str} → select {nxt}"
            )

    return steps


def build_steps_tsptw(route: list[int], coords: np.ndarray,
                      tw: np.ndarray) -> list[str]:
    """TSPTW: 先列 feasible 候选（含 arr/slack），最后 → select N"""
    steps = []
    current_time = 0.0
    unvisited = set(route[1:-1])
    prev_wait_str = ""  # 上一步的等待信息（在本步开头显示）

    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        t_depart = current_time
        d = _dist(coords, curr, nxt)
        arr = t_depart + d

        if nxt == 0:
            steps.append(
                f"[{i}] t={t_depart:.2f} from {curr}{prev_wait_str} "
                f"→ return depot (d={d:.3f})"
            )
            break

        unvisited.discard(nxt)

        # 构建 feasible 候选：可达（arr <= deadline）的节点
        feasible_candidates = []
        for v in (unvisited | {nxt}):
            vd = _dist(coords, curr, v)
            v_arr = t_depart + vd
            if v_arr <= tw[v][1]:
                v_slack = tw[v][1] - v_arr
                feasible_candidates.append((v, vd, v_arr, v_slack))
        feasible_candidates.sort(key=lambda x: x[1])
        total_feasible = len(feasible_candidates)

        # 确保 nxt 在 top 3
        shown = []
        nxt_in_top = False
        for v, vd, v_arr, v_slack in feasible_candidates[:3]:
            shown.append((v, vd, v_arr, v_slack))
            if v == nxt:
                nxt_in_top = True
        if not nxt_in_top:
            nxt_arr = arr
            nxt_slack = tw[nxt][1] - arr
            if len(shown) >= 3:
                shown[-1] = (nxt, d, nxt_arr, nxt_slack)
            else:
                shown.append((nxt, d, nxt_arr, nxt_slack))

        parts = [
            f"{v}(d={vd:.3f},arr={v_arr:.2f},slack={v_slack:.2f})"
            for v, vd, v_arr, v_slack in shown
        ]
        feasible_str = ", ".join(parts)
        if total_feasible > 3:
            feasible_str += ", ..."

        # 可达性（从选择的节点出发）
        if arr < tw[nxt][0]:
            effective_time = tw[nxt][0]
            prev_wait_str = f" (arr={arr:.2f}, wait {tw[nxt][0] - arr:.2f})"
        else:
            effective_time = arr
            prev_wait_str = ""

        feasible_from_nxt = sum(
            1 for v in unvisited
            if effective_time + _dist(coords, nxt, v) <= tw[v][1]
        )
        reach_str = f" #reachable={feasible_from_nxt}/{len(unvisited)}" if unvisited else ""

        steps.append(
            f"[{i}] t={t_depart:.2f} from {curr}{prev_wait_str if i > 0 else ''} "
            f"| feasible: {feasible_str}{reach_str} → select {nxt}"
        )

        current_time = effective_time

        if i > 0 and i % 10 == 0 and unvisited:
            nodes_str = ", ".join(str(v) for v in sorted(unvisited))
            steps.append(f"Unvisited: {{{nodes_str}}}")

        # 重置 wait_str（已在本步开头用过，下一步的 wait 在 nxt 处理后设置）
        # prev_wait_str 已在上面正确设置

    return steps


def build_steps_vrptw(routes: list[list[int]], coords: np.ndarray,
                      tw: np.ndarray) -> list[str]:
    """VRPTW: 先列 feasible 候选（含 arr/slack），最后 → select N"""
    steps = []
    all_unvisited = set()
    for route in routes:
        all_unvisited.update(v for v in route if v != 0)

    for r_idx, route in enumerate(routes):
        nodes_str = ", ".join(str(v) for v in sorted(all_unvisited))
        steps.append(f"Unvisited: {{{nodes_str}}}")

        current_time = 0.0
        step_in_route = 0
        prev_wait_str = ""
        for i in range(len(route) - 1):
            curr, nxt = route[i], route[i + 1]
            t_depart = current_time
            d = _dist(coords, curr, nxt)
            arr = t_depart + d
            step_in_route += 1

            if nxt == 0:
                if not all_unvisited:
                    steps.append(
                        f"[R{r_idx+1},{step_in_route}] t={t_depart:.2f} from {curr}{prev_wait_str} "
                        f"→ all customers served, return depot (d={d:.3f})"
                    )
                else:
                    # 检查 Case A vs Case B
                    feasible_nodes = [
                        v for v in all_unvisited
                        if t_depart + _dist(coords, curr, v) <= tw[v][1]
                    ]
                    if not feasible_nodes:
                        # Case A: 无可达节点
                        check_parts = []
                        for v in sorted(all_unvisited)[:5]:
                            v_arr = t_depart + _dist(coords, curr, v)
                            check_parts.append(
                                f"{v}(arr={v_arr:.2f}>deadline={tw[v][1]:.2f})"
                            )
                        if len(all_unvisited) > 5:
                            check_parts.append("...")
                        steps.append(
                            f"[R{r_idx+1},{step_in_route}] t={t_depart:.2f} from {curr}{prev_wait_str} "
                            f"| check: {', '.join(check_parts)} "
                            f"→ no feasible → return depot (d={d:.3f})"
                        )
                    else:
                        # Case B: 策略性回depot
                        cands = []
                        for v in feasible_nodes:
                            vd = _dist(coords, curr, v)
                            v_arr = t_depart + vd
                            v_slack = tw[v][1] - v_arr
                            cands.append((v, vd, v_arr, v_slack))
                        cands.sort(key=lambda x: x[1])
                        shown = cands[:3]
                        parts = [
                            f"{v}(d={vd:.3f},arr={v_arr:.2f},slack={v_slack:.2f})"
                            for v, vd, v_arr, v_slack in shown
                        ]
                        feasible_str = ", ".join(parts)
                        if len(feasible_nodes) > 3:
                            feasible_str += ", ..."
                        steps.append(
                            f"[R{r_idx+1},{step_in_route}] t={t_depart:.2f} from {curr}{prev_wait_str} "
                            f"| feasible: {feasible_str} "
                            f"→ remaining nodes better served by new route, return depot (d={d:.3f})"
                        )
                break

            all_unvisited.discard(nxt)

            # 构建 feasible 候选
            feasible_candidates = []
            for v in (all_unvisited | {nxt}):
                vd = _dist(coords, curr, v)
                v_arr = t_depart + vd
                if v_arr <= tw[v][1]:
                    v_slack = tw[v][1] - v_arr
                    feasible_candidates.append((v, vd, v_arr, v_slack))
            feasible_candidates.sort(key=lambda x: x[1])
            total_feasible = len(feasible_candidates)

            # 确保 nxt 在 top 3
            shown = []
            nxt_in_top = False
            for v, vd, v_arr, v_slack in feasible_candidates[:3]:
                shown.append((v, vd, v_arr, v_slack))
                if v == nxt:
                    nxt_in_top = True
            if not nxt_in_top:
                nxt_slack = tw[nxt][1] - arr
                if len(shown) >= 3:
                    shown[-1] = (nxt, d, arr, nxt_slack)
                else:
                    shown.append((nxt, d, arr, nxt_slack))

            parts = [
                f"{v}(d={vd:.3f},arr={v_arr:.2f},slack={v_slack:.2f})"
                for v, vd, v_arr, v_slack in shown
            ]
            feasible_str = ", ".join(parts)
            if total_feasible > 3:
                feasible_str += ", ..."

            # 等待处理
            if arr < tw[nxt][0]:
                prev_wait_str = f" (arr={arr:.2f}, wait {tw[nxt][0] - arr:.2f})"
                current_time = tw[nxt][0]
            else:
                prev_wait_str = ""
                current_time = arr

            steps.append(
                f"[R{r_idx+1},{step_in_route}] t={t_depart:.2f} from {curr}"
                f"{prev_wait_str if i > 0 else ''} "
                f"| feasible: {feasible_str} → select {nxt}"
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


def _build_verification(routes: list[list[int]], n: int, multi_route: bool) -> list[str]:
    """构建验证步骤：核对所有节点恰好覆盖一次。"""
    lines = []
    if multi_route:
        route_summaries = []
        for i, route in enumerate(routes):
            customers = sorted(v for v in route if v != 0)
            nodes_str = ",".join(str(v) for v in customers)
            route_summaries.append(f"R{i+1}:{{{nodes_str}}}={len(customers)}")
        lines.append(" | ".join(route_summaries))
        total = sum(len([v for v in r if v != 0]) for r in routes)
        lines.append(f"Total: {total}/{n} customers. " +
                     ("✓ All covered, no duplicates." if total == n else "✗ Error."))
    else:
        customers = sorted(v for v in routes[0] if v != 0)
        lines.append(f"Visited: {{{','.join(str(v) for v in customers)}}} = {len(customers)}/{n}")
        lines.append("✓ All covered." if len(customers) == n else "✗ Error.")
    return lines


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

    # Verification
    n = instance["n"]
    verify_lines = _build_verification(routes, n, multi_route)

    # Assemble
    think_parts = [
        f"1. **Strategy**: {strategy}",
        "",
        "2. **Step-by-step construction**:",
        *step_lines,
        "",
        "3. **Verification**:",
        *verify_lines,
        "",
        f"4. **Final {'routes' if multi_route else 'route'}**:",
        answer,
    ]

    think_content = "\n".join(think_parts)
    return f"<think>\n{think_content}\n</think>\n{answer}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def _rewrite_demand_precision(user_prompt: str) -> str:
    """将 user prompt 中的 demand=X.XXXX 重写为 demand=X.XX（2位小数）。"""
    return re.sub(
        r"demand=([\d.]+)",
        lambda m: f"demand={round(float(m.group(1)), 2):.2f}",
        user_prompt,
    )


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

    # CVRP: 重写 user prompt 中的 demand 精度为 2 位
    if pt == "cvrp":
        user_prompt = _rewrite_demand_precision(user_prompt)

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


def process_record_reject(record: dict, rng) -> dict | None:
    """新格式下 reject 已内嵌在 build_steps_cvrp/vrptw 中（Case A/B），
    此函数保留接口兼容性，直接复用 process_record。"""
    return None


def main():
    parser = argparse.ArgumentParser(
        description="从求解器的解 programmatically 构造思维链 SFT 数据")
    parser.add_argument("--input", nargs="+", required=True,
                        help="输入的 solutions JSONL 文件路径")
    parser.add_argument("--output", required=True,
                        help="输出的 chains JSONL 文件路径")
    parser.add_argument("--reject_ratio", type=float, default=0.0,
                        help="额外生成带拒绝事件 chain 的比例 (0-1)，0 表示不生成")
    args = parser.parse_args()

    records = []
    for path in args.input:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    print(f"读取 {len(records)} 条求解器解")

    rng = np.random.default_rng(42)
    success, fail, reject_count = 0, 0, 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in records:
            use_reject = (args.reject_ratio > 0 and rng.random() < args.reject_ratio)

            if use_reject:
                # 尝试生成 reject chain 替换 solver chain
                reject_result = process_record_reject(rec, rng)
                if reject_result:
                    out_f.write(json.dumps(reject_result, ensure_ascii=False) + "\n")
                    reject_count += 1
                    continue
                # 生成失败（无可注入节点），fallback 到 solver chain

            result = process_record(rec)
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                success += 1
            else:
                fail += 1

    total = success + reject_count
    pct = reject_count / total * 100 if total else 0
    print(f"完成: {success} 条 solver chain + {reject_count} 条 reject chain = {total} 条")
    print(f"Reject 占比: {pct:.1f}% (目标 {args.reject_ratio*100:.0f}%)")
    if fail:
        print(f"失败: {fail} 条")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
