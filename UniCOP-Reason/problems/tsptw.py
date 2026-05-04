"""TSPTW：带时间窗的旅行商问题（单路线）。"""

import numpy as np
from .base import ProblemBase
from utils.parse import parse_single_route


def _tn(n: int) -> float:
    """返回 n 个节点问题的参考时间跨度（约等于期望最优巡游长度）。"""
    table = {20: 10.9, 50: 26.6, 100: 52.6}
    return table.get(n, (n + 1) * 0.5214)


class TSPTW(ProblemBase):
    name = "tsptw"
    multi_route = False

    def generate_instance(self, n: int, rng: np.random.Generator) -> dict:
        coords = rng.uniform(0, 1, size=(n + 1, 2))
        tn = _tn(n)

        # 随机时间窗：开窗时间均匀分布于 [0, Tn]，窗宽为 Tn 的 50%~75%
        time_windows = np.zeros((n + 1, 2))
        time_windows[0] = [0.0, 1e9]
        l = rng.uniform(0.0, tn, size=n)
        width_factors = 0.5 + 0.25 * rng.uniform(size=n)
        u = l + tn * width_factors
        time_windows[1:, 0] = l
        time_windows[1:, 1] = u

        feasible_tour = _greedy_edf_tour(n, coords, time_windows)
        return {"n": n, "coords": coords, "time_windows": time_windows, "feasible_tour": feasible_tour}

    def build_prompt(self, instance: dict) -> list[dict]:
        n, coords, tw = instance["n"], instance["coords"], instance["time_windows"]
        lines = [f"Plan the optimal route for the following TSPTW instance ({n} customer nodes):\n"]
        lines.append("Node information (format: node ID: coordinates(x,y)  time window[earliest, latest]):")
        for i in range(n + 1):
            tag  = " (depot)" if i == 0 else ""
            late = "inf" if tw[i][1] > 1e8 else f"{tw[i][1]:.3f}"
            lines.append(
                f"  Node {i}{tag}: ({coords[i][0]:.3f}, {coords[i][1]:.3f})"
                f"  [{tw[i][0]:.3f}, {late}]"
            )
        # 输出格式已在 system prompt 中说明，不重复
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": "\n".join(lines)},
        ]

    def get_tour_distance(self, completion: str, instance: dict) -> float | None:
        route = parse_single_route(completion, instance["n"])
        if route is None:
            return None
        _, dist = _simulate(route, instance["coords"], instance["time_windows"])
        return dist

    def is_feasible(self, completion: str, instance: dict) -> bool:
        n, coords, tw = instance["n"], instance["coords"], instance["time_windows"]
        route = parse_single_route(completion, n)
        if route is None:
            return False
        if (route[0] != 0 or route[-1] != 0
                or set(route[1:-1]) != set(range(1, n + 1))
                or len(route[1:-1]) != n):
            return False
        satisfied, _ = _simulate(route, coords, tw)
        return satisfied == n


def _greedy_edf_tour(n, coords, tw):
    """最早截止时间优先贪心，尽力找可行巡游（用于 sanity_check）。"""
    coords = np.array(coords)
    remaining = list(range(1, n + 1))
    tour, t, pos = [0], 0.0, 0
    while remaining:
        reachable = [nd for nd in remaining
                     if t + np.linalg.norm(coords[nd] - coords[pos]) <= tw[nd][1]]
        node = (min(reachable, key=lambda x: tw[x][1])
                if reachable
                else min(remaining, key=lambda x: np.linalg.norm(coords[x] - coords[pos])))
        t = max(t + float(np.linalg.norm(coords[node] - coords[pos])), tw[node][0])
        tour.append(node)
        remaining.remove(node)
        pos = node
    tour.append(0)
    return tour


def _simulate(route, coords, tw):
    """返回 (在时间窗内到达的客户节点数, 总距离)。"""
    coords, tw = np.array(coords), np.array(tw)
    current_time, satisfied, distance = 0.0, 0, 0.0
    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        travel        = float(np.linalg.norm(coords[nxt] - coords[curr]))
        current_time += travel
        distance     += travel
        if nxt != 0:
            if current_time < tw[nxt][0]:
                current_time = tw[nxt][0]
            if current_time <= tw[nxt][1]:
                satisfied += 1
    return satisfied, distance


_SYSTEM = """You are a route planning expert solving the Travelling Salesman Problem with Time Windows (TSPTW).
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
Route: 0 -> A -> B -> C -> ... -> 0"""
