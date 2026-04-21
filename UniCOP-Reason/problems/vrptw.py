"""VRPTW：带时间窗的车辆路径问题（多路线，无容量约束）。"""

import numpy as np
from .base import ProblemBase
from utils.parse import parse_multi_route


class VRPTW(ProblemBase):
    name = "vrptw"
    multi_route = True

    def generate_instance(self, n: int, rng: np.random.Generator) -> dict:
        coords = rng.uniform(0, 1, size=(n + 1, 2))

        # 时间窗：a_i >= dist(depot, i) 保证可行，多车直接派车即可
        time_windows = np.zeros((n + 1, 2))
        time_windows[0] = [0.0, 1e9]
        for i in range(1, n + 1):
            dist_depot = float(np.linalg.norm(coords[i] - coords[0]))
            a_i = dist_depot + rng.uniform(0.0, 0.2)
            width = rng.uniform(0.3, 0.5)
            time_windows[i] = [a_i, a_i + width]

        # 每个客户单独一条路线，平凡可行解（用于 sanity_check）
        feasible_routes = [[0, i, 0] for i in range(1, n + 1)]

        return {
            "n": n, "coords": coords,
            "time_windows": time_windows,
            "feasible_routes": feasible_routes,
        }

    def build_prompt(self, instance: dict) -> list[dict]:
        n, coords, tw = instance["n"], instance["coords"], instance["time_windows"]
        lines = [f"Plan routes for the following VRPTW instance ({n} customer nodes):\n"]
        lines.append("Node information (format: node ID: coordinates(x,y)  time window[earliest, latest]):")
        for i in range(n + 1):
            tag  = " (depot)" if i == 0 else ""
            late = "inf" if tw[i][1] > 1e8 else f"{tw[i][1]:.3f}"
            lines.append(
                f"  Node {i}{tag}: ({coords[i][0]:.3f}, {coords[i][1]:.3f})"
                f"  [{tw[i][0]:.3f}, {late}]"
            )
        # 约束和输出格式已在 system prompt 中说明，不重复
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": "\n".join(lines)},
        ]

    def get_tour_distance(self, completion: str, instance: dict) -> float | None:
        routes = parse_multi_route(completion, instance["n"])
        if routes is None:
            return None
        return sum(_route_distance(r, instance["coords"]) for r in routes)

    def is_feasible(self, completion: str, instance: dict) -> bool:
        n, coords, tw = instance["n"], instance["coords"], instance["time_windows"]
        routes = parse_multi_route(completion, n)
        if routes is None:
            return False
        all_customers = [v for r in routes for v in r if v != 0]
        if (not all(r[0] == 0 and r[-1] == 0 for r in routes)
                or sorted(all_customers) != list(range(1, n + 1))):
            return False
        return all(_route_feasible(r, coords, tw) for r in routes)


def _route_feasible(route, coords, tw) -> bool:
    """判断单条路线是否全程满足时间窗（路线级 c_core）。"""
    coords, tw = np.array(coords), np.array(tw)
    current_time = 0.0
    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        current_time += float(np.linalg.norm(coords[nxt] - coords[curr]))
        if nxt != 0:
            if current_time < tw[nxt][0]:
                current_time = tw[nxt][0]
            if current_time > tw[nxt][1]:
                return False
    return True


def _route_distance(route, coords) -> float:
    coords = np.array(coords)
    return sum(
        float(np.linalg.norm(coords[route[i + 1]] - coords[route[i]]))
        for i in range(len(route) - 1)
    )


_SYSTEM = """You are a logistics scheduling expert solving the Vehicle Routing Problem with Time Windows (VRPTW).
Rules:
- Multiple vehicles depart from node 0 (depot); each vehicle visits a subset of customers and returns to node 0
- All customer nodes are visited exactly once
- Travel time between nodes = Euclidean distance
- Each customer node has a time window [earliest, latest]: arrival time must be <= latest
- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue
- Objective: minimize total travel distance across all routes
Before answering, think through the problem in <think>...</think>. Consider how time window compatibility and geographic proximity together influence which customers can share a route and in what order. Insertion heuristics or time-window-first grouping may offer useful intuitions.
After completing your analysis, output in the following format (one route per line, nodes in visit order):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0"""
