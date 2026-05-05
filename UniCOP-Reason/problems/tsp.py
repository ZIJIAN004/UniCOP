"""TSP：无约束旅行商问题。"""

import numpy as np
from .base import ProblemBase
from utils.parse import parse_single_route


class TSP(ProblemBase):
    name = "tsp"
    multi_route = False

    # ── 生成实例 ──────────────────────────────────────────────────────

    def generate_instance(self, n: int, rng: np.random.Generator) -> dict:
        coords = rng.uniform(0, 1, size=(n + 1, 2))
        # TSP 无约束，任意顺序都可行；用最近邻生成一条参考路径
        perm = rng.permutation(n) + 1
        feasible_tour = [0] + perm.tolist() + [0]
        return {"n": n, "coords": coords, "feasible_tour": feasible_tour}

    # ── Prompt ────────────────────────────────────────────────────────

    def build_prompt(self, instance: dict) -> list[dict]:
        n, coords = instance["n"], instance["coords"]
        lines = [f"Plan the shortest route for the following TSP instance ({n} customer nodes):\n"]
        lines.append("Node coordinates (format: node ID: (x, y)):")
        for i in range(n + 1):
            tag = " (depot)" if i == 0 else ""
            lines.append(f"  Node {i}{tag}: ({coords[i][0]:.3f}, {coords[i][1]:.3f})")
        # 输出格式已在 system prompt 中说明，不重复
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": "\n".join(lines)},
        ]

    def get_tour_distance(self, completion: str, instance: dict) -> float | None:
        route = parse_single_route(completion, instance["n"])
        if route is None:
            return None
        return self.total_distance(route, instance["coords"])

    def is_feasible(self, completion: str, instance: dict) -> bool:
        n = instance["n"]
        route = parse_single_route(completion, n)
        if route is None:
            return False
        return (route[0] == 0 and route[-1] == 0
                and set(route[1:-1]) == set(range(1, n + 1))
                and len(route[1:-1]) == n)


_SYSTEM = """You are a route planning expert solving the Travelling Salesman Problem (TSP).
Rules: Starting from node 0, visit all customer nodes exactly once and return to node 0, minimizing total distance.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these four sections in order:

1. **Strategy**: Analyze the node distribution and state your approach. Examples: "Angular sweep counterclockwise from depot", "Convex hull first then insert interior nodes", "Divide into 3 geographic segments and connect". Reference specific node groups or spatial features.

2. **Step-by-step construction**: Build the route one node at a time. Each step format:
   [step] at N → M (d=X.XXX, total=X.XX) | alt: A(X.XX), B(X.XX)
   - d = distance from current to chosen node
   - total = cumulative route distance so far
   - alt = 2-3 nearest unvisited alternatives with distances
   Every 10 steps, insert a line: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.

3. **Verification**: List all visited nodes and confirm total count equals n. Format: "Visited: {1,2,...} = N/N" then "✓ All covered."

4. **Final route**: Write the complete route in "Route: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final route (copied from think):
Route: 0 -> A -> B -> C -> ... -> 0"""
