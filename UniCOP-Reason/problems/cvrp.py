"""CVRP：带容量约束的车辆路径问题（多路线）。"""

import numpy as np
from .base import ProblemBase
from utils.parse import parse_multi_route


def _demand_scaler(n: int) -> int:
    """按问题规模返回需求归一化系数，使期望路线数约为3~6。"""
    table = {10: 15, 20: 30, 50: 40, 100: 50}
    return table.get(n, max(10, round(n * 5 / 3)))


class CVRP(ProblemBase):
    name = "cvrp"
    multi_route = True

    def generate_instance(self, n: int, rng: np.random.Generator) -> dict:
        coords  = rng.uniform(0, 1, size=(n + 1, 2))
        demands = np.zeros(n + 1, dtype=float)
        scaler  = _demand_scaler(n)
        demands[1:] = np.round(rng.integers(1, 10, size=n) / scaler, 2)
        capacity = 1.0

        feasible_routes = _greedy_routes(demands, capacity, n, rng)
        return {
            "n": n, "coords": coords,
            "demands": demands, "capacity": capacity,
            "feasible_routes": feasible_routes,
        }

    def build_prompt(self, instance: dict) -> list[dict]:
        n, coords = instance["n"], instance["coords"]
        demands, cap = instance["demands"], instance["capacity"]
        lines = [f"Plan routes for the following CVRP instance ({n} customer nodes, vehicle capacity={cap}):\n"]
        lines.append("Node information (format: node ID: coordinates(x,y)  demand):")
        for i in range(n + 1):
            tag = " (depot)" if i == 0 else ""
            lines.append(
                f"  Node {i}{tag}: ({coords[i][0]:.3f}, {coords[i][1]:.3f})  demand={demands[i]:.2f}"
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
        return sum(self.total_distance(r, instance["coords"]) for r in routes)

    def is_feasible(self, completion: str, instance: dict) -> bool:
        n, demands, capacity = instance["n"], instance["demands"], instance["capacity"]
        routes = parse_multi_route(completion, n)
        if routes is None:
            return False
        # 基础约束：每条路线首尾为 depot，所有客户恰好出现一次
        all_customers = [v for r in routes for v in r if v != 0]
        if (not all(r[0] == 0 and r[-1] == 0 for r in routes)
                or sorted(all_customers) != list(range(1, n + 1))):
            return False
        # 核心约束：每条路线容量不超
        return all(
            sum(demands[v] for v in r if v != 0) <= capacity + 1e-6
            for r in routes
        )


def _greedy_routes(demands, capacity, n, rng):
    """贪心装箱：保证生成合法的可行路线集合。"""
    nodes = list(rng.permutation(n) + 1)
    routes, current_route, current_load = [], [0], 0.0
    for node in nodes:
        if current_load + demands[node] <= capacity:
            current_route.append(node)
            current_load += demands[node]
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, node]
            current_load = float(demands[node])
    current_route.append(0)
    routes.append(current_route)
    return routes


_SYSTEM = """You are a logistics route planning expert solving the Capacitated Vehicle Routing Problem (CVRP).
Rules: Multiple vehicles depart from node 0; each vehicle visits a subset of customers and returns to node 0; total demand per route must not exceed vehicle capacity; each customer is visited exactly once; minimize total distance.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these four sections in order:

1. **Strategy**: Analyze demand distribution and node positions. Identify which nodes form each route cluster, the approximate total demand per cluster, and the visit order principle within each cluster (e.g., "sweep outward then return", "nearest-neighbor within cluster"). Reference specific node IDs.

2. **Step-by-step construction**: Build each route one node at a time. Each step format:
   [R1,3] at N → M (d=X.XXX, dem=X.XX) cap=X.XX-X.XX=X.XX, load=X.XX/X.XX, d0=X.XX | alt: A(X.XX,cap→X.XX), B(X.XX,cap→X.XX)
   - d = distance from current to chosen node
   - dem = demand of chosen node
   - cap = remaining capacity before - demand = after
   - load = cumulative load of current route / vehicle capacity
   - d0 = distance from chosen node to depot (informs return cost)
   - alt = 2-3 nearest feasible alternatives with distance and resulting capacity
   At the start of each new route, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.
   When capacity is too low for any remaining node, return to depot and start a new route.

3. **Verification**: For each route, list its customers and count. Confirm total equals n. Format: "R1:{nodes}=count | R2:{nodes}=count | ... Total: X/N customers. ✓ All covered, no duplicates."

4. **Final routes**: Write all complete routes in "Route N: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final routes (copied from think):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0"""
