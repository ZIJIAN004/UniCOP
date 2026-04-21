"""
TSPDL：带 Draft Limit 的旅行商问题（单路线）。

Draft 模型：
  - 车辆（船）满载出发，initial_load = capacity
  - 在节点 i 卸货 demand[i]，当前载重减少
  - 约束：到达节点 i 时的当前载重 ≤ draft_limit[i]
  - 这意味着 draft_limit 小的节点必须在已卸货较多之后访问
"""

import numpy as np
from .base import ProblemBase
from utils.parse import parse_single_route


class TSPDL(ProblemBase):
    name = "tspdl"
    multi_route = False

    def generate_instance(self, n: int, rng: np.random.Generator) -> dict:
        coords  = rng.uniform(0, 1, size=(n + 1, 2))
        Q = n  # 总容量 = 客户数（单位需求）

        # 需求：depot=0，所有客户=1（单位需求）
        demands = np.zeros(n + 1, dtype=float)
        demands[1:] = 1.0

        # Draft limit：depot 无限制，50% 客户随机受限，其余不受限
        draft_limits = np.full(n + 1, float(Q))
        draft_limits[0] = 1e9

        num_constrained = max(1, round(n * 0.5))

        # 拒绝采样：保证存在可行访问顺序
        feasible = False
        while not feasible:
            idx = rng.permutation(n)[:num_constrained]  # 随机选受限节点（0-indexed 客户）
            limits = rng.integers(1, Q, size=num_constrained)  # draft in [1, Q-1]
            cnt = np.bincount(limits, minlength=Q)
            feasible = (np.cumsum(cnt) <= np.arange(Q)).all()

        for k, node_idx in enumerate(idx):
            draft_limits[1 + node_idx] = float(limits[k])

        # 可行巡游：按 draft_limit 降序访问（最严格约束排最后）
        order = sorted(range(1, n + 1), key=lambda x: -draft_limits[x])
        feasible_tour = [0] + order + [0]

        return {
            "n": n, "coords": coords,
            "demands": demands, "capacity": float(Q),
            "draft_limits": draft_limits, "feasible_tour": feasible_tour,
        }

    def build_prompt(self, instance: dict) -> list[dict]:
        n, coords   = instance["n"], instance["coords"]
        demands, dl = instance["demands"], instance["draft_limits"]
        capacity    = instance["capacity"]
        lines = [
            f"Plan the route for the following TSPDL instance ({n} customer nodes, initial load={capacity:.1f}):\n",
            "Node information (format: node ID: coordinates(x,y)  unload  draft_limit):",
        ]
        for i in range(n + 1):
            tag = " (depot)" if i == 0 else ""
            dl_str = "inf" if dl[i] > 1e8 else f"{dl[i]:.1f}"
            lines.append(
                f"  Node {i}{tag}: ({coords[i][0]:.3f}, {coords[i][1]:.3f})"
                f"  unload={demands[i]:.0f}  draft_limit={dl_str}"
            )
        lines.append(
            "\nNote: nodes with a higher draft_limit (looser restriction) should generally be visited earlier."
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
        _, dist = _simulate(route, instance["coords"],
                            instance["demands"], instance["draft_limits"],
                            instance["capacity"])
        return dist

    def is_feasible(self, completion: str, instance: dict) -> bool:
        n, coords    = instance["n"], instance["coords"]
        demands, dl  = instance["demands"], instance["draft_limits"]
        capacity     = instance["capacity"]
        route = parse_single_route(completion, n)
        if route is None:
            return False
        if (route[0] != 0 or route[-1] != 0
                or set(route[1:-1]) != set(range(1, n + 1))
                or len(route[1:-1]) != n):
            return False
        satisfied, _ = _simulate(route, coords, demands, dl, capacity)
        return satisfied == n


def _simulate(route, coords, demands, draft_limits, capacity):
    """返回 (满足 draft_limit 的客户节点数, 总距离)。"""
    coords, demands, dl = np.array(coords), np.array(demands), np.array(draft_limits)
    current_load = float(capacity)
    satisfied, distance = 0, 0.0
    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        distance += float(np.linalg.norm(coords[nxt] - coords[curr]))
        if nxt != 0:
            if current_load <= dl[nxt]:
                satisfied += 1
            current_load -= demands[nxt]
    return satisfied, distance


_SYSTEM = """You are a logistics route planning expert solving the Travelling Salesman Problem with Draft Limits (TSPDL).
Rules:
- Start from node 0 (depot), visit all customer nodes exactly once, and return to node 0
- Travel time between nodes = Euclidean distance
- The vehicle departs fully loaded (initial load = total capacity); load decreases as cargo is unloaded at each customer
- Upon arriving at a node (before unloading), current load must be <= that node's draft_limit
- Objective: minimize total travel distance
Key insight: A node with a smaller draft_limit requires a lower load upon arrival, so it must be visited after enough cargo has already been unloaded. Visit high draft_limit nodes first, most constrained nodes last.
Before answering, think through the problem in <think>...</think>. Consider how the order in which nodes are visited determines the load level upon arrival, and how draft limits interact with that progression. Monotone constraint ordering or greedy sequencing by limit may be relevant.
After completing your analysis, output in the following format:
Route: 0 -> A -> B -> C -> ... -> 0"""
