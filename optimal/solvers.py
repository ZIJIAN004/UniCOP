"""
optimal/solvers.py
为 UniCOP-Reason 的问题实例求 (近) 最优解，作为 optimality gap 的分母基线。

求解器分配（固定，不混用）：
  TSP   → LKH        （RUNS 调大求最优；未提供 LKH_BIN 直接报错，不回退）
  CVRP  → PyVRP/HGS  （运行时长 timeout 调大求最优）
  TSPTW → PyVRP/HGS  （建模为单车辆 VRPTW）
  VRPTW → PyVRP/HGS
  TSPDL → 暂不支持    （PyVRP 不原生支持 draft limit，见 README.md）

求最优的旋钮：LKH 调大 lkh_runs；HGS 调大 timeout（MaxRuntime）。n≤100 时二者均
可达真实最优级别（注：LKH/HGS 为启发式，给出的是最优级/best-known，非可证明最优）。

关键口径：返回的 cost 一律用原始 [0,1] 坐标的欧氏边长重算，与 evaluate.py 的
prob.get_tour_distance 完全一致。求解器内部对坐标做整数缩放（×_COORD_SCALE）只用于
求“访问顺序”，最终 cost 不受缩放影响。
"""

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

# ── LKH 二进制（仅 TSP 使用；通过环境变量覆盖）────────────────────────────────
LKH_BIN = os.environ.get("LKH_BIN", "")

# 坐标缩放：LKH/PyVRP 使用整数距离，原始坐标在 [0,1]，放大以保留精度
_COORD_SCALE = 1_000_000
# PyVRP 内部使用 32-bit 整数，时间窗上限需限制在安全范围内（2e9 < 2^31-1）
_MAX_TW_SCALED = 2_000_000_000

SUPPORTED = ("tsp", "cvrp", "tsptw", "vrptw")


# ─────────────────────────────────────────────────────────────────────────────
# cost 计算（原始坐标欧氏边长之和，与 evaluate.py 口径一致）
# ─────────────────────────────────────────────────────────────────────────────

def _route_cost(route: list[int], coords: np.ndarray) -> float:
    return sum(
        float(np.linalg.norm(coords[route[i + 1]] - coords[route[i]]))
        for i in range(len(route) - 1)
    )


def _total_cost(routes: list[list[int]], coords: np.ndarray) -> float:
    return sum(_route_cost(r, coords) for r in routes)


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def solve_instance(problem_type: str, instance: dict, *,
                   timeout: int = 30, lkh_bin: str = LKH_BIN,
                   seed: int = 42, lkh_runs: int = 10) -> dict:
    """
    求单个实例的 (近) 最优解。求解器固定：TSP→LKH，其余→PyVRP/HGS。
    为逼近最优，把 LKH 的 RUNS(lkh_runs) 与 HGS 的运行时长(timeout) 调大即可。

    Args:
        timeout:   PyVRP/HGS 每实例运行时长(秒)，越大越接近最优；同时作 LKH 每实例时间上限。
        lkh_bin:   LKH 二进制路径（TSP 必需；为空则 TSP 直接报错，不回退）。
        lkh_runs:  LKH 独立运行次数，取最优；越大越稳。
    Returns dict:
      {
        "cost":     float | None,   # 近最优总距离（None 表示求解失败/不可行）
        "routes":   list[list[int]],# 0-indexed，每条以 depot(0) 首尾
        "solver":   str,            # "lkh" | "pyvrp"
        "feasible": bool,           # 求解器是否给出可行解
      }
    """
    pt = problem_type.lower()
    coords = np.asarray(instance["coords"], dtype=float)
    try:
        if pt == "tsp":
            if not lkh_bin:
                raise RuntimeError(
                    "TSP 固定用 LKH 求最优，但未提供 LKH 二进制。请 export LKH_BIN=/path/to/LKH "
                    "(与 UniCOP-Distill/lkh_solver.py 同一环境变量) 或用 --lkh_bin 指定。"
                )
            routes = [_solve_tsp_lkh(instance, lkh_bin, seed, timeout, runs=lkh_runs)]
            solver = "lkh"
        elif pt == "cvrp":
            routes, solver = _solve_cvrp_pyvrp(instance, timeout), "pyvrp"
        elif pt == "tsptw":
            routes, solver = [_solve_tsptw_pyvrp(instance, timeout)], "pyvrp"
        elif pt == "vrptw":
            routes, solver = _solve_vrptw_pyvrp(instance, timeout), "pyvrp"
        elif pt == "tspdl":
            raise NotImplementedError(
                "TSPDL 暂不支持（PyVRP 无 draft-limit 模型），见 optimal/README.md"
            )
        else:
            raise ValueError(f"未知问题类型: {problem_type}")

        if any(r is None for r in routes) or not routes:
            return {"cost": None, "routes": [], "solver": solver, "feasible": False}

        return {
            "cost": _total_cost(routes, coords),
            "routes": routes,
            "solver": solver,
            "feasible": True,
        }
    except Exception as e:  # noqa: BLE001 — 求解失败不应中断整批，记录后跳过
        return {"cost": None, "routes": [], "solver": pt, "feasible": False,
                "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# TSP — LKH
# ─────────────────────────────────────────────────────────────────────────────

def _solve_tsp_lkh(instance, lkh_bin, seed, timeout, runs=10) -> Optional[list[int]]:
    n = instance["n"]
    coords = np.asarray(instance["coords"], dtype=float)

    with tempfile.TemporaryDirectory() as tmp:
        tsp_f = os.path.join(tmp, "p.tsp")
        par_f = os.path.join(tmp, "p.par")
        tour_f = os.path.join(tmp, "p.tour")

        with open(tsp_f, "w") as f:
            f.write(f"NAME: tsp\nTYPE: TSP\nDIMENSION: {n + 1}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")
            for i in range(n + 1):
                x = int(coords[i][0] * _COORD_SCALE)
                y = int(coords[i][1] * _COORD_SCALE)
                f.write(f"{i + 1} {x} {y}\n")
            f.write("EOF\n")

        with open(par_f, "w") as f:
            f.write(f"PROBLEM_FILE = {tsp_f}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_f}\n")
            f.write(f"RUNS = {runs}\n")            # 多次独立运行取最优，越大越稳
            f.write(f"SEED = {seed}\n")
            f.write("TRACE_LEVEL = 0\n")
            f.write(f"TIME_LIMIT = {timeout}\n")   # 每实例时间上限(秒)

        # subprocess timeout 比 LKH TIME_LIMIT 宽裕 30s，防止写 tour 文件时被 kill
        result = subprocess.run([lkh_bin, par_f], capture_output=True,
                                text=True, timeout=timeout + 30)
        if result.returncode != 0:
            raise RuntimeError(f"LKH 返回非零状态码 {result.returncode}: "
                               f"{result.stderr[:300]}")

        tour = _parse_lkh_tour(tour_f)
        if tour is None:
            return None
        return _reroot_at_depot(tour, depot_lkh=1)


def _parse_lkh_tour(tour_f) -> Optional[list[int]]:
    """解析 LKH tour 文件，返回 1-indexed 节点列表（不含首尾 -1）。"""
    if not os.path.exists(tour_f):
        return None
    nodes, in_section = [], False
    with open(tour_f) as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_section = True
                continue
            if not in_section:
                continue
            if line in ("-1", "EOF"):
                break
            nodes.append(int(line))
    return nodes or None


def _reroot_at_depot(tour: list[int], depot_lkh: int = 1) -> list[int]:
    """旋转 tour 使其从 depot 开始，转 0-indexed，首尾补 depot(0)。"""
    idx = tour.index(depot_lkh)
    rotated = tour[idx:] + tour[:idx]
    route = [v - 1 for v in rotated]
    return [0] + route[1:] + [0]


# ─────────────────────────────────────────────────────────────────────────────
# PyVRP 公共辅助
# ─────────────────────────────────────────────────────────────────────────────

def _scaled_coords(coords: np.ndarray, n: int) -> list[tuple[int, int]]:
    return [(int(coords[i][0] * _COORD_SCALE), int(coords[i][1] * _COORD_SCALE))
            for i in range(n + 1)]


def _add_edges(m, locs, coords_scaled, with_duration=False):
    """为 PyVRP Model 添加所有 (i, j) 边，距离 = 缩放坐标的欧氏整数值。"""
    n = len(locs)
    for i in range(n):
        xi, yi = coords_scaled[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords_scaled[j]
            dist = int(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
            if with_duration:
                m.add_edge(locs[i], locs[j], distance=dist, duration=dist)
            else:
                m.add_edge(locs[i], locs[j], distance=dist)


def _pyvrp_solve(m, timeout):
    from pyvrp.stop import MaxRuntime
    result = m.solve(stop=MaxRuntime(timeout), display=False)
    if not result.is_feasible():
        return None
    routes = list(result.best.routes())
    return routes or None


# ─────────────────────────────────────────────────────────────────────────────
# CVRP — PyVRP/HGS
# ─────────────────────────────────────────────────────────────────────────────

def _solve_cvrp_pyvrp(instance, timeout) -> list:
    from pyvrp import Model

    n = instance["n"]
    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacity = instance["capacity"]
    S = _COORD_SCALE
    cs = _scaled_coords(coords, n)

    m = Model()
    depot = m.add_depot(x=cs[0][0], y=cs[0][1])
    clients = [
        m.add_client(x=cs[i][0], y=cs[i][1],
                     delivery=int(round(demands[i] * S)), service_duration=0)
        for i in range(1, n + 1)
    ]
    locs = [depot] + clients
    _add_edges(m, locs, cs, with_duration=False)
    m.add_vehicle_type(num_available=n, capacity=int(round(capacity * S)))

    routes = _pyvrp_solve(m, timeout)
    if routes is None:
        return [None]
    return [[0] + [int(v) for v in r] + [0] for r in routes]


# ─────────────────────────────────────────────────────────────────────────────
# TSPTW — PyVRP（单车辆 VRPTW）
# ─────────────────────────────────────────────────────────────────────────────

def _solve_tsptw_pyvrp(instance, timeout) -> Optional[list[int]]:
    from pyvrp import Model

    n = instance["n"]
    coords = np.asarray(instance["coords"], dtype=float)
    tw = np.asarray(instance["time_windows"], dtype=float)
    S = _COORD_SCALE
    cs = _scaled_coords(coords, n)

    m = Model()
    depot = m.add_depot(
        x=cs[0][0], y=cs[0][1],
        tw_early=int(tw[0][0] * S),
        tw_late=min(int(tw[0][1] * S), _MAX_TW_SCALED),
    )
    clients = [
        m.add_client(
            x=cs[i][0], y=cs[i][1], delivery=1,
            tw_early=int(tw[i][0] * S),
            tw_late=min(int(tw[i][1] * S), _MAX_TW_SCALED),
            service_duration=0,
        )
        for i in range(1, n + 1)
    ]
    locs = [depot] + clients
    _add_edges(m, locs, cs, with_duration=True)
    m.add_vehicle_type(num_available=1, capacity=n + 1)

    routes = _pyvrp_solve(m, timeout)
    if routes is None:
        return None
    return [0] + [int(v) for v in routes[0]] + [0]


# ─────────────────────────────────────────────────────────────────────────────
# VRPTW — PyVRP/HGS
# ─────────────────────────────────────────────────────────────────────────────

def _solve_vrptw_pyvrp(instance, timeout) -> list:
    from pyvrp import Model

    n = instance["n"]
    coords = np.asarray(instance["coords"], dtype=float)
    tw = np.asarray(instance["time_windows"], dtype=float)
    S = _COORD_SCALE
    cs = _scaled_coords(coords, n)

    m = Model()
    depot = m.add_depot(
        x=cs[0][0], y=cs[0][1],
        tw_early=int(tw[0][0] * S),
        tw_late=min(int(tw[0][1] * S), _MAX_TW_SCALED),
    )
    clients = [
        m.add_client(
            x=cs[i][0], y=cs[i][1], delivery=1,
            tw_early=int(tw[i][0] * S),
            tw_late=min(int(tw[i][1] * S), _MAX_TW_SCALED),
            service_duration=0,
        )
        for i in range(1, n + 1)
    ]
    locs = [depot] + clients
    _add_edges(m, locs, cs, with_duration=True)
    m.add_vehicle_type(num_available=n, capacity=n + 1)

    routes = _pyvrp_solve(m, timeout)
    if routes is None:
        return [None]
    return [[0] + [int(v) for v in r] + [0] for r in routes]
