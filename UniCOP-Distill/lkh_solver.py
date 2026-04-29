"""
lkh_solver.py
为各 COP 问题类型调用对应求解器，返回 UniCOP-Reason 格式的近最优解。

求解器分配：
  TSP   → LKH（快速，TSP 是 LKH 最擅长的场景）
  CVRP  → PyVRP（HGS，对 n=50/100 远快于 LKH-3，无 timeout 问题）
  TSPTW → PyVRP（1辆车 VRPTW，时间窗约束满足率高）
  VRPTW → PyVRP（原生支持，约束满足率高）

LKH 二进制路径配置（仅 TSP 使用，修改下方常量或通过环境变量覆盖）：
  LKH_BIN = LKH 路径
"""

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

# ── 二进制路径（在此处修改，或通过环境变量覆盖）────────────────────────────────
LKH_BIN  = os.environ.get("LKH_BIN", "/home/ntu/LKH/LKH")
LKH3_BIN = LKH_BIN  # 保持向后兼容，两者指向同一二进制

# 坐标缩放系数：LKH/PyVRP 使用整数距离，原始坐标在 [0,1]，放大以保留精度
_COORD_SCALE = 1_000_000
# PyVRP 内部使用 32-bit 整数，时间窗上限需限制在安全范围内（2e9 < 2^31-1）
_MAX_TW_SCALED = 2_000_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def solve(problem_type: str, instance: dict,
          lkh_bin: str = LKH_BIN,
          runs: int = 1, seed: int = 42, timeout: int = 60) -> Optional[str]:
    """
    统一入口：根据问题类型调用对应求解器。
    返回 UniCOP-Reason 格式的解字符串，失败返回 None。
    lkh_bin 仅用于 TSP；CVRP/TSPTW/VRPTW 均使用 PyVRP。
    """
    try:
        if problem_type == "tsp":
            return _solve_tsp(instance, lkh_bin, runs=runs, seed=seed, timeout=timeout)
        elif problem_type == "cvrp":
            return _solve_cvrp(instance, timeout=timeout)
        elif problem_type == "tsptw":
            return _solve_tsptw_pyvrp(instance, timeout=timeout)
        elif problem_type == "vrptw":
            return _solve_vrptw_pyvrp(instance, timeout=timeout)
        else:
            raise ValueError(f"未知问题类型: {problem_type}")
    except Exception as e:
        print(f"    [SOLVER ERROR] {problem_type}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TSP — LKH-2
# ─────────────────────────────────────────────────────────────────────────────

def _solve_tsp(instance, lkh_bin, runs=1, seed=42, timeout=60):
    n      = instance["n"]
    coords = np.array(instance["coords"])

    with tempfile.TemporaryDirectory() as tmp:
        tsp_f  = os.path.join(tmp, "p.tsp")
        par_f  = os.path.join(tmp, "p.par")
        tour_f = os.path.join(tmp, "p.tour")

        with open(tsp_f, "w") as f:
            f.write(f"NAME: tsp\nTYPE: TSP\nDIMENSION: {n+1}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")
            for i in range(n + 1):
                x, y = int(coords[i][0] * _COORD_SCALE), int(coords[i][1] * _COORD_SCALE)
                f.write(f"{i+1} {x} {y}\n")
            f.write("EOF\n")

        _write_par(par_f, tsp_f, tour_f, runs, seed, timeout)
        _run_lkh(lkh_bin, par_f, timeout)

        tour = _parse_tour(tour_f)
        if tour is None:
            return None

        route = _reroot_at_depot(tour, depot_lkh=1)
        return _fmt_single(route)


# ─────────────────────────────────────────────────────────────────────────────
# PyVRP 公共辅助：添加所有边（Euclidean 距离）
# ─────────────────────────────────────────────────────────────────────────────

def _add_pyvrp_edges(m, locs, coords_scaled, with_duration=False):
    """
    为 PyVRP Model 添加所有 (i, j) 边，距离 = 缩放后坐标的欧氏距离整数值。
    locs: add_depot/add_client 返回的对象列表（Depot + Client）。
    with_duration=True 时同时设置 duration（时间窗问题需要）。
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# CVRP — PyVRP（HGS，对大规模实例远快于 LKH-3）
# ─────────────────────────────────────────────────────────────────────────────

def _solve_cvrp(instance, timeout=120):
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    n        = instance["n"]
    coords   = np.array(instance["coords"])
    demands  = np.array(instance["demands"])
    capacity = instance["capacity"]

    S       = _COORD_SCALE
    cap_int = int(round(capacity * S))

    coords_scaled = [(int(coords[i][0] * S), int(coords[i][1] * S)) for i in range(n + 1)]

    m = Model()
    depot = m.add_depot(x=coords_scaled[0][0], y=coords_scaled[0][1])
    clients = [
        m.add_client(
            x=coords_scaled[i][0], y=coords_scaled[i][1],
            delivery=int(round(demands[i] * S)),
            service_duration=0,
        )
        for i in range(1, n + 1)
    ]
    locs = [depot] + clients
    _add_pyvrp_edges(m, locs, coords_scaled, with_duration=False)
    m.add_vehicle_type(num_available=n, capacity=cap_int)

    result = m.solve(stop=MaxRuntime(timeout), display=False)
    if not result.is_feasible():
        return None

    pyvrp_routes = list(result.best.routes())
    if not pyvrp_routes:
        return None

    routes = [[0] + [int(v) for v in r] + [0] for r in pyvrp_routes]
    return _fmt_multi(routes)


# ─────────────────────────────────────────────────────────────────────────────
# TSPTW — PyVRP（1辆车）
# ─────────────────────────────────────────────────────────────────────────────

def _solve_tsptw_pyvrp(instance, timeout=60):
    """
    将 TSPTW 建模为单车辆 VRPTW（num_available=1）交给 PyVRP 求解。
    """
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    n      = instance["n"]
    coords = np.array(instance["coords"])
    tw     = np.array(instance["time_windows"])

    S = _COORD_SCALE
    coords_scaled = [(int(coords[i][0] * S), int(coords[i][1] * S)) for i in range(n + 1)]

    m = Model()
    depot = m.add_depot(
        x=coords_scaled[0][0], y=coords_scaled[0][1],
        tw_early=int(tw[0][0] * S),
        tw_late=min(int(tw[0][1] * S), _MAX_TW_SCALED),
    )
    clients = [
        m.add_client(
            x=coords_scaled[i][0], y=coords_scaled[i][1],
            delivery=1,
            tw_early=int(tw[i][0] * S),
            tw_late=min(int(tw[i][1] * S), _MAX_TW_SCALED),
            service_duration=0,
        )
        for i in range(1, n + 1)
    ]
    locs = [depot] + clients
    _add_pyvrp_edges(m, locs, coords_scaled, with_duration=True)
    m.add_vehicle_type(num_available=1, capacity=n + 1)

    result = m.solve(stop=MaxRuntime(timeout), display=False)
    if not result.is_feasible():
        return None

    routes = list(result.best.routes())
    if not routes:
        return None

    return _fmt_single([0] + [int(v) for v in routes[0]] + [0])


# ─────────────────────────────────────────────────────────────────────────────
# VRPTW — PyVRP
# ─────────────────────────────────────────────────────────────────────────────

def _solve_vrptw_pyvrp(instance, timeout=120):
    """
    用 PyVRP 求解 VRPTW（无容量约束版）。
    """
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    n      = instance["n"]
    coords = np.array(instance["coords"])
    tw     = np.array(instance["time_windows"])

    S = _COORD_SCALE
    coords_scaled = [(int(coords[i][0] * S), int(coords[i][1] * S)) for i in range(n + 1)]

    m = Model()
    depot = m.add_depot(
        x=coords_scaled[0][0], y=coords_scaled[0][1],
        tw_early=int(tw[0][0] * S),
        tw_late=min(int(tw[0][1] * S), _MAX_TW_SCALED),
    )
    clients = [
        m.add_client(
            x=coords_scaled[i][0], y=coords_scaled[i][1],
            delivery=1,
            tw_early=int(tw[i][0] * S),
            tw_late=min(int(tw[i][1] * S), _MAX_TW_SCALED),
            service_duration=0,
        )
        for i in range(1, n + 1)
    ]
    locs = [depot] + clients
    _add_pyvrp_edges(m, locs, coords_scaled, with_duration=True)
    m.add_vehicle_type(num_available=n, capacity=n + 1)

    result = m.solve(stop=MaxRuntime(timeout), display=False)
    if not result.is_feasible():
        return None

    pyvrp_routes = list(result.best.routes())
    if not pyvrp_routes:
        return None

    routes = [[0] + [int(v) for v in r] + [0] for r in pyvrp_routes]
    return _fmt_multi(routes)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _write_par(par_f, prob_f, tour_f, runs, seed, timeout=60):
    with open(par_f, "w") as f:
        f.write(f"PROBLEM_FILE = {prob_f}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_f}\n")
        f.write(f"RUNS = {runs}\n")
        f.write(f"SEED = {seed}\n")
        f.write("TRACE_LEVEL = 0\n")
        f.write(f"TIME_LIMIT = {timeout}\n")


def _run_lkh(lkh_bin, par_f, timeout):
    # subprocess timeout 比 LKH TIME_LIMIT 宽裕 30s，防止 LKH 写 tour 文件时被 kill
    result = subprocess.run(
        [lkh_bin, par_f],
        capture_output=True, text=True, timeout=timeout + 30
    )
    if result.returncode != 0:
        raise RuntimeError(f"LKH 返回非零状态码: {result.returncode}\n{result.stderr[:500]}")


def _parse_tour(tour_f) -> Optional[list]:
    """解析 LKH 输出的 tour 文件，返回 1-indexed 节点列表（不含首尾-1）。"""
    if not os.path.exists(tour_f):
        return None
    nodes = []
    in_section = False
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
    return nodes if nodes else None


def _reroot_at_depot(tour: list, depot_lkh: int = 1) -> list:
    """将 tour 旋转使其从 depot（LKH 节点 1）开始，转换为 0-indexed 返回（含首尾 0）。"""
    idx = tour.index(depot_lkh)
    rotated = tour[idx:] + tour[:idx]
    route = [v - 1 for v in rotated]   # 1-indexed → 0-indexed
    return [0] + route[1:] + [0]


def _split_multi_routes(tour: list, depot_lkh: int = 1) -> list[list]:
    """
    将含多次 depot 访问的 LKH tour 拆分为多条路线（0-indexed）。
    LKH-3 multi-vehicle 输出中 depot 出现多次，用于分隔路线。
    """
    routes = []
    current = [0]   # 0-indexed depot
    for v in tour:
        if v == depot_lkh:
            if len(current) > 1:
                current.append(0)
                routes.append(current)
                current = [0]
        else:
            current.append(v - 1)   # 1-indexed → 0-indexed
    if len(current) > 1:
        current.append(0)
        routes.append(current)
    return routes if routes else [[0, 0]]


def _fmt_single(route: list) -> str:
    """TSP / TSPTW 格式：Route only"""
    route_str = " -> ".join(str(v) for v in route)
    return f"Route: {route_str}"


def _fmt_multi(routes: list[list]) -> str:
    """CVRP / VRPTW 格式：Route 1: ... \nRoute 2: ..."""
    lines = []
    for k, route in enumerate(routes, 1):
        lines.append(f"Route {k}: {' -> '.join(str(v) for v in route)}")
    return "\n".join(lines)
