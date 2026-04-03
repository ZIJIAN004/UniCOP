"""
lkh_solver.py
为各 COP 问题类型调用对应求解器，返回 UniCOP-Reason 格式的近最优解。

求解器分配：
  TSP   → LKH-2
  CVRP  → LKH-3
  TSPTW → PyVRP（1辆车 VRPTW，时间窗约束满足率高）
  VRPTW → PyVRP（原生支持，约束满足率高）

LKH 二进制路径配置（修改下方常量，或通过环境变量覆盖）：
  LKH_BIN  = LKH-2 路径，处理 TSP
  LKH3_BIN = LKH-3 路径，处理 CVRP
"""

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

# ── 二进制路径（在此处修改，或通过环境变量覆盖）────────────────────────────────
LKH_BIN  = os.environ.get("LKH_BIN",  "/path/to/LKH")     # LKH-2 binary
LKH3_BIN = os.environ.get("LKH3_BIN", "/path/to/LKH3")    # LKH-3 binary

# 坐标缩放系数：LKH/PyVRP 使用整数距离，原始坐标在 [0,1]，放大以保留精度
_COORD_SCALE = 1_000_000
# PyVRP 内部使用 32-bit 整数，时间窗上限需限制在安全范围内（2e9 < 2^31-1）
_MAX_TW_SCALED = 2_000_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def solve(problem_type: str, instance: dict,
          lkh_bin: str = LKH_BIN, lkh3_bin: str = LKH3_BIN,
          runs: int = 1, seed: int = 42, timeout: int = 60) -> Optional[str]:
    """
    统一入口：根据问题类型调用对应求解器。
    返回 UniCOP-Reason 格式的解字符串，失败返回 None。
    """
    try:
        if problem_type == "tsp":
            return _solve_tsp(instance, lkh_bin, runs=runs, seed=seed, timeout=timeout)
        elif problem_type == "cvrp":
            return _solve_cvrp(instance, lkh3_bin, runs=runs, seed=seed, timeout=timeout)
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
# TSPTW — PyVRP（1辆车）
# ─────────────────────────────────────────────────────────────────────────────

def _solve_tsptw_pyvrp(instance, timeout=60):
    """
    将 TSPTW 建模为单车辆 VRPTW（num_available=1）交给 PyVRP 求解。
    相比 LKH-2 的 TSPTW 模式，PyVRP 时间窗约束满足率更高。
    """
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    n      = instance["n"]
    coords = np.array(instance["coords"])
    tw     = np.array(instance["time_windows"])

    S = _COORD_SCALE

    m = Model()
    m.add_depot(
        x=int(coords[0][0] * S),
        y=int(coords[0][1] * S),
        tw_early=int(tw[0][0] * S),
        tw_late=min(int(tw[0][1] * S), _MAX_TW_SCALED),
    )
    for i in range(1, n + 1):
        m.add_client(
            x=int(coords[i][0] * S),
            y=int(coords[i][1] * S),
            tw_early=int(tw[i][0] * S),
            tw_late=min(int(tw[i][1] * S), _MAX_TW_SCALED),
            demand=1,
            service_duration=0,
        )
    # 单车辆，容量足够覆盖所有节点（不作为约束）
    m.add_vehicle_type(num_available=1, capacity=n + 1)

    result = m.solve(stop=MaxRuntime(timeout), display=False)
    if not result.is_feasible():
        return None

    routes = list(result.best.routes())
    if not routes:
        return None

    # TSPTW 只有 1 条路线；PyVRP client index 与添加顺序一致（1-indexed → 对应节点 1..n）
    nodes = [int(v) for v in routes[0]]   # client indices == our node indices (1..n)
    route = [0] + nodes + [0]
    return _fmt_single(route)


# ─────────────────────────────────────────────────────────────────────────────
# CVRP — LKH-3
# ─────────────────────────────────────────────────────────────────────────────

def _solve_cvrp(instance, lkh3_bin, runs=1, seed=42, timeout=120):
    n        = instance["n"]
    coords   = np.array(instance["coords"])
    demands  = np.array(instance["demands"])
    capacity = instance["capacity"]

    cap_int = int(round(capacity * _COORD_SCALE))

    with tempfile.TemporaryDirectory() as tmp:
        vrp_f  = os.path.join(tmp, "p.vrp")
        par_f  = os.path.join(tmp, "p.par")
        tour_f = os.path.join(tmp, "p.tour")

        with open(vrp_f, "w") as f:
            f.write(f"NAME: cvrp\nTYPE: CVRP\nDIMENSION: {n+1}\n")
            f.write(f"CAPACITY: {cap_int}\nEDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i in range(n + 1):
                x, y = int(coords[i][0] * _COORD_SCALE), int(coords[i][1] * _COORD_SCALE)
                f.write(f"{i+1} {x} {y}\n")
            f.write("DEMAND_SECTION\n")
            f.write("1 0\n")
            for i in range(1, n + 1):
                f.write(f"{i+1} {int(round(demands[i] * _COORD_SCALE))}\n")
            f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

        _write_par(par_f, vrp_f, tour_f, runs, seed, timeout)
        _run_lkh(lkh3_bin, par_f, timeout)

        tour = _parse_tour(tour_f)
        if tour is None:
            return None

        routes = _split_multi_routes(tour, depot_lkh=1)
        return _fmt_multi(routes)


# ─────────────────────────────────────────────────────────────────────────────
# VRPTW — PyVRP
# ─────────────────────────────────────────────────────────────────────────────

def _solve_vrptw_pyvrp(instance, timeout=120):
    """
    用 PyVRP 求解 VRPTW（无容量约束版）。
    PyVRP 是 HGS 的 Python 封装，对时间窗约束的处理优于 LKH-3。
    """
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    n      = instance["n"]
    coords = np.array(instance["coords"])
    tw     = np.array(instance["time_windows"])

    S = _COORD_SCALE

    m = Model()
    m.add_depot(
        x=int(coords[0][0] * S),
        y=int(coords[0][1] * S),
        tw_early=int(tw[0][0] * S),
        tw_late=min(int(tw[0][1] * S), _MAX_TW_SCALED),
    )
    for i in range(1, n + 1):
        m.add_client(
            x=int(coords[i][0] * S),
            y=int(coords[i][1] * S),
            tw_early=int(tw[i][0] * S),
            tw_late=min(int(tw[i][1] * S), _MAX_TW_SCALED),
            demand=1,
            service_duration=0,
        )
    # 无容量约束：最多 n 辆车，容量足够覆盖所有节点
    m.add_vehicle_type(num_available=n, capacity=n + 1)

    result = m.solve(stop=MaxRuntime(timeout), display=False)
    if not result.is_feasible():
        return None

    pyvrp_routes = list(result.best.routes())
    if not pyvrp_routes:
        return None

    # PyVRP client index 与添加顺序一致（1-indexed → 对应节点 1..n）
    routes = []
    for r in pyvrp_routes:
        nodes = [int(v) for v in r]
        routes.append([0] + nodes + [0])

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
    result = subprocess.run(
        [lkh_bin, par_f],
        capture_output=True, text=True, timeout=timeout
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
    """TSP / TSPTW 格式：Node selection + Route"""
    nodes_str = ", ".join(str(v) for v in route[1:-1])
    route_str = " -> ".join(str(v) for v in route)
    return f"Node selection: {nodes_str}\nRoute: {route_str}"


def _fmt_multi(routes: list[list]) -> str:
    """CVRP / VRPTW 格式：Route 1: ... \nRoute 2: ..."""
    lines = []
    for k, route in enumerate(routes, 1):
        lines.append(f"Route {k}: {' -> '.join(str(v) for v in route)}")
    return "\n".join(lines)
