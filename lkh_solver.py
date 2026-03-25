"""
lkh_solver.py
为各 COP 问题类型调用 LKH / LKH-3 二进制程序，返回 UniCOP-Reason 格式的近最优解。

二进制路径配置（修改下方常量，或通过环境变量覆盖）：
  LKH_BIN  = LKH-2 路径，处理 TSP / TSPTW
  LKH3_BIN = LKH-3 路径，处理 CVRP / VRPTW / CVRPTW

TSPDL 不被 LKH 原生支持（Draft Limit 为自定义约束），使用内置贪心求解器。
"""

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np

# ── 二进制路径（在此处修改，或通过环境变量覆盖）────────────────────────────────
LKH_BIN  = os.environ.get("LKH_BIN",  "/path/to/LKH")     # LKH-2 binary
LKH3_BIN = os.environ.get("LKH3_BIN", "/path/to/LKH3")    # LKH-3 binary

# 坐标缩放系数：LKH 使用整数距离，原始坐标在 [0,1]，放大以保留精度
_COORD_SCALE = 1_000_000


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
    _solvers = {
        "tsp":    _solve_tsp,
        "tsptw":  _solve_tsptw,
        "tspdl":  _solve_tspdl_greedy,
        "cvrp":   _solve_cvrp,
        "vrptw":  _solve_vrptw,
        "cvrptw": _solve_cvrptw,
    }
    fn = _solvers.get(problem_type)
    if fn is None:
        raise ValueError(f"未知问题类型: {problem_type}")

    bin_path = lkh3_bin if problem_type in ("cvrp", "vrptw", "cvrptw") else lkh_bin
    try:
        return fn(instance, bin_path, runs=runs, seed=seed, timeout=timeout)
    except Exception as e:
        print(f"    [LKH ERROR] {problem_type}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TSP
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

        _write_par(par_f, tsp_f, tour_f, runs, seed)
        _run_lkh(lkh_bin, par_f, timeout)

        tour = _parse_tour(tour_f)
        if tour is None:
            return None

        route = _reroot_at_depot(tour, depot_lkh=1)
        return _fmt_single(route)


# ─────────────────────────────────────────────────────────────────────────────
# TSPTW
# ─────────────────────────────────────────────────────────────────────────────

def _solve_tsptw(instance, lkh_bin, runs=1, seed=42, timeout=60):
    n      = instance["n"]
    coords = np.array(instance["coords"])
    tw     = np.array(instance["time_windows"])

    with tempfile.TemporaryDirectory() as tmp:
        tsp_f  = os.path.join(tmp, "p.tsp")
        par_f  = os.path.join(tmp, "p.par")
        tour_f = os.path.join(tmp, "p.tour")

        with open(tsp_f, "w") as f:
            f.write(f"NAME: tsptw\nTYPE: TSPTW\nDIMENSION: {n+1}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")
            for i in range(n + 1):
                x, y = int(coords[i][0] * _COORD_SCALE), int(coords[i][1] * _COORD_SCALE)
                f.write(f"{i+1} {x} {y}\n")
            f.write("TIME_WINDOW_SECTION\n")
            for i in range(n + 1):
                a = int(tw[i][0] * _COORD_SCALE)
                b = int(min(tw[i][1], 1e8) * _COORD_SCALE) if tw[i][1] < 1e8 else 10**12
                f.write(f"{i+1} {a} {b}\n")
            f.write("EOF\n")

        _write_par(par_f, tsp_f, tour_f, runs, seed)
        _run_lkh(lkh_bin, par_f, timeout)

        tour = _parse_tour(tour_f)
        if tour is None:
            return None

        route = _reroot_at_depot(tour, depot_lkh=1)
        return _fmt_single(route)


# ─────────────────────────────────────────────────────────────────────────────
# TSPDL —— LKH 不支持 draft limit，使用贪心最近邻求解
# ─────────────────────────────────────────────────────────────────────────────

def _solve_tspdl_greedy(instance, *args, **kwargs):
    """
    贪心最近邻（draft-limit 约束）：
    每步在满足 current_load <= draft_limit[v] 的未访问节点中选最近的。
    """
    n           = instance["n"]
    coords      = np.array(instance["coords"])
    demands     = np.array(instance["demands"])
    draft_limits = np.array(instance["draft_limits"])
    capacity    = float(instance["capacity"])

    current_load = capacity
    current_node = 0
    unvisited    = set(range(1, n + 1))
    route        = [0]

    while unvisited:
        feasible = [v for v in unvisited if current_load <= draft_limits[v] + 1e-9]
        pool     = feasible if feasible else list(unvisited)   # 无可行节点时放宽（兜底）
        nearest  = min(pool, key=lambda v: float(np.linalg.norm(coords[v] - coords[current_node])))
        route.append(nearest)
        current_load -= demands[nearest]
        current_node  = nearest
        unvisited.remove(nearest)

    route.append(0)
    nodes_str = ", ".join(str(v) for v in route[1:-1])
    route_str = " -> ".join(str(v) for v in route)
    return f"Node selection: {nodes_str}\nRoute: {route_str}"


# ─────────────────────────────────────────────────────────────────────────────
# CVRP
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

        _write_par(par_f, vrp_f, tour_f, runs, seed)
        _run_lkh(lkh3_bin, par_f, timeout)

        tour = _parse_tour(tour_f)
        if tour is None:
            return None

        routes = _split_multi_routes(tour, depot_lkh=1)
        return _fmt_multi(routes)


# ─────────────────────────────────────────────────────────────────────────────
# VRPTW
# ─────────────────────────────────────────────────────────────────────────────

def _solve_vrptw(instance, lkh3_bin, runs=1, seed=42, timeout=120):
    n      = instance["n"]
    coords = np.array(instance["coords"])
    tw     = np.array(instance["time_windows"])

    with tempfile.TemporaryDirectory() as tmp:
        vrp_f  = os.path.join(tmp, "p.vrp")
        par_f  = os.path.join(tmp, "p.par")
        tour_f = os.path.join(tmp, "p.tour")

        with open(vrp_f, "w") as f:
            f.write(f"NAME: vrptw\nTYPE: VRPTW\nDIMENSION: {n+1}\n")
            # VRPTW 无容量约束，设一个足够大的容量
            f.write(f"CAPACITY: {n * _COORD_SCALE}\nEDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i in range(n + 1):
                x, y = int(coords[i][0] * _COORD_SCALE), int(coords[i][1] * _COORD_SCALE)
                f.write(f"{i+1} {x} {y}\n")
            f.write("DEMAND_SECTION\n")
            for i in range(n + 1):
                f.write(f"{i+1} {'0' if i == 0 else '1'}\n")
            f.write("TIME_WINDOW_SECTION\n")
            for i in range(n + 1):
                a = int(tw[i][0] * _COORD_SCALE)
                b = int(min(tw[i][1], 1e8) * _COORD_SCALE) if tw[i][1] < 1e8 else 10**12
                f.write(f"{i+1} {a} {b}\n")
            f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

        _write_par(par_f, vrp_f, tour_f, runs, seed)
        _run_lkh(lkh3_bin, par_f, timeout)

        tour = _parse_tour(tour_f)
        if tour is None:
            return None

        routes = _split_multi_routes(tour, depot_lkh=1)
        return _fmt_multi(routes)


# ─────────────────────────────────────────────────────────────────────────────
# CVRPTW
# ─────────────────────────────────────────────────────────────────────────────

def _solve_cvrptw(instance, lkh3_bin, runs=1, seed=42, timeout=120):
    n        = instance["n"]
    coords   = np.array(instance["coords"])
    demands  = np.array(instance["demands"])
    capacity = float(instance["capacity"])
    tw       = np.array(instance["time_windows"])

    cap_int = int(round(capacity * _COORD_SCALE))

    with tempfile.TemporaryDirectory() as tmp:
        vrp_f  = os.path.join(tmp, "p.vrp")
        par_f  = os.path.join(tmp, "p.par")
        tour_f = os.path.join(tmp, "p.tour")

        with open(vrp_f, "w") as f:
            f.write(f"NAME: cvrptw\nTYPE: CVRPTW\nDIMENSION: {n+1}\n")
            f.write(f"CAPACITY: {cap_int}\nEDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i in range(n + 1):
                x, y = int(coords[i][0] * _COORD_SCALE), int(coords[i][1] * _COORD_SCALE)
                f.write(f"{i+1} {x} {y}\n")
            f.write("DEMAND_SECTION\n")
            f.write("1 0\n")
            for i in range(1, n + 1):
                f.write(f"{i+1} {int(round(demands[i] * _COORD_SCALE))}\n")
            f.write("TIME_WINDOW_SECTION\n")
            for i in range(n + 1):
                a = int(tw[i][0] * _COORD_SCALE)
                b = int(min(tw[i][1], 1e8) * _COORD_SCALE) if tw[i][1] < 1e8 else 10**12
                f.write(f"{i+1} {a} {b}\n")
            f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

        _write_par(par_f, vrp_f, tour_f, runs, seed)
        _run_lkh(lkh3_bin, par_f, timeout)

        tour = _parse_tour(tour_f)
        if tour is None:
            return None

        routes = _split_multi_routes(tour, depot_lkh=1)
        return _fmt_multi(routes)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _write_par(par_f, prob_f, tour_f, runs, seed):
    with open(par_f, "w") as f:
        f.write(f"PROBLEM_FILE = {prob_f}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_f}\n")
        f.write(f"RUNS = {runs}\n")
        f.write(f"SEED = {seed}\n")
        f.write("TRACE_LEVEL = 0\n")
        f.write("TIME_LIMIT = 60\n")


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
    """CVRP / VRPTW / CVRPTW 格式：Route 1: ... \nRoute 2: ..."""
    lines = []
    for k, route in enumerate(routes, 1):
        lines.append(f"Route {k}: {' -> '.join(str(v) for v in route)}")
    return "\n".join(lines)
