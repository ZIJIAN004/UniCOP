"""
Reference heuristic solvers for FOARL optimality gap computation.

Nearest-neighbor variants with optional 2-opt (TSP only).
Quality is sufficient for reward normalization — not meant to be optimal.
"""

import numpy as np


def solve_reference(instance: dict, problem_type: str) -> float:
    coords = np.array(instance["coords"])

    if problem_type == "tsp":
        return _solve_tsp(coords)
    if problem_type == "cvrp":
        return _solve_cvrp(coords, np.array(instance["demands"]),
                           instance["capacity"])
    if problem_type == "tsptw":
        return _solve_tsptw(coords, np.array(instance["time_windows"]))
    if problem_type == "vrptw":
        return _solve_vrptw(coords, np.array(instance["time_windows"]))
    if problem_type == "tspdl":
        return _solve_tspdl(coords, np.array(instance["demands"]),
                            np.array(instance["draft_limits"]),
                            instance["capacity"])
    raise ValueError(f"Unsupported problem type for ref_solver: {problem_type}")


def _dist(coords, i, j):
    return float(np.linalg.norm(coords[i] - coords[j]))


def _tour_dist(tour, coords):
    return sum(_dist(coords, tour[i], tour[i + 1]) for i in range(len(tour) - 1))


# ── TSP: nearest-neighbor + 2-opt ────────────────────────────────────────────

def _solve_tsp(coords):
    n = len(coords) - 1
    visited = [False] * (n + 1)
    visited[0] = True
    tour = [0]
    cur = 0

    for _ in range(n):
        best_d, best_j = float('inf'), -1
        for j in range(1, n + 1):
            if not visited[j]:
                d = _dist(coords, cur, j)
                if d < best_d:
                    best_d, best_j = d, j
        tour.append(best_j)
        visited[best_j] = True
        cur = best_j
    tour.append(0)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                d_old = (_dist(coords, tour[i - 1], tour[i])
                         + _dist(coords, tour[j], tour[j + 1]))
                d_new = (_dist(coords, tour[i - 1], tour[j])
                         + _dist(coords, tour[i], tour[j + 1]))
                if d_new < d_old - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True

    return _tour_dist(tour, coords)


# ── CVRP: nearest-neighbor with capacity ─────────────────────────────────────

def _solve_cvrp(coords, demands, capacity):
    n = len(coords) - 1
    unvisited = set(range(1, n + 1))
    total = 0.0

    while unvisited:
        cur = 0
        remaining = capacity
        added_any = False

        while True:
            best_d, best_j = float('inf'), -1
            for j in unvisited:
                if demands[j] <= remaining + 1e-6:
                    d = _dist(coords, cur, j)
                    if d < best_d:
                        best_d, best_j = d, j
            if best_j == -1:
                break
            total += best_d
            remaining -= demands[best_j]
            unvisited.remove(best_j)
            cur = best_j
            added_any = True

        total += _dist(coords, cur, 0)

        if not added_any and unvisited:
            best_d, best_j = float('inf'), -1
            for j in unvisited:
                d = _dist(coords, 0, j)
                if d < best_d:
                    best_d, best_j = d, j
            total += best_d + _dist(coords, best_j, 0)
            unvisited.remove(best_j)

    return total


# ── TSPTW: nearest feasible neighbor ─────────────────────────────────────────

def _solve_tsptw(coords, time_windows):
    n = len(coords) - 1
    visited = [False] * (n + 1)
    visited[0] = True
    tour = [0]
    cur = 0
    cur_time = 0.0

    for _ in range(n):
        best_d, best_j = float('inf'), -1
        for j in range(1, n + 1):
            if visited[j]:
                continue
            d = _dist(coords, cur, j)
            if cur_time + d <= time_windows[j][1] + 1e-6 and d < best_d:
                best_d, best_j = d, j

        if best_j == -1:
            for j in range(1, n + 1):
                if not visited[j]:
                    d = _dist(coords, cur, j)
                    if d < best_d:
                        best_d, best_j = d, j

        tour.append(best_j)
        visited[best_j] = True
        arrive = cur_time + best_d
        cur_time = max(arrive, time_windows[best_j][0])
        cur = best_j

    tour.append(0)
    return _tour_dist(tour, coords)


# ── VRPTW: nearest feasible neighbor, multi-route ────────────────────────────

def _solve_vrptw(coords, time_windows):
    n = len(coords) - 1
    unvisited = set(range(1, n + 1))
    total = 0.0

    while unvisited:
        cur = 0
        cur_time = 0.0
        added_any = False

        while True:
            best_d, best_j = float('inf'), -1
            for j in unvisited:
                d = _dist(coords, cur, j)
                if cur_time + d <= time_windows[j][1] + 1e-6 and d < best_d:
                    best_d, best_j = d, j
            if best_j == -1:
                break
            total += best_d
            arrive = cur_time + best_d
            cur_time = max(arrive, time_windows[best_j][0])
            unvisited.remove(best_j)
            cur = best_j
            added_any = True

        total += _dist(coords, cur, 0)

        if not added_any and unvisited:
            best_d, best_j = float('inf'), -1
            for j in unvisited:
                d = _dist(coords, 0, j)
                if d < best_d:
                    best_d, best_j = d, j
            total += best_d + _dist(coords, best_j, 0)
            unvisited.remove(best_j)

    return total


# ── TSPDL: nearest-neighbor with draft limit ─────────────────────────────────

def _solve_tspdl(coords, demands, draft_limits, capacity):
    n = len(coords) - 1
    visited = [False] * (n + 1)
    visited[0] = True
    tour = [0]
    cur = 0
    load = 0.0

    for _ in range(n):
        best_d, best_j = float('inf'), -1
        for j in range(1, n + 1):
            if visited[j]:
                continue
            new_load = load + demands[j]
            if (new_load <= capacity + 1e-6
                    and new_load <= draft_limits[j] + 1e-6):
                d = _dist(coords, cur, j)
                if d < best_d:
                    best_d, best_j = d, j

        if best_j == -1:
            for j in range(1, n + 1):
                if not visited[j]:
                    d = _dist(coords, cur, j)
                    if d < best_d:
                        best_d, best_j = d, j

        tour.append(best_j)
        visited[best_j] = True
        load += demands[best_j]
        cur = best_j

    tour.append(0)
    return _tour_dist(tour, coords)
