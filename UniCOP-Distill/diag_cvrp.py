"""诊断 CVRP PyVRP 求解失败原因。在服务器上跑: python diag_cvrp.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "UniCOP-Reason"))

import numpy as np
import pyvrp
print(f"PyVRP version: {pyvrp.__version__}")

from pyvrp import Model
from pyvrp.stop import MaxRuntime
from problems import get_problem

_S = 1_000_000

problem = get_problem("cvrp")
rng = np.random.default_rng(seed=42)
inst = problem.generate_instance(20, rng)

n = inst["n"]
coords = np.array(inst["coords"])
demands = np.array(inst["demands"])
capacity = inst["capacity"]

print(f"\nn={n}, capacity={capacity}")
print(f"demands (raw): {demands[1:]}")
print(f"demands range: [{demands[1:].min():.4f}, {demands[1:].max():.4f}]")
print(f"demands sum:   {demands[1:].sum():.4f}")
print(f"min routes needed: {int(np.ceil(demands[1:].sum() / capacity))}")

cap_int = int(round(capacity * _S))
demand_ints = [int(round(demands[i] * _S)) for i in range(1, n + 1)]
print(f"\nscaled capacity:  {cap_int}")
print(f"scaled demands:   {demand_ints}")
print(f"scaled demand sum: {sum(demand_ints)}")

coords_scaled = [(int(coords[i][0] * _S), int(coords[i][1] * _S)) for i in range(n + 1)]

m = Model()
depot = m.add_depot(x=coords_scaled[0][0], y=coords_scaled[0][1])
clients = []
for i in range(1, n + 1):
    c = m.add_client(
        x=coords_scaled[i][0], y=coords_scaled[i][1],
        delivery=int(round(demands[i] * _S)),
        service_duration=0,
    )
    clients.append(c)

locs = [depot] + clients
n_locs = len(locs)
edge_count = 0
for i in range(n_locs):
    xi, yi = coords_scaled[i]
    for j in range(n_locs):
        if i == j:
            continue
        xj, yj = coords_scaled[j]
        dist = int(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
        m.add_edge(locs[i], locs[j], distance=dist)
        edge_count += 1

print(f"\nedges added: {edge_count} (expected {n_locs * (n_locs - 1)})")
m.add_vehicle_type(num_available=n, capacity=cap_int)

print(f"\nSolving (timeout=30s)...")
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = m.solve(stop=MaxRuntime(30), display=True)
    if w:
        for warning in w:
            print(f"WARNING: {warning.message}")

print(f"\nfeasible: {result.is_feasible()}")
print(f"best cost: {result.best.distance() if result.best else 'N/A'}")
if result.is_feasible():
    for k, route in enumerate(result.best.routes()):
        print(f"  Route {k+1}: {[int(v) for v in route]}")
else:
    print("INFEASIBLE — printing stats for debug:")
    print(f"  num_iterations: {result.num_iterations}")
    print(f"  runtime: {result.runtime:.2f}s")
