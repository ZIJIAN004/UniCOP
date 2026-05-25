"""
optimal/ — LKH / HGS 近最优基线，用于 UniCOP-Reason 的 optimality gap。

  solvers.py       逐实例求 (近) 最优解（TSP→LKH，CVRP/TSPTW/VRPTW→PyVRP/HGS）
  build_optimal.py 复现 evaluate.py 测试集并缓存近最优 cost（CLI）
  loader.py        读取缓存、计算 gap
"""

from optimal.solvers import solve_instance, SUPPORTED
from optimal.loader import load_costs, optimality_gap, cache_path

__all__ = ["solve_instance", "SUPPORTED", "load_costs", "optimality_gap", "cache_path"]
