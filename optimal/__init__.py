"""
optimal/ — LKH / HGS 近最优基线，用于 UniCOP-Reason 的 optimality gap。
位于 UniCOP 仓库根（与 UniCOP-Reason 平级），复用 UniCOP-Reason/problems 生成逻辑。

  generate_testset.py  阶段一：用 generate_instance 冻结测试实例到 instances/（CLI）
  build_optimal.py     阶段二：读冻结实例求近最优 cost 到 cache/（CLI）
  solvers.py           单实例求解（TSP→LKH，CVRP/TSPTW/VRPTW→PyVRP/HGS）
  loader.py            读冻结实例/缓存、计算 gap
"""

from optimal.solvers import solve_instance, SUPPORTED
from optimal.loader import load_costs, load_instances, optimality_gap, cache_path

__all__ = ["solve_instance", "SUPPORTED", "load_costs", "load_instances",
           "optimality_gap", "cache_path"]
