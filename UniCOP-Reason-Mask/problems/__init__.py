"""
问题注册表：通过名称获取问题实例。

用法：
    from problems import get_problem
    prob = get_problem("tsptw")
    instance = prob.generate_instance(n=10, rng=np.random.default_rng(42))
"""

from .tsp     import TSP
from .cvrp    import CVRP
from .tsptw   import TSPTW
from .tspdl   import TSPDL
from .vrptw   import VRPTW

_REGISTRY = {
    "tsp":    TSP,
    "cvrp":   CVRP,
    "tsptw":  TSPTW,
    "tspdl":  TSPDL,
    "vrptw":  VRPTW,
}

SUPPORTED_PROBLEMS = ["tsp", "cvrp", "tsptw", "tspdl", "vrptw"]


def get_problem(name: str, **kwargs):
    """
    Args:
        name: 问题名称，如 "tsptw"
        **kwargs: 传给问题类构造函数的参数（如 time_slack=3.0）
    Returns:
        ProblemBase 实例
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"未知问题类型: {name}。支持: {SUPPORTED_PROBLEMS}")
    return _REGISTRY[name](**kwargs)
