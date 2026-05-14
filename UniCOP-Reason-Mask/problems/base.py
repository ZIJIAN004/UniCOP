"""
所有问题类的抽象基类。
每个问题只需继承此类并实现 generate_instance / build_prompt /
is_feasible / get_tour_distance 四个方法。

reward 计算由 terminal_reward.py + pomo_prm.py 双信号架构处理，
本基类不再负责聚合 reward 公式。
"""

import json
import numpy as np
from abc import ABC, abstractmethod


class ProblemBase(ABC):
    name: str = "base"
    multi_route: bool = False   # True: CVRP/VRPTW；False: TSP/TSPTW/TSPDL

    # ── 必须实现的接口 ────────────────────────────────────────────────

    @abstractmethod
    def generate_instance(self, n: int, rng: np.random.Generator) -> dict:
        """
        生成一个问题实例（保证至少存在一条可行解）。
        返回 dict，所有数据以 Python 原生类型或 np.ndarray 存储。
        """

    @abstractmethod
    def build_prompt(self, instance: dict) -> list[dict]:
        """
        将实例格式化为 chat 格式的 prompt。
        返回 [{"role": "system/user", "content": "..."}]
        """

    @abstractmethod
    def get_tour_distance(self, completion: str, instance: dict) -> float | None:
        """
        返回路径的实际总距离（欧式距离之和）。
        路径可解析时返回 float，无法解析时返回 None。
        不检查可行性，仅计算距离。
        """

    @abstractmethod
    def is_feasible(self, completion: str, instance: dict) -> bool:
        """
        严格判断输出是否完全可行：
          - 路径可解析
          - 所有客户节点各访问恰好一次，首尾为 depot
          - 所有核心约束（容量/时间窗/draft limit）均满足
        """

    # ── 序列化（统一处理 numpy → list） ───────────────────────────────

    def to_json(self, instance: dict) -> str:
        serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in instance.items()
        }
        def _default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        return json.dumps(serializable, default=_default)

    def from_json(self, s: str) -> dict:
        return json.loads(s)

    # ── 通用工具 ───────────────────────────────────────────────────────

    @staticmethod
    def euclidean(a, b) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def total_distance(route: list[int], coords) -> float:
        coords = np.array(coords)
        return sum(
            float(np.linalg.norm(coords[route[i + 1]] - coords[route[i]]))
            for i in range(len(route) - 1)
        )
