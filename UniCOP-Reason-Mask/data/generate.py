"""数据集生成：统一调度，各问题独立实现实例生成逻辑。"""

import numpy as np
from datasets import Dataset, concatenate_datasets
from problems import get_problem
from config import config


def build_dataset(problem_type: str, num_samples: int, seed: int, n: int) -> Dataset:
    """
    生成单问题类型的 HuggingFace Dataset，包含字段：
        - prompt:        list[dict]，chat 格式
        - problem_data:  str，JSON 序列化的问题数据
        - problem_type:  str，问题类型名称（供 reward 函数 dispatch 使用）
    """
    prob = get_problem(problem_type)
    rng  = np.random.default_rng(seed)

    prompts, problem_datas = [], []
    for _ in range(num_samples):
        instance  = prob.generate_instance(n, rng)
        prompt    = prob.build_prompt(instance)
        data_str  = prob.to_json(instance)
        prompts.append(prompt)
        problem_datas.append(data_str)

    return Dataset.from_dict({
        "prompt":       prompts,
        "problem_data": problem_datas,
        "problem_type": [problem_type] * num_samples,
    })


def build_mixed_dataset(
    problem_types: list[str],
    num_samples_each: int,
    seed: int,
    n: int,
) -> Dataset:
    """
    生成混合问题类型的 Dataset，将各问题的样本合并后随机打乱。

    Args:
        problem_types:    问题类型列表，如 ["tsp", "cvrp", "tsptw", "tspdl"]
        num_samples_each: 每种问题的样本数
        seed:             随机种子（各问题类型使用不同偏移种子保证独立性）
        n:                问题规模
    Returns:
        打乱顺序的混合 Dataset
    """
    datasets = [
        build_dataset(pt, num_samples=num_samples_each, seed=seed + i, n=n)
        for i, pt in enumerate(problem_types)
    ]
    return concatenate_datasets(datasets).shuffle(seed=seed)


if __name__ == "__main__":
    for ptype in ["tsp", "cvrp", "tsptw", "tspdl", "vrptw"]:
        ds = build_dataset(ptype, num_samples=2, seed=42, n=10)
        print(f"[{ptype}] prompt preview:")
        print(ds[0]["prompt"][1]["content"][:300])
        print()
