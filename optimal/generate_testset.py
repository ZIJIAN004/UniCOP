"""
optimal/generate_testset.py
阶段一：用现有 generate_instance 生成并【冻结】测试实例到磁盘。
求解（LKH/HGS）是独立的阶段二，见 build_optimal.py，它读取这里冻结的实例。

为什么要冻结：把测试集落盘成文件后，求解、评测都读同一批实例，彻底消除“重新生成
是否一致”的隐患；生成与求解解耦，可分别重跑。

复现对齐：每个 (problem_type, n) 用 `np.random.default_rng(seed)` 独立播种，按顺序
调用 `generate_instance`，与 evaluate.py:evaluate_single 完全一致（其每次也重新
default_rng(9999) 后顺序生成）。因此 seed=9999 冻结的实例与 evaluate.py 现场生成的
逐一相同。

用法（在 UniCOP 仓库根目录下）：
  python -m optimal.generate_testset                          # 默认 4类×{20,50,100}×1000
  python -m optimal.generate_testset --sizes 20 50 100 --num_instances 1000 --seed 9999
  python -m optimal.generate_testset --problem_types tsp cvrp --sizes 50

输出：optimal/instances/{type}_n{n}_seed{seed}_N{num}.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# optimal/ 在 UniCOP 顶层；problems/ 在 UniCOP-Reason 下
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../UniCOP/optimal
_UNICOP = os.path.dirname(_HERE)                            # .../UniCOP
_REASON = os.path.join(_UNICOP, "UniCOP-Reason")           # problems/ utils/ 所在
for _p in (_UNICOP, _REASON):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from problems import get_problem, SUPPORTED_PROBLEMS  # noqa: E402

# 本模块支持的（有 LKH/HGS 求解器的）问题类型；TSPDL 暂不在内
SUPPORTED = ("tsp", "cvrp", "tsptw", "vrptw")
_DEFAULT_SEED = 9999  # 必须与 evaluate.py:evaluate_single 一致

INSTANCES_DIR = os.path.join(_HERE, "instances")


def instances_path(problem_type: str, n: int, num: int, seed: int,
                   inst_dir: str = INSTANCES_DIR) -> str:
    return os.path.join(inst_dir, f"{problem_type}_n{n}_seed{seed}_N{num}.json")


def generate_frozen(problem_type: str, n: int, num: int, seed: int,
                    inst_dir: str) -> str:
    """生成 num 个实例并落盘，返回文件路径。"""
    prob = get_problem(problem_type)
    rng = np.random.default_rng(seed=seed)

    # 用 prob.to_json 统一处理 numpy→list（与 data/generate.py 的 problem_data 同口径）
    instances = [json.loads(prob.to_json(prob.generate_instance(n, rng)))
                 for _ in range(num)]

    payload = {
        "problem_type": problem_type,
        "n": n,
        "seed": seed,
        "num_instances": num,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "instances": instances,
    }

    os.makedirs(inst_dir, exist_ok=True)
    fpath = instances_path(problem_type, n, num, seed, inst_dir)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return fpath


def main():
    ap = argparse.ArgumentParser(description="阶段一：冻结测试实例（不求解）")
    ap.add_argument("--problem_types", nargs="+", default=list(SUPPORTED),
                    help=f"默认 {list(SUPPORTED)}（TSPDL 暂不支持）")
    ap.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100],
                    help="客户节点数列表，默认 20 50 100")
    ap.add_argument("--num_instances", type=int, default=1000,
                    help="每个 (type, n) 的实例数，默认 1000")
    ap.add_argument("--seed", type=int, default=_DEFAULT_SEED,
                    help=f"随机种子，必须与 evaluate.py 一致，默认 {_DEFAULT_SEED}")
    ap.add_argument("--out_dir", default=INSTANCES_DIR)
    args = ap.parse_args()

    for pt in args.problem_types:
        if pt.lower() not in SUPPORTED:
            ap.error(f"不支持的问题类型: {pt}（支持 {list(SUPPORTED)}；"
                     f"problems 全集 {SUPPORTED_PROBLEMS}）")

    print("=" * 60)
    print(f"冻结测试集  types={args.problem_types}  sizes={args.sizes}")
    print(f"num_instances={args.num_instances}  seed={args.seed}")
    print(f"out_dir={args.out_dir}")
    print("=" * 60)

    total = 0
    for n in args.sizes:
        for pt in args.problem_types:
            fpath = generate_frozen(pt, n, args.num_instances, args.seed, args.out_dir)
            total += args.num_instances
            print(f"  ✓ {pt:6s} n={n:<4d} × {args.num_instances}  →  "
                  f"{os.path.relpath(fpath, _UNICOP)}")

    print(f"\n完成：{len(args.problem_types)}×{len(args.sizes)} 个数据集，"
          f"共 {total} 个实例。求解请跑: python -m optimal.build_optimal")


if __name__ == "__main__":
    main()
