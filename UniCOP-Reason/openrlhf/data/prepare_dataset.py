"""
父目录 problems/ 问题实例 → OpenRLHF GRPO jsonl

输出格式 (每行一条):
    {"messages": [{"role": "system", ...}, {"role": "user", ...}],
     "instance_id": "tsp10_train_000001"}

- messages: 父目录 problem.build_prompt(instance) 返回值,保持 chat 格式
- 在最后一条 user 消息 content 末尾嵌入 [[instance_id:xxx]] 标记,
  训练时 OpenRLHF 应用 chat_template 后标记会被包进完整 prompt 文本,
  reward server 可以从中 regex 解出 instance_id 回查实例

同时输出一份 {stem}_instances.json 给 reward server 读.

用法:
    python data/prepare_dataset.py --problem_type tsp --problem_size 10 --num_train 20000
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# 让本脚本 import 父目录的 problems/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # /.../UniCOP/UniCOP-Reason
sys.path.insert(0, str(PROJECT_ROOT))

from problems import get_problem   # noqa: E402


def _inject_instance_marker(messages: list[dict], instance_id: str) -> list[dict]:
    """
    在最后一条 user 消息的 content 末尾插入 [[instance_id:xxx]].
    为 reward server 后续 regex 提取做准备.
    不改父目录 build_prompt 的原始输出 (返回一份新的 list of dict).
    """
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = dict(out[i])
            out[i]["content"] = out[i]["content"] + f"\n\n[[instance_id:{instance_id}]]"
            return out
    raise ValueError("messages 里找不到 role='user' 的项,无法插入 instance_id")


def _to_jsonable(obj):
    """把 np.ndarray / np.float32 等转成原生类型以便 JSON 序列化."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", type=str, required=True,
                        choices=["tsp", "cvrp", "vrptw", "tsptw", "tspdl"])
    parser.add_argument("--problem_size", type=int, required=True)
    parser.add_argument("--num_train", type=int, default=20000)
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default=str(SCRIPT_DIR / "processed"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    problem = get_problem(args.problem_type)

    for split, n_samples, seed_offset in [
        ("train", args.num_train, 0),
        ("test",  args.num_test,  10 ** 6),
    ]:
        out_path = os.path.join(
            args.output_dir,
            f"{args.problem_type}{args.problem_size}_{split}.jsonl",
        )
        print(f"生成 {split}: {n_samples} 条 → {out_path}")

        instances_dict = {}
        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(n_samples):
                # 父目录签名: generate_instance(n, rng: np.random.Generator)
                rng = np.random.default_rng(args.seed + seed_offset + i)
                inst = problem.generate_instance(n=args.problem_size, rng=rng)

                instance_id = f"{args.problem_type}{args.problem_size}_{split}_{i:06d}"

                # 父目录签名: build_prompt(instance) -> list[dict]
                messages = problem.build_prompt(inst)
                messages = _inject_instance_marker(messages, instance_id)

                f.write(json.dumps({
                    "messages": messages,
                    "instance_id": instance_id,
                }, ensure_ascii=False) + "\n")

                instances_dict[instance_id] = _to_jsonable(inst)

        inst_path = os.path.join(
            args.output_dir,
            f"{args.problem_type}{args.problem_size}_{split}_instances.json",
        )
        with open(inst_path, "w", encoding="utf-8") as f:
            json.dump(instances_dict, f, ensure_ascii=False)
        print(f"  instances dump → {inst_path}")

    print("完成。")


if __name__ == "__main__":
    main()
