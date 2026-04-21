"""
快速验证脚本：不需要 GPU，逐一测试所有问题类型的核心模块。

运行：python sanity_check.py
"""

import numpy as np
from problems import get_problem, SUPPORTED_PROBLEMS
from utils.parse import parse_single_route, parse_multi_route, all_visited_once
from terminal_reward import compute_terminal_reward


def test_problem(ptype: str):
    print(f"\n{'='*50}")
    print(f"  测试问题类型: {ptype.upper()}")
    print(f"{'='*50}")

    prob = get_problem(ptype)
    rng  = np.random.default_rng(42)
    n    = 8

    # ── 1. 生成实例 ────────────────────────────────────────────────
    instance = prob.generate_instance(n, rng)
    print(f"[1] 实例生成  keys={list(instance.keys())}")

    # ── 2. 序列化 ──────────────────────────────────────────────────
    data_str = prob.to_json(instance)
    restored = prob.from_json(data_str)
    print(f"[2] 序列化    JSON 长度={len(data_str)}")

    # ── 3. Prompt ──────────────────────────────────────────────────
    prompt = prob.build_prompt(instance)
    assert len(prompt) == 2
    print(f"[3] Prompt    roles={[m['role'] for m in prompt]}")
    print(f"    内容预览: {prompt[1]['content'][:120].strip()}...")

    # ── 4. Terminal Reward：用内置可行解构造模拟输出 ───────────────
    # 多路线问题用 'Route N: ' 格式（与 parse_multi_route 正则一致），
    # 单路线问题用 '路径: '。 sanity 测试只验证可行解 > 格式错误，不测 PRM。
    if prob.multi_route:
        routes = restored.get("feasible_routes", [])
        if routes:
            lines = [f"Route {i+1}: {' -> '.join(map(str, r))}" for i, r in enumerate(routes)]
            fake_output = "<think>分析...</think>\n" + "\n".join(lines)
        else:
            fake_output = ""
    else:
        tour = restored.get("feasible_tour", [])
        route_str = " -> ".join(map(str, tour)) if tour else ""
        fake_output = f"<think>分析...</think>\n路径: {route_str}"

    r_good = compute_terminal_reward(fake_output, restored, ptype)
    r_bad  = compute_terminal_reward("无效输出", restored, ptype)
    print(f"[4] Terminal  可行解={r_good:.3f}  格式错误={r_bad:.3f}")
    assert r_good > r_bad, "可行解 terminal reward 应高于格式错误！"
    print(f"    ✓ Terminal reward 函数正常")


def test_parsers():
    print(f"\n{'='*50}")
    print(f"  测试解析器")
    print(f"{'='*50}")

    # 单路线（parse 仅在 </think> 之后的答案区查找）
    text1 = "<think>thinking...</think>\n路径: 0 -> 3 -> 1 -> 7 -> 2 -> 0"
    r = parse_single_route(text1, n=7)
    assert r == [0, 3, 1, 7, 2, 0], f"解析失败: {r}"
    print(f"[单路线] ✓  {r}")

    # 单路线缺 </think>：应返 None
    assert parse_single_route("路径: 0 -> 1 -> 0", n=7) is None
    print(f"[单路线-无</think>] ✓ 正确返 None")

    # 多路线
    text2 = "<think>...</think>\n路线1: 0 -> 3 -> 1 -> 0\n路线2: 0 -> 7 -> 5 -> 0"
    rs = parse_multi_route(text2, n=7)
    assert rs is not None and len(rs) == 2
    print(f"[多路线] ✓  {rs}")

    # 多路线缺 </think>：应返 None
    assert parse_multi_route("路线1: 0 -> 1 -> 0", n=7) is None
    print(f"[多路线-无</think>] ✓ 正确返 None")

    # 覆盖检查
    ok = all_visited_once([[0,1,2,0],[0,3,4,0],[0,5,0]], n=5)
    assert ok
    fail = all_visited_once([[0,1,2,0],[0,1,3,0]], n=3)
    assert not fail
    print(f"[覆盖检查] ✓")


if __name__ == "__main__":
    test_parsers()
    for ptype in SUPPORTED_PROBLEMS:
        test_problem(ptype)

    print(f"\n{'='*50}")
    print("  所有检查通过！可以开始训练。")
    print(f"  运行: python train.py --problem tsptw --problem_size 10")
    print(f"{'='*50}")
