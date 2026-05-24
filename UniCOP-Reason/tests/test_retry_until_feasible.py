"""
Smoke test: evaluate.py 的 retry_until_feasible 路径.

覆盖场景:
  1. 第一轮就 fully feasible → rounds_used=0, converged=True
  2. 第一轮 missing 1 节点, 第二轮补齐 → rounds_used=1, converged=True
  3. 第一轮 duplicate, 第二轮去重 → rounds_used=1, converged=True
  4. parse 失败, 第二轮 parse OK + feasible → rounds_used=1
  5. 连续 3 轮都不收敛 → rounds_used = max_rounds, converged=False
  6. cov OK 但 cap 违例 → 早停 stopped_reason="cov_ok_other_violation"
  7. _strip_think_for_history truncated 边界 (没 </think>)

跑法 (本地, 无需 GPU):
    cd UniCOP-Reason-Mask
    python tests/test_retry_until_feasible.py
"""

import os
import sys
import types

# Allow running as standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 屏蔽 torch / pomo_prm / terminal_reward 的真实 import (这测试不用)
# evaluate.py 顶部会 import 一堆, 我们只用其 retry helper, 全 stub 即可
for mod_name in ("torch", "pomo_prm"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# terminal_reward.compute_terminal_components 给个 no-op stub
if "terminal_reward" not in sys.modules:
    _fake_tr = types.ModuleType("terminal_reward")
    def _stub_components(*_a, **_kw):
        return {"coverage": 0.0, "constraint": 0.0}
    _fake_tr.compute_terminal_components = _stub_components
    sys.modules["terminal_reward"] = _fake_tr

# 现在可以安全 import evaluate 的 retry helper
from evaluate import (
    _diagnose_routes,
    _build_retry_feedback,
    _strip_think_for_history,
    _retry_loop_one,
)


# ── Stub Problem 类 ──────────────────────────────────────────────────────
# 模拟 problems/cvrp.py 但完全不依赖 numpy/torch, 让测试纯 Python.

class _StubCVRP:
    """模仿 problems/cvrp.py 接口: multi_route + parse + is_feasible."""
    multi_route = True

    def __init__(self, capacity_demands=None):
        # capacity_demands: dict customer_id → demand (用于 capacity check)
        # None = 全 0 demand, capacity 永远满足
        self.capacity_demands = capacity_demands or {}
        self.capacity = 1.0  # arbitrary

    def is_feasible(self, completion: str, instance: dict) -> bool:
        """严格可行: parse OK + 全访客 + 无重复 + capacity 满足."""
        from utils.parse import parse_multi_route
        n = instance["n"]
        routes = parse_multi_route(completion, n)
        if routes is None:
            return False
        all_customers = [v for r in routes for v in r if v != 0]
        if sorted(all_customers) != list(range(1, n + 1)):
            return False
        if not all(r[0] == 0 and r[-1] == 0 for r in routes):
            return False
        # capacity check
        for r in routes:
            load = sum(self.capacity_demands.get(v, 0.0) for v in r if v != 0)
            if load > self.capacity + 1e-6:
                return False
        return True


class _StubTSP:
    """模仿 problems/tsp.py 接口: multi_route=False, 单路线."""
    multi_route = False

    def is_feasible(self, completion: str, instance: dict) -> bool:
        from utils.parse import parse_single_route
        n = instance["n"]
        route = parse_single_route(completion, n)
        if route is None:
            return False
        if route[0] != 0 or route[-1] != 0:
            return False
        customers = route[1:-1]
        if sorted(customers) != list(range(1, n + 1)):
            return False
        # depot 0 不能出现在中间
        if any(v == 0 for v in customers):
            return False
        return True


def _make_instance(n=5):
    return {"n": n}


# ── Mock generate_fn ─────────────────────────────────────────────────────

class _MockGen:
    """按预设序列返回 completion. 每次调用消费一项."""

    def __init__(self, queue: list):
        # queue: list of completion text (or tuple)
        self.queue = list(queue)
        self.calls = []   # 记录调用参数, 便于 assert prompt 累积正确

    def __call__(self, prompts, num_samples, temperature, max_len, batch_size):
        assert len(prompts) == 1, f"retry 必须 batch=1, got {len(prompts)}"
        assert num_samples == 1
        assert batch_size == 1
        self.calls.append({
            "prompt_len_messages": len(prompts[0]),
            "last_user_msg": prompts[0][-1]["content"] if prompts[0][-1]["role"] == "user" else None,
        })
        if not self.queue:
            raise AssertionError("MockGen 队列空, 测试期望的 retry 次数错了")
        item = self.queue.pop(0)
        # 包成 generate_fn 返回格式: list[list[tuple]]
        if isinstance(item, tuple):
            return [[item]]
        return [[(item, False, len(item))]]


# ── Helpers for test routes ──────────────────────────────────────────────

def _route_text(routes_str: str, with_think: bool = True) -> str:
    """构造 completion text. routes_str 是 'Route 1: 0 -> 1 -> 2 -> 0' 之类."""
    if with_think:
        return f"<think>\nplanning...\n</think>\n\n{routes_str}"
    return routes_str


_ORIG_PROMPT = [
    {"role": "system", "content": "You are a CVRP solver."},
    {"role": "user", "content": "Solve CVRP n=5."},
]


# ── Test cases ───────────────────────────────────────────────────────────

def test_1_first_round_feasible():
    """场景 1: 第一轮就完全可行, 0 retry."""
    prob = _StubCVRP()
    inst = _make_instance(5)
    initial = _route_text("Route 1: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0")
    initial_item = (initial, False, len(initial))

    gen = _MockGen([])  # 不应被调用
    final_item, info = _retry_loop_one(
        gen, _ORIG_PROMPT, inst, prob, "cvrp", initial_item,
        max_completion_length=512, temperature=0.0, max_rounds=3,
    )
    assert info["rounds_used"] == 0, info
    assert info["converged"] is True
    assert info["stopped_reason"] == "feasible"
    assert len(gen.calls) == 0, "第一轮可行不应调 generate"
    print("✓ test_1_first_round_feasible")


def test_2_missing_then_complete():
    """场景 2: 第一轮缺节点 3, 第二轮补齐."""
    prob = _StubCVRP()
    inst = _make_instance(5)
    bad = _route_text("Route 1: 0 -> 1 -> 2 -> 4 -> 5 -> 0")
    good = _route_text("Route 1: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0")
    initial_item = (bad, False, len(bad))

    gen = _MockGen([good])
    final_item, info = _retry_loop_one(
        gen, _ORIG_PROMPT, inst, prob, "cvrp", initial_item,
        max_completion_length=512, temperature=0.0, max_rounds=3,
    )
    assert info["rounds_used"] == 1, info
    assert info["converged"] is True
    assert info["final_missing"] == []
    assert len(gen.calls) == 1
    # 第一句直接陈述问题: "You missed customers [3] in your previous answer."
    last_user = gen.calls[0]["last_user_msg"]
    assert last_user.startswith("You missed customers [3]"), \
        f"第一句应直接列出缺失: {last_user[:200]}"
    assert "exactly once" in last_user
    assert "1..5" in last_user
    # CVRP 应该提到 capacity (其它约束 bullet)
    assert "capacity" in last_user.lower()
    # 目标 bullet
    assert "minimized" in last_user.lower()
    print("✓ test_2_missing_then_complete")


def test_3_duplicate_then_complete():
    """场景 3: 第一轮重复 2, 第二轮去重."""
    prob = _StubCVRP()
    inst = _make_instance(5)
    bad = _route_text("Route 1: 0 -> 1 -> 2 -> 2 -> 4 -> 5 -> 0")  # 缺 3 重 2
    good = _route_text("Route 1: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0")
    initial_item = (bad, False, len(bad))

    gen = _MockGen([good])
    final_item, info = _retry_loop_one(
        gen, _ORIG_PROMPT, inst, prob, "cvrp", initial_item,
        max_completion_length=512, temperature=0.0, max_rounds=3,
    )
    assert info["rounds_used"] == 1, info
    assert info["converged"] is True
    last_user = gen.calls[0]["last_user_msg"]
    # 既缺 [3] 又重 [2], 第一句必须两个都列出且用 "and" 连接
    assert "You missed customers [3]" in last_user, last_user[:200]
    assert "visited customers [2] more than once" in last_user, last_user[:200]
    assert " and " in last_user
    print("✓ test_3_duplicate_then_complete")


def test_4_parse_fail_then_ok():
    """场景 4: 第一轮 parse 失败 (无 Route 字样), 第二轮 OK."""
    prob = _StubCVRP()
    inst = _make_instance(5)
    bad = "<think>blah</think>\n\nI'm not sure, sorry."
    good = _route_text("Route 1: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0")
    initial_item = (bad, False, len(bad))

    gen = _MockGen([good])
    final_item, info = _retry_loop_one(
        gen, _ORIG_PROMPT, inst, prob, "cvrp", initial_item,
        max_completion_length=512, temperature=0.0, max_rounds=3,
    )
    assert info["rounds_used"] == 1
    assert info["converged"] is True
    last_user = gen.calls[0]["last_user_msg"]
    # parse 失败的第一句是 "could not be parsed", 后续 bullet 一样要全
    assert "could not be parsed" in last_user
    assert "1..5" in last_user
    assert "exactly once" in last_user
    assert "minimized" in last_user.lower()
    print("✓ test_4_parse_fail_then_ok")


def test_5_never_converge_hits_max_rounds():
    """场景 5: 连续 3 轮都缺节点, 撞 max_rounds."""
    prob = _StubCVRP()
    inst = _make_instance(5)
    bad = _route_text("Route 1: 0 -> 1 -> 2 -> 4 -> 5 -> 0")  # 缺 3
    initial_item = (bad, False, len(bad))

    gen = _MockGen([bad, bad, bad])  # retry 3 次都缺
    final_item, info = _retry_loop_one(
        gen, _ORIG_PROMPT, inst, prob, "cvrp", initial_item,
        max_completion_length=512, temperature=0.0, max_rounds=3,
    )
    assert info["rounds_used"] == 3, info
    assert info["converged"] is False
    assert info["stopped_reason"] == "max_rounds"
    assert info["final_missing"] == [3]
    assert len(gen.calls) == 3
    print("✓ test_5_never_converge_hits_max_rounds")


def test_6_cov_ok_capacity_violation_breaks():
    """场景 6: cov=1 + 无重复, 但 capacity 违例 → 早停 (retry 救不了 cap)."""
    # demand 让 customer 1+2+3+4+5 = 1.0 + 0 + 0 + 0 + 0 = 1.0 边界 OK,
    # 但单条路线全装下: 把 demand[1]=0.6, demand[2]=0.6 → 单路 1+2 = 1.2 > 1.0 cap
    prob = _StubCVRP(capacity_demands={1: 0.6, 2: 0.6})
    inst = _make_instance(5)
    bad_cap = _route_text("Route 1: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0")  # cov OK 但超 cap
    initial_item = (bad_cap, False, len(bad_cap))

    gen = _MockGen([])  # 不应被调用 (cov OK 早停)
    final_item, info = _retry_loop_one(
        gen, _ORIG_PROMPT, inst, prob, "cvrp", initial_item,
        max_completion_length=512, temperature=0.0, max_rounds=3,
    )
    assert info["rounds_used"] == 0, info
    assert info["converged"] is False
    assert info["stopped_reason"] == "cov_ok_other_violation"
    assert info["final_missing"] == []
    assert info["final_duplicates"] == []
    assert len(gen.calls) == 0, "cov OK 但 cap 违例时不应 retry"
    print("✓ test_6_cov_ok_capacity_violation_breaks")


def test_7_strip_think_edge_cases():
    """场景 7: _strip_think_for_history 各边界."""
    # 正常: 剥到 </think> 之后
    assert _strip_think_for_history(
        "<think>...</think>\n\nRoute 1: 0 -> 1 -> 0"
    ) == "Route 1: 0 -> 1 -> 0"

    # 多个 </think>: 用最后一个 (rfind)
    out = _strip_think_for_history("</think>x</think>\nfinal")
    assert out == "final"

    # 没 </think> (truncated): 占位
    out = _strip_think_for_history("<think>still thinking when cut off")
    assert "truncated" in out.lower()

    # 有 </think> 但后面空: 第二种占位
    out = _strip_think_for_history("<think>blah</think>\n   \n")
    assert "no final route" in out.lower()
    print("✓ test_7_strip_think_edge_cases")


def test_8_diagnose_routes_parse_fail_returns_all_missing():
    """场景 8: parse 失败时 missing 应包含全部 1..n (用于 feedback 提示)."""
    prob = _StubCVRP()
    inst = _make_instance(5)
    diag = _diagnose_routes("garbage no routes here", inst, prob)
    assert diag["parse_ok"] is False
    assert diag["missing"] == [1, 2, 3, 4, 5]
    assert diag["duplicates"] == []
    assert diag["feasible_strict"] is False
    print("✓ test_8_diagnose_routes_parse_fail_returns_all_missing")


def test_9_feedback_single_route_tsp():
    """场景 9: TSP (单路线无其它约束) — 框架第一句直陈 + bullet 清单."""
    diag = {"parse_ok": True, "missing": [3], "duplicates": []}
    msg = _build_retry_feedback(diag, multi_route=False, n=10, problem_type="tsp")
    # 第一句直陈
    assert msg.startswith("You missed customers [3]"), msg[:200]
    # 单路线格式
    assert "Route: 0 -> ... -> 0" in msg
    assert "Route 1:" not in msg
    # bullet: 1..n + depot 首尾 + minimize distance
    assert "1..10" in msg
    assert "exactly twice" in msg
    assert "not appear in the middle" in msg
    assert "minimized" in msg.lower()
    # TSP 无其它约束 bullet (cap/TW/DL)
    assert "capacity" not in msg.lower()
    assert "time window" not in msg.lower()
    assert "draft limit" not in msg.lower()
    print("✓ test_9_feedback_single_route_tsp")


def test_10_feedback_multi_route_cvrp():
    """场景 10: CVRP (多路线 + cap) — depot 每 route 首尾 + capacity bullet."""
    diag = {"parse_ok": True, "missing": [3], "duplicates": []}
    msg = _build_retry_feedback(diag, multi_route=True, n=20, problem_type="cvrp")
    assert msg.startswith("You missed customers [3]")
    assert "1..20" in msg
    assert "first and last node of every route" in msg.lower()
    assert "capacity" in msg.lower()
    assert "time window" not in msg.lower()  # CVRP 无 TW
    assert "minimized" in msg.lower()
    print("✓ test_10_feedback_multi_route_cvrp")


def test_11_tsp_retry_loop_end_to_end():
    """场景 11: TSP retry loop 端到端 (multi_route=False 分支)."""
    prob = _StubTSP()
    inst = _make_instance(5)
    bad = _route_text("Route: 0 -> 1 -> 2 -> 4 -> 5 -> 0")
    good = _route_text("Route: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0")
    initial_item = (bad, False, len(bad))

    gen = _MockGen([good])
    final_item, info = _retry_loop_one(
        gen, _ORIG_PROMPT, inst, prob, "tsp", initial_item,
        max_completion_length=512, temperature=0.0, max_rounds=3,
    )
    assert info["rounds_used"] == 1 and info["converged"] is True
    last_user = gen.calls[0]["last_user_msg"]
    assert last_user.startswith("You missed customers [3]")
    assert "Route: 0 -> ... -> 0" in last_user
    print("✓ test_11_tsp_retry_loop_end_to_end")


def test_12_feedback_vrptw_cap_plus_tw():
    """场景 12: VRPTW — 必含 capacity + time windows."""
    diag = {"parse_ok": True, "missing": [3], "duplicates": []}
    msg = _build_retry_feedback(diag, multi_route=True, n=20, problem_type="vrptw")
    assert msg.startswith("You missed customers [3]")
    assert "capacity" in msg.lower()
    assert "time window" in msg.lower()
    assert "minimized" in msg.lower()
    print("✓ test_12_feedback_vrptw_cap_plus_tw")


def test_13_feedback_tsptw_time_window_only():
    """场景 13: TSPTW — 必含 time windows, 不应提 capacity."""
    diag = {"parse_ok": True, "missing": [3], "duplicates": []}
    msg = _build_retry_feedback(diag, multi_route=False, n=10, problem_type="tsptw")
    assert msg.startswith("You missed customers [3]")
    assert "time window" in msg.lower()
    assert "capacity" not in msg.lower()
    assert "draft limit" not in msg.lower()
    print("✓ test_13_feedback_tsptw_time_window_only")


def test_14_feedback_tspdl_draft_limit_only():
    """场景 14: TSPDL — 必含 draft limit."""
    diag = {"parse_ok": True, "missing": [3], "duplicates": []}
    msg = _build_retry_feedback(diag, multi_route=False, n=10, problem_type="tspdl")
    assert msg.startswith("You missed customers [3]")
    assert "draft limit" in msg.lower()
    assert "time window" not in msg.lower()
    assert "capacity" not in msg.lower()
    print("✓ test_14_feedback_tspdl_draft_limit_only")


def test_15_first_sentence_dup_only():
    """场景 15: 仅重复无缺失, 第一句应为 'You visited ... more than once'."""
    diag = {"parse_ok": True, "missing": [], "duplicates": [4, 7]}
    msg = _build_retry_feedback(diag, multi_route=True, n=10, problem_type="cvrp")
    assert msg.startswith("You visited customers [4, 7] more than once"), msg[:200]
    assert " and " not in msg.split("\n")[0]  # 第一句没有 and 连接 (只有重复一类)
    print("✓ test_15_first_sentence_dup_only")


def test_16_first_sentence_both_missed_and_dup():
    """场景 16: 缺 + 重 同时存在, 第一句必须用 'and' 连接两部分."""
    diag = {"parse_ok": True, "missing": [3, 8], "duplicates": [5]}
    msg = _build_retry_feedback(diag, multi_route=True, n=10, problem_type="cvrp")
    first_line = msg.split("\n")[0]
    assert "missed customers [3, 8]" in first_line
    assert "visited customers [5] more than once" in first_line
    assert " and " in first_line
    print("✓ test_16_first_sentence_both_missed_and_dup")


# ── Runner ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Smoke test: evaluate.retry_until_feasible 路径")
    print("=" * 60)
    test_1_first_round_feasible()
    test_2_missing_then_complete()
    test_3_duplicate_then_complete()
    test_4_parse_fail_then_ok()
    test_5_never_converge_hits_max_rounds()
    test_6_cov_ok_capacity_violation_breaks()
    test_7_strip_think_edge_cases()
    test_8_diagnose_routes_parse_fail_returns_all_missing()
    test_9_feedback_single_route_tsp()
    test_10_feedback_multi_route_cvrp()
    test_11_tsp_retry_loop_end_to_end()
    test_12_feedback_vrptw_cap_plus_tw()
    test_13_feedback_tsptw_time_window_only()
    test_14_feedback_tspdl_draft_limit_only()
    test_15_first_sentence_dup_only()
    test_16_first_sentence_both_missed_and_dup()
    print("=" * 60)
    print("  ✅ 全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
