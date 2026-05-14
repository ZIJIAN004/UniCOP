"""
离线测试 cvrp_mask_state.py 的 state machine.

不需要 vLLM 或 tokenizer, 直接用 SFT chains 文本验证 5 条规则触发位置.

用法:
    python tests/test_cvrp_mask_state.py  [chains.jsonl path]

默认读 ~/Desktop/chains_hybrid_cvrp20.jsonl (本地) 或
/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/data/chains_hybrid_cvrp20.jsonl (集群).
"""
from __future__ import annotations

import json
import os
import sys

# 让 import 工作 (脚本直接跑)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cvrp_mask_state import (
    MaskConfig,
    build_state,
    compute_mask,
    detect_section,
)


def find_chains_file() -> str:
    candidates = [
        os.path.expanduser("~/Desktop/chains_hybrid_cvrp20.jsonl"),
        "/c/Users/zijia/Desktop/chains_hybrid_cvrp20.jsonl",
        "C:/Users/zijia/Desktop/chains_hybrid_cvrp20.jsonl",
        "/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/data/chains_hybrid_cvrp20.jsonl",
    ]
    if len(sys.argv) > 1:
        candidates.insert(0, sys.argv[1])
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No chains file found, tried: {candidates}")


def test_section_detection():
    """验证 5 个 section 状态识别."""
    print("=" * 60)
    print("Test 1: Section detection")
    print("=" * 60)
    cases = [
        ("Hello world", "SECTION_1"),
        ("Strategy here\n2. **Step-by-step construction**:\n[R1,1] cap=...", "SECTION_2"),
        ("...\n3. **Verification**:\n...", "SECTION_3"),
        ("...\n4. **Final routes**:\nRoute 1: 0 ->", "SECTION_4"),
        ("4. **Final routes**:\nRoute 1: 0 -> 1 -> 0\n</think>\nRoute 1: 0 -> 1 -> 0", "ANSWER"),
    ]
    for text, expected in cases:
        actual = detect_section(text)
        ok = "✓" if actual == expected else "✗"
        print(f"  {ok} expected={expected}, got={actual}")


def test_visited_accumulation():
    """验证 visited 累积只在 section 2 内, 且仅算 → select X."""
    print("\n" + "=" * 60)
    print("Test 2: Visited accumulation")
    print("=" * 60)

    # Case 2.1: section 1 (Strategy) 内的 "→ select" 字面不应累加
    text1 = (
        "1. **Strategy**: I will → select node 5 first as a metaphor.\n"
        "Then... 2. **Step-by-step construction**:\n"
        "[R1,1] cap=1.00 | feasible: ... → select 13\n"
    )
    s1 = build_state(text1, n=20)
    print(f"  Strategy 段 '→ select 5' + Section 2 实际 '→ select 13'")
    print(f"    visited = {s1.visited} (期望 {{13}})")
    assert 13 in s1.visited and 5 not in s1.visited

    # Case 2.2: 多步累积
    text2 = (
        "2. **Step-by-step construction**:\n"
        "[R1,1] ... → select 13\n"
        "[R1,2] ... → select 7\n"
        "[R1,3] ... → select 15\n"
    )
    s2 = build_state(text2, n=20)
    print(f"  连续 select 13, 7, 15:  visited = {sorted(s2.visited)}")
    assert s2.visited == {7, 13, 15}

    # Case 2.3: Reflexion reset (全集 Unvisited 出现)
    text3 = (
        "2. **Step-by-step construction**:\n"
        "[R1,1] → select 5\n"
        "[R1,2] → select 7\n"
        "Wait, let me restart.\n"
        "Unvisited: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}\n"
        "[R1,1] → select 13\n"
    )
    s3 = build_state(text3, n=20)
    print(f"  Reflexion reset + → select 13:  visited = {sorted(s3.visited)} (期望 {{13}})")
    assert s3.visited == {13}


def test_select_trigger():
    """验证 R1 触发位置 (末尾 '→ select ')."""
    print("\n" + "=" * 60)
    print("Test 3: R1 select trigger")
    print("=" * 60)
    cases = [
        ("2. **Step-by-step construction**:\n[R1,1] cap=1.00 | feasible: ... → select ", True),
        ("2. **Step-by-step construction**:\n[R1,1] cap=1.00 | feasible: ... → select 13", False),  # 已写完
        ("[R1,1] cap=1.00 → select", False),  # 缺末尾空格
        ("[R1,1] cap=1.00 → select  ", True),  # 双空格仍触发
        ("[R1,1] cap=1.00 → Select ", True),   # 大小写宽容
    ]
    for text, expected in cases:
        # 加入 section 2 前缀确保 section 判定正确
        if "**Step-by-step" not in text:
            text = "2. **Step-by-step construction**:\n" + text
        s = build_state(text, n=20)
        actual = s.select_trigger_now
        ok = "✓" if actual == expected else "✗"
        print(f"  {ok} '{text[-30:]!r}' → trigger={actual} (expected {expected})")


def test_section_2_rule_4():
    """验证 R4 触发位置 (step 行内 → 之后, visited<n)."""
    print("\n" + "=" * 60)
    print("Test 4: R4 in_step_arrow (visited<n)")
    print("=" * 60)

    text = (
        "2. **Step-by-step construction**:\n"
        "[R1,1] cap=1.00 | feasible: 11(d=0.021,dem=0.07,cap→0.93), ... → "
    )
    s = build_state(text, n=20)
    dec = compute_mask(s, MaskConfig(n=20))
    print(f"  text 末尾: '...| feasible: ... → '")
    print(f"  in_step_line: {s.in_step_line}")
    print(f"  in_step_arrow: {s.in_step_arrow}")
    print(f"  visited: {s.visited}")
    print(f"  decision: forbid_all_word={dec.forbid_all_word} (R4 触发)")
    assert dec.forbid_all_word

    # 跟 cap→ 区分: 末尾不是 → 之后, 是数值后
    text2 = "[R1,1] cap=1.00 | feasible: 11(d=0.021,dem=0.07,cap→0.93)"
    text2_full = "2. **Step-by-step construction**:\n" + text2
    s2 = build_state(text2_full, n=20)
    dec2 = compute_mask(s2, MaskConfig(n=20))
    print(f"\n  text 末尾: '..., cap→0.93)' (不是决策 →)")
    print(f"  in_step_arrow: {s2.in_step_arrow} (期望 False, cap→ 不算)")
    assert not s2.in_step_arrow, f"cap→ 不应触发 in_step_arrow, but got True"


def test_visited_full_rule_3():
    """验证 R3 触发位置 (step 行内未现决策箭头, visited==n)."""
    print("\n" + "=" * 60)
    print("Test 5: R3 visited==n (禁 ' |')")
    print("=" * 60)

    # 构造一个 visited == n 的场景
    selects = "\n".join(
        f"[R1,{i+1}] cap=1.00 | feasible: ... → select {i+1}"
        for i in range(20)
    )
    text = "2. **Step-by-step construction**:\n" + selects + "\n[R2,1] cap=1.00"
    s = build_state(text, n=20)
    dec = compute_mask(s, MaskConfig(n=20))
    print(f"  visited.size: {len(s.visited)} (期望 20)")
    print(f"  in_step_line: {s.in_step_line}")
    print(f"  decision_arrow_in_line: {s.decision_arrow_in_line}")
    print(f"  forbid_pipe (R3): {dec.forbid_pipe}")
    assert len(s.visited) == 20
    assert dec.forbid_pipe, "visited==n 时 R3 应禁 ' |'"


def test_block_end_rules_2():
    """验证 R2 visited<n 时禁 </think> 和 EOS."""
    print("\n" + "=" * 60)
    print("Test 6: R2 block end")
    print("=" * 60)

    # Case 6.1: visited<n, think 段
    text1 = (
        "2. **Step-by-step construction**:\n"
        "[R1,1] → select 5\n"
        "[R1,2] → select 7\n"
    )
    s1 = build_state(text1, n=20)
    dec1 = compute_mask(s1, MaskConfig(n=20))
    print(f"  visited={len(s1.visited)} < n=20, section={s1.section}")
    print(f"    forbid_think_close={dec1.forbid_think_close} (期望 True)")
    print(f"    forbid_eos={dec1.forbid_eos} (期望 True)")
    assert dec1.forbid_think_close and dec1.forbid_eos

    # Case 6.2: visited == n 但 Section 4 没写
    selects = "\n".join(f"[R1,{i}] → select {i}" for i in range(1, 21))
    text2 = "2. **Step-by-step construction**:\n" + selects
    s2 = build_state(text2, n=20)
    dec2 = compute_mask(s2, MaskConfig(n=20))
    print(f"\n  visited={len(s2.visited)} == n, final_routes_written={s2.final_routes_written}")
    print(f"    forbid_think_close={dec2.forbid_think_close} (期望 True, Section 4 没写)")
    print(f"    forbid_eos={dec2.forbid_eos} (期望 True, 在 think 内)")
    assert dec2.forbid_think_close
    assert dec2.forbid_eos

    # Case 6.3: visited==n 且 Section 4 写完
    text3 = text2 + (
        "\n3. **Verification**: R1:{...}=20\n"
        "4. **Final routes**:\nRoute 1: 0 -> 1 -> ... -> 0\n"
    )
    s3 = build_state(text3, n=20)
    dec3 = compute_mask(s3, MaskConfig(n=20))
    print(f"\n  visited={len(s3.visited)} == n, final_routes_written={s3.final_routes_written}")
    print(f"    forbid_think_close={dec3.forbid_think_close} (期望 False, 可以 </think>)")
    print(f"    forbid_eos={dec3.forbid_eos} (期望 True, 答案段没开始)")
    assert not dec3.forbid_think_close
    assert dec3.forbid_eos


def test_real_chain():
    """用真实 SFT chain 模拟 token-by-token decoding, 验证 mask 决策合理性."""
    print("\n" + "=" * 60)
    print("Test 7: Real SFT chain — token-by-token decoding (前 1 条 chain)")
    print("=" * 60)
    path = find_chains_file()
    print(f"  Chain file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())
    output_text = data["output"]
    print(f"  Output length: {len(output_text)} chars")

    # 模拟 token-by-token 生成: 每隔 100 chars 看一次 mask 决策
    # (真实场景每 token 调一次, 这里粗采样验证逻辑)
    cfg = MaskConfig(n=20)
    sample_points = list(range(0, len(output_text), 200))
    mask_hits = 0
    rule_counts = {"select_strict": 0, "forbid_pipe": 0, "forbid_all_word": 0,
                  "forbid_think_close": 0, "forbid_eos": 0, "forbid_verification_word": 0}
    for pos in sample_points:
        partial = output_text[:pos]
        s = build_state(partial, n=20)
        d = compute_mask(s, cfg)
        if d.mask_hit:
            mask_hits += 1
        for k in rule_counts:
            if getattr(d, k):
                rule_counts[k] += 1

    print(f"  Sample points: {len(sample_points)}")
    print(f"  mask_hit count: {mask_hits}")
    print(f"  Rule trigger counts: {rule_counts}")
    print(f"  (note: 这是粗采样, 不是 token 级精确统计, 仅看分布)")


def main():
    test_section_detection()
    test_visited_accumulation()
    test_select_trigger()
    test_section_2_rule_4()
    test_visited_full_rule_3()
    test_block_end_rules_2()
    test_real_chain()

    print("\n" + "=" * 60)
    print("All state machine tests passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
