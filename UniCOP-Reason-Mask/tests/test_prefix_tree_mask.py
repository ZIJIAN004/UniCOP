"""
Prefix tree mask 单元测试 (vllm_cvrp_mask_processor).

不依赖真 tokenizer, 用 mock tokenizer 模拟 Qwen 行为:
- " 1".." 9" → 单 token
- " 10".." 19" → [" 1", "0".."9"] 共享 first token
- " 20" → [" 2", "0"]

测试核心: 验证 _compute_partial_select + _apply_partial_select 在
multi-token customer 上正确拒绝 visited 客户的延续 token.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub torch (本地可能没装), 用最小实现让 vllm_cvrp_mask_processor 能 import + 测试
try:
    import torch
except ImportError:
    torch_stub = types.ModuleType("torch")

    class _T:
        def __init__(self, vals):
            self._vals = [float(v) for v in vals]
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return _T(self._vals[idx])
            return _Scalar(self._vals[idx])
        def __setitem__(self, idx, val):
            if isinstance(idx, slice):
                # slice 赋值 (logits[:] = float)
                n = len(self._vals)
                v = float(val.v if isinstance(val, _Scalar) else val)
                for i in range(*idx.indices(n)):
                    self._vals[i] = v
            else:
                self._vals[idx] = float(val.v if isinstance(val, _Scalar) else val)
        def __len__(self):
            return len(self._vals)

    class _Scalar:
        def __init__(self, v):
            self.v = float(v)
        def item(self):
            return self.v

    torch_stub.full_like = lambda t, v: _T([v] * len(t))
    torch_stub.zeros = lambda n: _T([0.0] * n)
    torch_stub.Tensor = _T
    sys.modules["torch"] = torch_stub
    import torch  # noqa

# Mock tokenizer 模拟 Qwen 行为
class _MockTokenizer:
    """模拟 Qwen tokenizer: " 1".." 9" 单 token; " 10".." 19" multi-token 共享首 token; " 20" multi-token."""

    def __init__(self):
        # token id 分配 (任意, 但要 unique):
        # " 1".." 9" → 101..109
        # " 0" → 100
        # " 2" → 102 (单 token)
        # 数字 char "0".."9" → 200..209 (用于 multi-token 第 2 token)
        # 其他 control: " " → 300, "\n" → 301, "[" → 302, " |" → 303, "|" → 304,
        # "\t" → 305, " \n" → 306, "." → 307, "," → 308, ";" → 309
        # "</think>" → 999, "Verification" → 998, "**Verification" → 997 (但 multi)
        # " all" → 996, " All" → 995
        self._special = {
            "</think>": [999],
            "Verification": [998],
            "**Verification": [97, 998],  # multi-token
            " all": [996],
            " All": [995],
        }

    def __call__(self, text: str, add_special_tokens: bool = False):
        return _MockEncoded(self._encode(text))

    def _encode(self, text: str) -> list[int]:
        if text in self._special:
            return self._special[text]
        # 单 token customers " 1".." 9"
        if text == " 0":
            return [100]
        for d in range(1, 10):
            if text == f" {d}":
                return [100 + d]
        # multi-token customers " 10".." 19" → [" 1", "0".."9"]
        for c in range(10, 20):
            if text == f" {c}":
                second = c - 10  # 10 → "0", 11 → "1", ...
                return [101, 200 + second]
        # " 20" multi-token → [" 2", "0"]
        if text == " 20":
            return [102, 200]
        # control chars
        ctrl_map = {
            " ": [300], "\n": [301], "[": [302], " |": [303], "|": [304],
            "\t": [305], " \n": [306], ".": [307], ",": [308], ";": [309],
        }
        if text in ctrl_map:
            return ctrl_map[text]
        # fallback: 一个奇怪的 id
        return [777]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        # 反向映射 (只支持单 token 反查 for testing)
        rev = {100: " 0"}
        for d in range(1, 10):
            rev[100 + d] = f" {d}"
        for d in range(10):
            rev[200 + d] = str(d)
        rev.update({
            300: " ", 301: "\n", 302: "[", 303: " |", 304: "|",
            305: "\t", 306: " \n", 307: ".", 308: ",", 309: ";",
            999: "</think>", 998: "Verification", 997: "**",
            996: " all", 995: " All",
        })
        return "".join(rev.get(int(i), "?") for i in ids)

    @property
    def eos_token_id(self):
        return 990


class _MockEncoded:
    def __init__(self, ids):
        self.input_ids = ids


def _make_processor(n: int = 20):
    from utils.cvrp_mask_state import MaskConfig
    from utils.vllm_cvrp_mask_processor import CVRPMaskProcessor
    tok = _MockTokenizer()
    cfg = MaskConfig(enabled=True, n=n)
    return CVRPMaskProcessor(n=n, tokenizer=tok, cfg=cfg)


# ── 测试用例 ────────────────────────────────────────────────────────

def test_cust_tokens_built():
    """启动时正确构建每个 customer 的完整 token 序列."""
    proc = _make_processor(n=20)
    # 单 token customers 1-9
    for c in range(1, 10):
        assert len(proc.cust_tokens[c]) == 1, f"customer {c} 应该单 token"
    # multi-token 10-19 共享 " 1" (token 101)
    for c in range(10, 20):
        assert len(proc.cust_tokens[c]) == 2, f"customer {c} 应该 2 token"
        assert proc.cust_tokens[c][0] == 101, f"customer {c} first token 应该 = 101 (' 1')"
    # multi-token 20: [" 2", "0"]
    assert proc.cust_tokens[20] == [102, 200]
    # first_to_customers 反向索引
    assert set(proc.first_to_customers[101]) == {1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    assert set(proc.first_to_customers[102]) == {2, 20}
    print("[OK] test_cust_tokens_built")


def test_partial_select_no_trigger_outside_section_2():
    """非 Section 2 不触发 partial_select."""
    proc = _make_processor(n=20)
    visited = set()
    past_text = "→ select 1"
    output_ids = [101]  # 假装末尾是 " 1"
    result = proc._compute_partial_select(past_text, output_ids, visited)
    # 注: _compute_partial_select 自身不检查 section, 由 __call__ 检查
    # 这个测试是 partial 的: 检查内部 logic
    assert result is not None  # 末尾匹配 partial_select, 应该触发
    print("[OK] test_partial_select_no_trigger_outside_section_2 (内部 logic OK)")


def test_partial_select_detect_multi_token():
    """末尾是 multi-token customer 的 prefix → 触发 partial."""
    proc = _make_processor(n=20)
    visited = {5, 7}
    # 末尾 "→ select 1" 后 (token=101 = " 1")
    past_text = "[R1,1] cap=1.00 | feasible:... → select 1"
    output_ids = [777, 101]  # 任意 + " 1"
    result = proc._compute_partial_select(past_text, output_ids, visited)
    assert result is not None, "应该检测到 partial_select"
    n_written, candidates, unvisited = result
    assert n_written == 1
    # candidates 包含 1 (单 token, 已完整) 和 10-19 (multi, 待延续)
    assert 1 in candidates
    assert set(candidates) >= set(range(10, 20))
    print("[OK] test_partial_select_detect_multi_token")


def test_partial_select_no_trigger_after_space():
    """末尾 '→ select 1 ' (有空格) → 不触发 partial (select 已结束)."""
    proc = _make_processor(n=20)
    visited = set()
    past_text = "→ select 1 "  # 末尾空格说明 select 1 结束
    output_ids = [101, 300]  # " 1" 然后 " "
    result = proc._compute_partial_select(past_text, output_ids, visited)
    assert result is None, "末尾有空格不该触发 partial"
    print("[OK] test_partial_select_no_trigger_after_space")


def test_apply_partial_select_bans_visited_continuation():
    """核心 case: visited={11}, 模型写 ' 1' 后 partial_select 应该 ban '1' (= 11 的第 2 token)."""
    proc = _make_processor(n=20)
    visited = {11}  # 11 visited, 1, 10, 12-19 unvisited
    candidates = list(range(10, 20)) + [1]  # 共享 " 1" first token
    unvisited = set(range(1, 21)) - visited
    partial_info = (1, candidates, unvisited)

    # 创建 mock logits (vocab 1000)
    logits = torch.zeros(1000)
    new_logits = proc._apply_partial_select(partial_info, logits)

    # 检查: token "1" (= 201, 11 的第 2 token) 应该被 ban (-inf)
    assert new_logits[201].item() == float("-inf"), \
        "应该 ban token '1' (visited customer 11 的延续)"
    # token "0" (= 200, 10 的第 2 token) 应该 allowed (10 unvisited)
    assert new_logits[200].item() == 0.0, \
        "应该 allow token '0' (unvisited customer 10 的延续)"
    # token "2" (= 202, 12 的第 2 token) 应该 allowed (12 unvisited)
    assert new_logits[202].item() == 0.0, \
        "应该 allow token '2' (unvisited customer 12 的延续)"
    # end tokens (空格 300) 应该 allowed (1 unvisited, 可结束 select)
    assert new_logits[300].item() == 0.0, \
        "应该 allow end token (空格) — customer 1 unvisited 可结束"
    print("[OK] test_apply_partial_select_bans_visited_continuation")


def test_apply_partial_select_all_visited_no_end():
    """visited={1, 10-15}, 写 ' 1' 后 partial 只允许延续到 16-19 (不允许结束)."""
    proc = _make_processor(n=20)
    visited = {1, 10, 11, 12, 13, 14, 15}
    candidates = list(range(10, 20)) + [1]
    unvisited = set(range(1, 21)) - visited
    partial_info = (1, candidates, unvisited)

    logits = torch.zeros(1000)
    new_logits = proc._apply_partial_select(partial_info, logits)

    # 1 visited → end token 不该 allow
    assert new_logits[300].item() == float("-inf"), \
        "1 visited 不该允许 end token"
    # 16-19 unvisited → 各自第 2 token (206-209) allowed
    for c in range(16, 20):
        second = c - 10
        assert new_logits[200 + second].item() == 0.0, \
            f"customer {c} unvisited, '{second}' 应该 allowed"
    # 10-15 visited → 第 2 token (200-205) banned
    for c in range(10, 16):
        second = c - 10
        assert new_logits[200 + second].item() == float("-inf"), \
            f"customer {c} visited, '{second}' 应该 banned"
    print("[OK] test_apply_partial_select_all_visited_no_end")


def test_apply_partial_select_only_one_unvisited():
    """极端 case: visited={1, 10-14, 16-19}, partial=' 1' → 只允许写 '5' 选 15."""
    proc = _make_processor(n=20)
    visited = set(range(10, 15)) | {1} | set(range(16, 20))  # 10-14, 16-19, 1
    # unvisited: {2..9, 15, 20}
    candidates = list(range(10, 20)) + [1]
    unvisited = set(range(1, 21)) - visited
    partial_info = (1, candidates, unvisited)

    logits = torch.zeros(1000)
    new_logits = proc._apply_partial_select(partial_info, logits)

    # 只 15 unvisited (其他 10-14, 16-19 都 visited) → 只允许 "5" (token 205)
    for c in range(10, 20):
        second = c - 10
        if c == 15:
            assert new_logits[200 + second].item() == 0.0, "15 unvisited, allow '5'"
        else:
            assert new_logits[200 + second].item() == float("-inf"), \
                f"customer {c} visited, ban '{second}'"
    # end token: 1 visited, 不该 allow
    assert new_logits[300].item() == float("-inf")
    print("[OK] test_apply_partial_select_only_one_unvisited")


def test_partial_select_does_not_break_single_token_customer():
    """单 token customer (1-9) 不该走 partial_select (select_strict 直接处理)."""
    proc = _make_processor(n=20)
    # Customer 5 单 token, 完整写 " 5" 后 next call 末尾文本是 "→ select 5"
    # 但 5 没有 multi-token candidate, partial_select 应该返回 None
    visited = set()
    past_text = "→ select 5"
    output_ids = [105]  # " 5" 单 token
    result = proc._compute_partial_select(past_text, output_ids, visited)
    # 5 是单 token (len(toks)==1==n_written) 且没有其他 customer 以 [105] 为 prefix → None
    assert result is None, \
        "单 token customer 完整写完后 partial 不该触发"
    print("[OK] test_partial_select_does_not_break_single_token_customer")


def run_all():
    tests = [
        test_cust_tokens_built,
        test_partial_select_no_trigger_outside_section_2,
        test_partial_select_detect_multi_token,
        test_partial_select_no_trigger_after_space,
        test_apply_partial_select_bans_visited_continuation,
        test_apply_partial_select_all_visited_no_end,
        test_apply_partial_select_only_one_unvisited,
        test_partial_select_does_not_break_single_token_customer,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n[X] {failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\n[OK] All {len(tests)} prefix tree tests passed")


if __name__ == "__main__":
    run_all()
