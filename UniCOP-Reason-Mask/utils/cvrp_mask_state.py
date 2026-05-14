"""
CVRP 思维链 constrained decoding state machine.

纯文本 state machine, 不依赖 vLLM 或 tokenizer. 给定一段已生成的文本,
计算当前 cursor 位置应该应用哪些 mask 规则.

设计原则:
  - Stateless 重算: 每次从完整 output_text 重建 state, 不维护跨调用 state
    (vLLM 0.7.3 worker process + tuple property 行为, stateful 不可靠, 见调研结论)
  - 5 条 mask 规则, 通过 MaskConfig 超参数控制
  - Section 状态机识别 think 5 个区段

参考: CVRP-LLM-Mask-完整规则.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Section 锚点 (按 SFT chain 严格定义) ──────────────────────────────
SECTION_2_ANCHOR = "2. **Step-by-step construction**:"
SECTION_3_ANCHOR = "3. **Verification**:"
SECTION_4_ANCHOR = "4. **Final routes**:"
THINK_CLOSE_LITERAL = "</think>"


# ── 决策箭头 (跟 cap→X.XX 数值箭头区分) ─────────────────────────────
DECISION_ARROW_RE = re.compile(
    r"→\s+(select|no\s+feasible|all\s+customers|remaining)",
    re.IGNORECASE,
)
SELECT_TRIGGER_RE = re.compile(r"→\s*[Ss]elect\s+$")   # 末尾匹配 = mask 触发
STEP_LINE_RE = re.compile(r"\[R\d+,\d+\][^\n]*$")       # 末尾在 step 行内
STEP_LINE_WITH_ARROW_RE = re.compile(r"\[R\d+,\d+\][^\n]*→\s*$")   # step 行 → 后
SELECT_ACCUM_RE = re.compile(r"→\s*[Ss]elect\s+(\d+)")  # 全文累积 visited
ANSWER_ROUTE_TOKEN_RE = re.compile(r"->\s*(\d+)")       # 答案段累积 answer_visited
UNVISITED_RESET_RE = re.compile(r"Unvisited:\s*\{([^}]*)\}")  # reflexion reset (任意位置)


@dataclass
class MaskConfig:
    """Mask 行为的所有超参数. 跟 config.py 字段对齐."""
    n: int = 20                             # CVRP n
    apply_select: bool = True               # R1
    apply_block_end: bool = True            # R2
    apply_visited_n: bool = True            # R3
    apply_block_all: bool = True            # R4
    apply_block_verif: bool = True          # R5
    debug_log: bool = False


@dataclass
class CVRPState:
    """从 output_text 重建的 state."""
    section: str                            # SECTION_1/2/3/4/ANSWER
    visited: set[int]                       # think 段已 select 的客户
    answer_visited: set[int]                # 答案段 (</think> 之后) 出现的客户
    final_routes_written: bool              # think 段是否已开始 "4. **Final routes**:"
    in_step_line: bool                      # 当前 cursor 是否在 [Ri,s] 行内
    in_step_arrow: bool                     # 在 step 行且已写 → 但还没写决策 (用于 R4)
    decision_arrow_in_line: bool            # 当前 step 行内是否已现决策箭头 (用于 R3)
    select_trigger_now: bool                # 末尾刚好是 "→ select " (R1 触发)
    section_2_double_newline_tail: bool     # section 2 末尾 \n\n (R5 触发位置)


@dataclass
class MaskDecision:
    """state machine 返回的 mask 决策, 等待 token-level processor 应用."""
    # R1: 强 mask (除 unvisited customer first tokens + depot 0 mask 不允许)
    select_strict: bool = False
    select_allowed: list[int] = field(default_factory=list)   # 允许的 customer ids (1..n)
    # R2, R3, R4, R5: 各自禁的 token 语义
    forbid_think_close: bool = False        # </think>
    forbid_eos: bool = False                # <|im_end|> / EOS
    forbid_pipe: bool = False               # " |"  (R3)
    forbid_all_word: bool = False           # " all" (R4)
    forbid_verification_word: bool = False  # "Verification" (R5)
    # 是否在 mask 触发位置 (用于 IS skip)
    mask_hit: bool = False


def detect_section(text: str) -> str:
    """识别当前 section. 单调推进 (一旦后续 anchor 出现, 不回退)."""
    if THINK_CLOSE_LITERAL in text:
        return "ANSWER"
    if SECTION_4_ANCHOR in text:
        return "SECTION_4"
    if SECTION_3_ANCHOR in text:
        return "SECTION_3"
    if SECTION_2_ANCHOR in text:
        return "SECTION_2"
    return "SECTION_1"


def _parse_unvisited_set(content: str, n: int) -> set[int]:
    """Parse 'Unvisited: {1, 2, ..., 20}' 内的数字集合."""
    nums = re.findall(r"\d+", content)
    return {int(x) for x in nums if 1 <= int(x) <= n}


def build_state(text: str, n: int) -> CVRPState:
    """从完整 output_text 重建 state. O(L) 全量扫描.

    用 cache 优化在 processor 层做 (用 len(output_ids) 作 version), 这里保持 pure.
    """
    section = detect_section(text)

    # ── visited (think 段累积) ──────────────────────────────────────
    # 只看 section 2 范围内的 "→ select X"
    if section == "SECTION_1":
        # 还没进 section 2, visited 空
        visited: set[int] = set()
    else:
        # 切出 section 2 范围 (从 SECTION_2_ANCHOR 到 SECTION_3_ANCHOR 或 EOF)
        start = text.find(SECTION_2_ANCHOR)
        end_anchor_idx = text.find(SECTION_3_ANCHOR)
        if end_anchor_idx == -1:
            section_2_text = text[start:] if start >= 0 else ""
        else:
            section_2_text = text[start:end_anchor_idx]

        # Reflexion reset: 找最后一个 "Unvisited: {全集 n 个}", 之前的 select 不算
        # (允许模型偶尔重启, 但 SFT 中开头 "Unvisited: {1..n}" 也会触发, 行为是空集 reset, 无害)
        last_reset_end = -1
        for m in UNVISITED_RESET_RE.finditer(section_2_text):
            content = m.group(1)
            parsed = _parse_unvisited_set(content, n)
            if len(parsed) == n:
                last_reset_end = m.end()  # reset 之后的文本才累积 select
        # 累积 select X (从 reset 点之后)
        accum_text = section_2_text[last_reset_end:] if last_reset_end >= 0 else section_2_text
        visited = {
            int(m.group(1))
            for m in SELECT_ACCUM_RE.finditer(accum_text)
            if 1 <= int(m.group(1)) <= n
        }

    # ── answer_visited (答案段累积) ─────────────────────────────────
    answer_visited: set[int] = set()
    if section == "ANSWER":
        answer_part = text.split(THINK_CLOSE_LITERAL, 1)[1]
        answer_visited = {
            int(m.group(1))
            for m in ANSWER_ROUTE_TOKEN_RE.finditer(answer_part)
            if 1 <= int(m.group(1)) <= n
        }

    # ── 章节 4 是否已开始 (用于 R2 ` </think>` 控制) ───────────────
    final_routes_written = SECTION_4_ANCHOR in text

    # ── 当前是否在 step 行内 + 行内已现决策箭头? ───────────────────
    in_step_line = bool(STEP_LINE_RE.search(text))
    in_step_arrow = bool(STEP_LINE_WITH_ARROW_RE.search(text))
    if in_step_line:
        # 取最后一行(从最后 \n 到末尾)
        current_line = text.rsplit("\n", 1)[-1]
        decision_arrow_in_line = bool(DECISION_ARROW_RE.search(current_line))
    else:
        decision_arrow_in_line = False

    # ── R1 触发: 末尾刚好是 "→ select " ─────────────────────────────
    select_trigger_now = bool(SELECT_TRIGGER_RE.search(text))

    # ── R5 触发位置: section 2 末尾刚好 \n\n (即将开新章节) ──────────
    section_2_double_newline_tail = (
        section == "SECTION_2" and text.endswith("\n\n")
    )

    return CVRPState(
        section=section,
        visited=visited,
        answer_visited=answer_visited,
        final_routes_written=final_routes_written,
        in_step_line=in_step_line,
        in_step_arrow=in_step_arrow,
        decision_arrow_in_line=decision_arrow_in_line,
        select_trigger_now=select_trigger_now,
        section_2_double_newline_tail=section_2_double_newline_tail,
    )


def compute_mask(state: CVRPState, cfg: MaskConfig) -> MaskDecision:
    """根据 state + config 计算每条规则触发情况.

    应用到 logits 的工作由上层 processor 完成 (token id 映射).
    """
    n = cfg.n
    dec = MaskDecision()
    in_think = state.section != "ANSWER"

    # ── 规则 1: select 后强 mask (仅 section 2 内) ──────────────────
    if cfg.apply_select and state.section == "SECTION_2" and state.select_trigger_now:
        unvisited = set(range(1, n + 1)) - state.visited
        if len(unvisited) > 0:
            dec.select_strict = True
            dec.select_allowed = sorted(unvisited)
            dec.mask_hit = True
        # else: visited == n 仍触发 select (异常), fallback 不 mask 避免 NaN

    # ── 规则 2: 禁结束 ────────────────────────────────────────────
    if cfg.apply_block_end:
        if in_think:
            # think 段: visited<n 或 Section 4 未写完 → 禁 </think>; EOS 始终禁 (think 内)
            if len(state.visited) < n or not state.final_routes_written:
                dec.forbid_think_close = True
            dec.forbid_eos = True
        else:
            # 答案段: answer_visited<n 时禁 EOS
            if len(state.answer_visited) < n:
                dec.forbid_eos = True

    # ── 规则 3: visited==n 时禁 " |" (强制走 → all customers served) ─
    if (cfg.apply_visited_n
        and state.section == "SECTION_2"
        and len(state.visited) == n
        and state.in_step_line
        and not state.decision_arrow_in_line):
        dec.forbid_pipe = True
        dec.mask_hit = True

    # ── 规则 4: visited<n 时禁 " all" (强制 → select/no/remaining) ──
    if (cfg.apply_block_all
        and state.section == "SECTION_2"
        and len(state.visited) < n
        and state.in_step_arrow):       # 在 step 行内 → 后位置
        dec.forbid_all_word = True
        dec.mask_hit = True

    # ── 规则 5: visited<n 时禁 Verification 入口 ──────────────────
    if (cfg.apply_block_verif
        and state.section == "SECTION_2"
        and len(state.visited) < n):
        dec.forbid_verification_word = True
        # 注: 这条规则保守, 不必标 mask_hit (Verification 不常出现, IS 影响小)

    return dec


# ── 公开 API ────────────────────────────────────────────────────────
__all__ = [
    "MaskConfig",
    "CVRPState",
    "MaskDecision",
    "build_state",
    "compute_mask",
    "detect_section",
]
