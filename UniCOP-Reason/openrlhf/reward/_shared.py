"""
桥接父目录的 POMO PRM 和 terminal reward 代码。

不复制代码,用 sys.path 引用,保证父目录改了 PRM 逻辑时本目录同步。

父目录真实 API:
    - pomo_prm.POMOPRM 类 (非模块级函数): 实例化 → 调用 compute_step_rewards
    - terminal_reward.compute_terminal_components(completion, instance, problem_type) -> dict

使用方式:
    from reward._shared import POMOPRM, compute_terminal_components
"""

import sys
from pathlib import Path

# 注入父目录到 sys.path (UniCOP/UniCOP-Reason/)
_PARENT = Path(__file__).resolve().parent.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# 从父目录 import
from pomo_prm import POMOPRM, StepRewards                  # noqa: E402
from terminal_reward import compute_terminal_components    # noqa: E402

__all__ = [
    "POMOPRM",
    "StepRewards",
    "compute_terminal_components",
    "parent_root",
]


def parent_root() -> Path:
    """返回父目录 (UniCOP/UniCOP-Reason) 绝对路径,供数据路径拼接。"""
    return _PARENT
