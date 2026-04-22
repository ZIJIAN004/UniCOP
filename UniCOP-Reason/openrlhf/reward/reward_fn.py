"""
colocate 模式的本地 reward 函数 (不走 HTTP)

OpenRLHF 也支持 --reward_fn python_module:function 直接本地调用,
适合显存富余 / POMO 模型能和训练同机的场景。

本地模式省了 HTTP 序列化开销, 但要求 POMOPRM 占训练进程的 GPU.
8×3090 紧张时用 remote_reward_server.py 更稳.

用法 (如果将来想切到 colocate):
    --reward_fn openrlhf.reward.reward_fn:batch_reward_fn
"""

import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reward._shared import (
    POMOPRM,
    compute_terminal_components,
)


_INSTANCES_CACHE: dict | None = None
_POMO_PRM: POMOPRM | None = None
_PROBLEM_TYPE: str = os.environ.get("PROBLEM_TYPE", "tsp")
_PROBLEM_SIZE: int = int(os.environ.get("PROBLEM_SIZE", "10"))

_INSTANCE_MARKER_RE = re.compile(r"\[\[instance_id:([^\]]+)\]\]")


def _ensure_loaded():
    """首次调用时加载 instances + POMOPRM (colocate 训练每个 rank 都会调一次)。"""
    global _INSTANCES_CACHE, _POMO_PRM
    if _INSTANCES_CACHE is not None:
        return

    import json

    data_dir = PROJECT_ROOT / "data" / "processed"
    _INSTANCES_CACHE = {}
    for split in ["train", "test"]:
        inst_file = data_dir / f"{_PROBLEM_TYPE}{_PROBLEM_SIZE}_{split}_instances.json"
        if inst_file.exists():
            with open(inst_file, encoding="utf-8") as f:
                _INSTANCES_CACHE.update(json.load(f))

    _POMO_PRM = POMOPRM(
        pomo_ckpt_dir=os.environ.get(
            "POMO_CKPT_DIR", "/Data04/yangzhihan/lzj/POMO-Baseline/result"),
        pomo_baseline_dir=os.environ.get(
            "POMO_BASELINE_DIR", "/Data04/yangzhihan/lzj/POMO-Baseline"),
        device="cuda",
        pipd_ckpt_dir=os.environ.get("PIPD_CKPT_DIR"),
        pipd_dir=os.environ.get("PIPD_DIR"),
    )


def _extract_instance_id(prompt: str) -> str | None:
    m = _INSTANCE_MARKER_RE.search(prompt)
    return m.group(1) if m else None


def batch_reward_fn(prompts: list[str], responses: list[str]) -> list[float]:
    """
    OpenRLHF colocate 模式 reward fn 签名:
        def f(prompts: list[str], responses: list[str]) -> list[float]
    """
    _ensure_loaded()
    assert _INSTANCES_CACHE is not None and _POMO_PRM is not None

    rewards = []
    for prompt, completion in zip(prompts, responses):
        instance_id = _extract_instance_id(prompt)
        if instance_id is None or instance_id not in _INSTANCES_CACHE:
            rewards.append(0.0)
            continue

        instance = _INSTANCES_CACHE[instance_id]

        # terminal
        t = compute_terminal_components(completion, instance, _PROBLEM_TYPE)
        terminal_total = t["parse"] + t["coverage"] + t["constraint"] + t["format"]

        # PRM
        step_rew = _POMO_PRM.compute_step_rewards(
            completion=completion,
            instance=instance,
            problem_type=_PROBLEM_TYPE,
        )
        prm_total = 0.0
        if step_rew.n > 0:
            prm_total = (
                sum(step_rew.customer_rewards) + sum(step_rew.depot_rewards)
            ) / step_rew.n

        rewards.append(1.0 * terminal_total + 0.5 * prm_total)

    return rewards
