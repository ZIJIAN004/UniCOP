"""
colocate 模式的本地 reward 函数 (不走 HTTP)

OpenRLHF 0.10.2 支持 --reward.remote_url /path/to/reward_fn.py 直接加载本地函数,
适合显存富余 / POMO 模型能和训练同机的场景。

函数签名 (OpenRLHF 0.10.2 规范):
    def reward_func(queries, prompts, labels, **kwargs) -> dict
        queries: list[str]  完整文本 (prompt + completion 拼接)
        prompts: list[str]  原始 prompt 文本
        labels:  list[str]  --data.label_key 指定的字段 (本项目传 instance_id)
        return: {"rewards": [...], "scores": [...], "extra_logs": {...}}

本地模式省了 HTTP 序列化开销, 但要求 POMOPRM 占训练进程的 GPU.
8×3090 紧张时用 remote_reward_server.py 更稳.

用法:
    --reward.remote_url /home/ntu/lzj/UniCOP/UniCOP-Reason/openrlhf/reward/reward_fn.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Any

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
_ASSISTANT_MARKERS = ["<|Assistant|>", "<|assistant|>", "<｜Assistant｜>"]


def _ensure_loaded():
    """首次调用时加载 instances + POMOPRM."""
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
            "POMO_CKPT_DIR", "/home/ntu/lzj/POMO-Baseline/result"),
        pomo_baseline_dir=os.environ.get(
            "POMO_BASELINE_DIR", "/home/ntu/lzj/POMO-Baseline"),
        device="cuda",
        pipd_ckpt_dir=os.environ.get("PIPD_CKPT_DIR"),
        pipd_dir=os.environ.get("PIPD_DIR"),
    )


def _extract_completion(query: str, prompt: str) -> str:
    """从 query (prompt+completion) 中提取 completion。"""
    if query.startswith(prompt):
        return query[len(prompt):]
    for marker in _ASSISTANT_MARKERS:
        idx = query.rfind(marker)
        if idx >= 0:
            tail = query[idx + len(marker):]
            if tail.startswith("<think>\n"):
                tail = tail[len("<think>\n"):]
            return tail
    return query


def _extract_instance_id(text: str) -> str | None:
    m = _INSTANCE_MARKER_RE.search(text)
    return m.group(1) if m else None


def reward_func(queries: list[str], prompts: list[str],
                labels: list[str], **kwargs: Any) -> dict:
    """OpenRLHF 0.10.2 colocate 模式 reward function。"""
    _ensure_loaded()
    assert _INSTANCES_CACHE is not None and _POMO_PRM is not None

    rewards: list[float] = []
    for i, (query, prompt) in enumerate(zip(queries, prompts)):
        completion = _extract_completion(query, prompt)

        instance_id = None
        if i < len(labels) and labels[i]:
            instance_id = labels[i]
        if not instance_id:
            instance_id = _extract_instance_id(prompt)

        if instance_id is None or instance_id not in _INSTANCES_CACHE:
            rewards.append(0.0)
            continue

        instance = _INSTANCES_CACHE[instance_id]

        t = compute_terminal_components(completion, instance, _PROBLEM_TYPE)
        terminal_total = t["parse"] + t["coverage"] + t["constraint"] + t["format"]

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

        rewards.append(1.0 * terminal_total + 1.0 * prm_total)

    return {
        "rewards": rewards,
        "scores": [max(0.0, min(1.0, r)) for r in rewards],
        "extra_logs": {"raw_rewards": rewards},
    }
