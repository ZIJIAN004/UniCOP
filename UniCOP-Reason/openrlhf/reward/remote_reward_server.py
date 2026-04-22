"""
OpenRLHF 远程 reward HTTP 服务

OpenRLHF 的 GRPO trainer 通过 --remote_rm_url http://host:port/get_reward 调用本服务,
传入 prompt + completion 列表, 返回每条 completion 的 scalar reward.

复用父目录实现:
    - terminal_reward.compute_terminal_components  (4 维 scalar 分量)
    - pomo_prm.POMOPRM 类 + compute_step_rewards 方法  (step-level, 这里聚合)

聚合策略 (同 completion 内):
    total = w_terminal * (parse + coverage + constraint + format)
          + w_prm * (sum(customer_rewards) + sum(depot_rewards)) / max(n, 1)

默认权重 w_terminal=1.0, w_prm=1.0 (与父目录 config.py terminal_alpha/prm_beta 对齐).

# TODO: step-level
# ------------------------------------------------------------
# 当前是 completion 级聚合, 丢失了 per-token advantage 信号.
# 要做真·step-level 需 fork OpenRLHF/openrlhf/trainer/ray/ppo_actor.py
# 的 ExperienceMaker.make_experience_list, 在 advantage 计算前
# 把 step_rewards.customer/depot_rewards 按 token 位置叠加进 reward 张量.
# 参考父目录 grpo_prm_trainer.py 中的 compute_advantages.
# ------------------------------------------------------------

用法 (在服务器上):
    conda activate /Data04/yangzhihan/envs/openrlhf_env
    cd /Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason/openrlhf
    python reward/remote_reward_server.py \
        --problem_type tsp \
        --problem_size 10 \
        --port 5000
    # POMO 路径用 argparse default (见 main() 末尾),无需每次传
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # openrlhf/
sys.path.insert(0, str(PROJECT_ROOT))

from reward._shared import (
    POMOPRM,
    compute_terminal_components,
)


# ── 全局状态 ─────────────────────────────────────────────────────────

class State:
    problem_type: str = "tsp"
    problem_size: int = 10
    instances: dict = {}               # instance_id -> instance dict
    pomo_prm: POMOPRM | None = None
    w_terminal: float = 1.0
    w_prm: float = 1.0


STATE = State()


# ── FastAPI 模型 ────────────────────────────────────────────────────

class RewardRequest(BaseModel):
    query: List[str]                   # prompt 列表 (带 [[instance_id:xxx]] 标记)
    response: List[str]                # completion 列表


class RewardResponse(BaseModel):
    rewards: List[float]


# ── 初始化 ──────────────────────────────────────────────────────────

def init_instances(args):
    """加载训练 + 测试实例库 (prepare_dataset.py 已生成)。"""
    data_dir = PROJECT_ROOT / "data" / "processed"
    for split in ["train", "test"]:
        inst_file = data_dir / f"{args.problem_type}{args.problem_size}_{split}_instances.json"
        if inst_file.exists():
            with open(inst_file, encoding="utf-8") as f:
                STATE.instances.update(json.load(f))
            print(f"[init] loaded {inst_file}")
        else:
            print(f"[init] WARNING: {inst_file} not found")

    print(f"[init] 共 {len(STATE.instances)} 条 instance")


def init_pomo(args):
    """初始化 POMOPRM (PRM 核心)。"""
    STATE.pomo_prm = POMOPRM(
        pomo_ckpt_dir=args.pomo_ckpt_dir,
        pomo_baseline_dir=args.pomo_baseline_dir,
        device="cuda",
        pipd_ckpt_dir=args.pipd_ckpt_dir,
        pipd_dir=args.pipd_dir,
    )
    # 预校验 ckpt 存在性,失败早抛
    STATE.pomo_prm.check_checkpoints(
        problem_types=[args.problem_type],
        problem_sizes=[args.problem_size],
    )
    print(f"[init] POMOPRM 加载完成 ({args.problem_type}{args.problem_size})")


# ── 核心 reward 计算 ───────────────────────────────────────────────

_INSTANCE_MARKER_RE = re.compile(r"\[\[instance_id:([^\]]+)\]\]")


def extract_instance_id(prompt: str) -> str | None:
    """从 prompt 文本里解出 instance_id (prepare_dataset.py 嵌入)."""
    m = _INSTANCE_MARKER_RE.search(prompt)
    return m.group(1) if m else None


def compute_reward(prompt: str, completion: str) -> float:
    """单条 (prompt, completion) 的聚合 scalar reward。"""
    instance_id = extract_instance_id(prompt)
    if instance_id is None or instance_id not in STATE.instances:
        return 0.0

    instance = STATE.instances[instance_id]

    # Terminal (4 维求和)
    terminal = compute_terminal_components(
        completion, instance, STATE.problem_type,
    )
    terminal_total = (
        terminal["parse"] + terminal["coverage"]
        + terminal["constraint"] + terminal["format"]
    )

    # PRM (step-level → 聚合 scalar)
    prm_total = 0.0
    if STATE.pomo_prm is not None:
        step_rew = STATE.pomo_prm.compute_step_rewards(
            completion=completion,
            instance=instance,
            problem_type=STATE.problem_type,
        )
        if step_rew.n > 0:
            prm_total = (
                sum(step_rew.customer_rewards) + sum(step_rew.depot_rewards)
            ) / step_rew.n

    return STATE.w_terminal * terminal_total + STATE.w_prm * prm_total


# ── FastAPI app ────────────────────────────────────────────────────

app = FastAPI()


@app.post("/get_reward", response_model=RewardResponse)
def get_reward(req: RewardRequest) -> RewardResponse:
    assert len(req.query) == len(req.response), \
        f"query/response 长度不一致: {len(req.query)} vs {len(req.response)}"
    rewards = [compute_reward(q, r) for q, r in zip(req.query, req.response)]
    return RewardResponse(rewards=rewards)


@app.get("/health")
def health():
    return {
        "ok": True,
        "n_instances": len(STATE.instances),
        "pomo_loaded": STATE.pomo_prm is not None,
        "problem": f"{STATE.problem_type}{STATE.problem_size}",
    }


# ── main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", type=str, required=True)
    parser.add_argument("--problem_size", type=int, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--pomo_ckpt_dir", type=str,
                        default="/Data04/yangzhihan/lzj/POMO-Baseline/result")
    parser.add_argument("--pomo_baseline_dir", type=str,
                        default="/Data04/yangzhihan/lzj/POMO-Baseline")
    parser.add_argument("--pipd_ckpt_dir", type=str, default=None,
                        help="TSPTW 用 PIP-D ckpt 根目录 (仅 TSPTW 需要)")
    parser.add_argument("--pipd_dir", type=str, default=None,
                        help="TSPTW 用 PIP-D 代码目录 (仅 TSPTW 需要)")
    parser.add_argument("--skip_pomo", action="store_true",
                        help="不加载 POMO, 只算 terminal reward (调试用)")
    parser.add_argument("--w_terminal", type=float, default=1.0)
    parser.add_argument("--w_prm", type=float, default=1.0)
    args = parser.parse_args()

    STATE.problem_type = args.problem_type
    STATE.problem_size = args.problem_size
    STATE.w_terminal = args.w_terminal
    STATE.w_prm = args.w_prm

    init_instances(args)
    if not args.skip_pomo:
        init_pomo(args)

    print(f"[run] 启动 reward server @ {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
