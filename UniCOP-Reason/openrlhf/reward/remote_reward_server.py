"""
OpenRLHF 远程 reward HTTP 服务

OpenRLHF 0.10.2 的 GRPO trainer 通过 --reward.remote_url http://host:port/get_reward
调用本服务, 传入 prompt + completion 列表, 返回每条 completion 的 scalar reward.

OpenRLHF /get_reward 请求格式 (0.10.2):
    {"query": [prompt+completion 拼接文本],
     "prompts": [仅 prompt 文本],
     "labels": [label 字段, 本项目传 instance_id]}

    query  = hf_tokenizer.decode(observation_tokens, skip_special_tokens=False)
    prompts = 原始 prompt 字符串 (apply_chat_template 后)
    labels  = --data.label_key 指定字段的原始值

    返回格式:
    {"rewards": [float], "scores": [float], "extra_logs": {str: [float]}}

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
    conda activate /home/ntu/anaconda3/envs/zjh
    cd /home/ntu/lzj/UniCOP/UniCOP-Reason/openrlhf
    python reward/remote_reward_server.py \
        --problem_type tsp \
        --problem_size 10 \
        --port 5000
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
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
    STATE.pomo_prm.check_checkpoints(
        problem_types=[args.problem_type],
        problem_sizes=[args.problem_size],
    )
    print(f"[init] POMOPRM 加载完成 ({args.problem_type}{args.problem_size})")


# ── completion 提取 ───────────────────────────────────────────────

_INSTANCE_MARKER_RE = re.compile(r"\[\[instance_id:([^\]]+)\]\]")

_ASSISTANT_MARKERS = ["<|Assistant|>", "<|assistant|>", "<｜Assistant｜>"]


def _extract_completion(query: str, prompt: str) -> str:
    """从 query (prompt+completion 拼接) 中提取 completion 部分。

    OpenRLHF 的 query 是 tokenize → decode 后的全文, prompt 是原始文本,
    两者可能因 special token 编解码有微小差异, 所以先试精确前缀匹配,
    再 fallback 到 assistant marker 切分。
    """
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


# ── 核心 reward 计算 ───────────────────────────────────────────────

def compute_reward(completion: str, instance_id: str) -> float:
    """单条 completion 的聚合 scalar reward。"""
    if instance_id not in STATE.instances:
        return 0.0

    instance = STATE.instances[instance_id]

    terminal = compute_terminal_components(
        completion, instance, STATE.problem_type,
    )
    terminal_total = (
        terminal["parse"] + terminal["coverage"]
        + terminal["constraint"] + terminal["format"]
    )

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


@app.post("/get_reward")
async def get_reward(request: Request) -> Dict[str, Any]:
    """OpenRLHF 0.10.2 远程 reward 接口。

    请求体: {"query": [...], "prompts": [...], "labels": [...]}
    响应体: {"rewards": [...], "scores": [...], "extra_logs": {...}}
    """
    data = await request.json()
    queries: List[str] = data.get("query", [])
    prompts: List[str] = data.get("prompts", [])
    labels: List[str] = data.get("labels", [])

    n = len(queries)
    assert len(prompts) == n, \
        f"query/prompts 长度不一致: {n} vs {len(prompts)}"

    rewards: List[float] = []
    for i in range(n):
        completion = _extract_completion(queries[i], prompts[i])

        # instance_id: 优先从 labels 拿 (--data.label_key instance_id),
        # 其次从 prompt 文本 regex 提取
        instance_id = None
        if i < len(labels) and labels[i]:
            instance_id = labels[i]
        if not instance_id:
            instance_id = _extract_instance_id(prompts[i])

        if instance_id is None:
            rewards.append(0.0)
        else:
            rewards.append(compute_reward(completion, instance_id))

    return {
        "rewards": rewards,
        "scores": [max(0.0, min(1.0, r)) for r in rewards],
        "extra_logs": {
            "raw_rewards": rewards,
        },
    }


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
                        default="/home/ntu/lzj/POMO-Baseline/result")
    parser.add_argument("--pomo_baseline_dir", type=str,
                        default="/home/ntu/lzj/POMO-Baseline")
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
