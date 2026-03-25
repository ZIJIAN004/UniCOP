"""
generate_chains.py
后验推理蒸馏数据生成脚本：先用 LKH 求近最优解，再让 Gemini 给出简短推理链。

Pipeline：
  1. 生成 COP 实例
  2. LKH 求解 → 近最优答案
  3. 构建后验推理 prompt：告知 Gemini 答案，要求简短解释推理
  4. Gemini 输出 <think>...</think> + 答案
  5. 保存数据

输出格式（JSONL，每行一条）：
  {
    "id":            "tsp_n20_s42_i0",
    "problem_type":  "tsp",
    "n":             20,
    "sample_idx":    0,
    "prompt":        {"system": "...", "user": "..."},  <- 发给 Gemini 的后验推理 prompt
    "lkh_answer":    "Route: 0 -> 3 -> ...",            <- LKH 原始答案（参考/校验用）
    "output":        "<think>...</think>\nRoute: ...",  <- Gemini 完整输出，直接用于 SFT
    "output_tokens": 360,
    "prompt_tokens": 195,
    "total_tokens":  555,
    "timestamp":     "2026-03-25T10:00:00"
  }

运行示例：
  python generate_chains.py \
      --credentials /path/to/key.json --project my-gcp-project \
      --lkh_bin /path/to/LKH --lkh3_bin /path/to/LKH3
  python generate_chains.py ... --problems tsp --sizes 5 10 --num_samples 5
  python generate_chains.py --stats_only
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from google import genai
from google.genai import types

from lkh_solver import solve as lkh_solve, LKH_BIN, LKH3_BIN

# ── UniCOP-Reason 的 problems/ 路径 ───────────────────────────────────────────
_DEFAULT_UNICOP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "UniCOP-Reason")

# ── 默认测试矩阵 ──────────────────────────────────────────────────────────────
PROBLEM_TYPES = ["tsp", "tsptw", "tspdl", "cvrp", "vrptw", "cvrptw"]
NODE_SIZES    = [5, 10, 20, 50]
GEMINI_MODEL  = "gemini-2.5-pro"

# ── 后验推理的 system prompt 后缀（追加到各问题原有 system prompt 之后）────────
_POSTHOC_SUFFIX = (
    "\n\nYou are given the near-optimal solution. "
    "Do NOT search for a new solution. "
    "First write a brief reasoning in <think>...</think> explaining the key properties "
    "that make this solution good (spatial structure, constraint order, route efficiency, etc.). "
    "Keep the <think> block concise (a few hundred words at most). "
    "Then copy the solution exactly in the required output format."
)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def setup_problems_path(unicop_path: str):
    path = os.path.abspath(unicop_path)
    if not os.path.isdir(os.path.join(path, "problems")):
        raise FileNotFoundError(f"找不到 problems/ 目录：{path}\n请通过 --unicop_path 指定正确路径")
    if path not in sys.path:
        sys.path.insert(0, path)


def build_posthoc_prompt(lkh_answer: str, orig_prompt: list[dict]) -> dict:
    """
    构建后验推理 prompt。
    - system：原始规则 + 后验推理指令
    - user：原始问题描述 + LKH 答案
    返回 {"system": str, "user": str}
    """
    system, user = "", ""
    for msg in orig_prompt:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            user = msg["content"]

    system_posthoc = system + _POSTHOC_SUFFIX

    user_posthoc = (
        user
        + f"\n\nThe near-optimal solution is:\n{lkh_answer}"
        + "\n\nNow think briefly about why this solution is good, "
          "then output it in the required format."
    )

    return {"system": system_posthoc, "user": user_posthoc}


def _answer_has_content(answer: str, problem_type: str) -> bool:
    """粗校验：答案中是否包含期望的关键词。"""
    if not answer or not answer.strip():
        return False
    if problem_type in ("tsp", "tsptw", "tspdl"):
        return "route:" in answer.lower() or "0 ->" in answer
    else:
        return "route" in answer.lower() and "0 ->" in answer


def load_existing_ids(output_path: str) -> set:
    ids = set()
    if not os.path.exists(output_path):
        return ids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI 客户端 & Gemini 调用
# ─────────────────────────────────────────────────────────────────────────────

def build_client(credentials_path: str, project: str, location: str):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(credentials_path)
    return genai.Client(vertexai=True, project=project, location=location)


def call_gemini(client, system: str, user: str, model: str) -> dict:
    """
    调用 Gemini（不开启 thinking mode）。
    Gemini 的完整可见输出即为 SFT 训练目标：
      <think>...</think>
      [answer in required format]
    """
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system if system else None,
            temperature=1.0,
        ),
    )

    full_output = "".join(
        part.text for part in response.candidates[0].content.parts
        if hasattr(part, "text") and part.text
    )
    usage = response.usage_metadata
    return {
        "output":          full_output,                                   # 完整输出，直接用于 SFT
        "output_tokens":   getattr(usage, "candidates_token_count", None),
        "prompt_tokens":   getattr(usage, "prompt_token_count",     None),
        "total_tokens":    getattr(usage, "total_token_count",      None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 统计打印
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(output_path: str):
    if not os.path.exists(output_path):
        print(f"文件不存在：{output_path}")
        return

    stats    = defaultdict(list)
    lkh_fail = defaultdict(int)
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["problem_type"], r["n"])
            if r.get("output_tokens") is not None:
                stats[key].append(r["output_tokens"])
            if not r.get("lkh_answer"):
                lkh_fail[key] += 1

    if not stats:
        print("（暂无数据）")
        return

    col = 14
    print(f"\n{'Problem':<10} {'n':>5}  {'count':>6}  {'mean_output':>{col}}  "
          f"{'max_output':>{col}}  {'lkh_fail':>8}")
    print("-" * 60)
    for (pt, n), toks in sorted(stats.items()):
        arr = np.array(toks)
        print(f"{pt:<10} {n:>5}  {len(toks):>6}  {arr.mean():>{col}.0f}  "
              f"{arr.max():>{col}.0f}  {lkh_fail[(pt,n)]:>8}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="后验推理蒸馏数据生成（LKH求解 + Gemini解释）")

    # Vertex AI 认证
    parser.add_argument("--credentials", type=str,
                        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
                        help="GCP 服务账号 JSON key 文件路径")
    parser.add_argument("--project",     type=str,
                        default=os.environ.get("GCP_PROJECT", ""),
                        help="GCP 项目 ID，如 keen-oasis-489308-m8")
    parser.add_argument("--location",    type=str, default="us-central1")

    # LKH
    parser.add_argument("--lkh_bin",  type=str, default=os.environ.get("LKH_BIN",  LKH_BIN),
                        help="LKH-2 二进制路径（处理 TSP / TSPTW）")
    parser.add_argument("--lkh3_bin", type=str, default=os.environ.get("LKH3_BIN", LKH3_BIN),
                        help="LKH-3 二进制路径（处理 CVRP / VRPTW / CVRPTW）")
    parser.add_argument("--lkh_runs",    type=int,   default=1,
                        help="LKH 每个实例的运行次数（默认 1，越大质量越高但越慢）")
    parser.add_argument("--lkh_timeout", type=int,   default=60,
                        help="LKH 单次求解超时秒数（默认 60）")

    # Gemini
    parser.add_argument("--model", type=str, default=GEMINI_MODEL)

    # 数据生成
    parser.add_argument("--unicop_path", type=str,  default=_DEFAULT_UNICOP_PATH)
    parser.add_argument("--problems",    type=str,  nargs="+", default=PROBLEM_TYPES,
                        choices=PROBLEM_TYPES)
    parser.add_argument("--sizes",       type=int,  nargs="+", default=NODE_SIZES)
    parser.add_argument("--num_samples", type=int,  default=50,
                        help="每个 (problem, n) 组合的样本数（默认 50）")
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--output",      type=str,  default="data/chains.jsonl")
    parser.add_argument("--sleep",       type=float, default=2.0,
                        help="每次 Gemini API 调用后的等待秒数（防限速）")
    parser.add_argument("--stats_only",  action="store_true",
                        help="只打印已有数据的统计，不发起新请求")
    args = parser.parse_args()

    if args.stats_only:
        print_stats(args.output)
        return

    # 参数检查
    if not args.credentials:
        raise ValueError("请通过 --credentials 或 GOOGLE_APPLICATION_CREDENTIALS 指定 GCP JSON key 路径")
    if not os.path.isfile(args.credentials):
        raise FileNotFoundError(f"找不到 credentials 文件：{args.credentials}")
    if not args.project:
        raise ValueError("请通过 --project 或 GCP_PROJECT 指定 GCP 项目 ID")

    setup_problems_path(args.unicop_path)
    from problems import get_problem  # noqa: E402

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    client       = build_client(args.credentials, args.project, args.location)
    existing_ids = load_existing_ids(args.output)

    total    = len(args.problems) * len(args.sizes) * args.num_samples
    n_done   = len(existing_ids)
    print(f"\n{'='*60}")
    print(f"  后验推理蒸馏数据生成（LKH + Gemini）")
    print(f"  模型:    {args.model}")
    print(f"  项目:    {args.project}  ({args.location})")
    print(f"  计划:    {total} 条  |  已有: {n_done} 条  |  剩余: {total - n_done} 条")
    print(f"  输出:    {args.output}")
    print(f"{'='*60}\n")

    combo_total = len(args.problems) * len(args.sizes)
    combo_idx   = 0

    with open(args.output, "a", encoding="utf-8") as fout:
        for pt in args.problems:
            problem = get_problem(pt)
            for n in args.sizes:
                combo_idx += 1
                pt_offset = sum(ord(c) for c in pt)
                rng = np.random.default_rng(seed=args.seed + n + pt_offset)
                print(f"[{combo_idx}/{combo_total}] {pt}  n={n}")

                for i in range(args.num_samples):
                    sample_id = f"{pt}_n{n}_s{args.seed}_i{i}"

                    # 无论是否跳过都要生成实例，保证 rng 状态与首次运行一致
                    instance = problem.generate_instance(n, rng)

                    if sample_id in existing_ids:
                        print(f"    [skip] {sample_id}")
                        continue

                    # ── Step 2: LKH 求解 ──────────────────────────────────
                    print(f"    #{i+1:>2}/{args.num_samples} {sample_id}  LKH...", end=" ", flush=True)
                    lkh_answer = lkh_solve(
                        pt, instance,
                        lkh_bin=args.lkh_bin, lkh3_bin=args.lkh3_bin,
                        runs=args.lkh_runs, seed=args.seed, timeout=args.lkh_timeout
                    )
                    if lkh_answer is None:
                        print("LKH FAILED, skip")
                        continue
                    print("ok →", end=" ", flush=True)

                    # ── Step 3: 构建后验推理 prompt ───────────────────────
                    orig_prompt = problem.build_prompt(instance)
                    prompt_dict = build_posthoc_prompt(lkh_answer, orig_prompt)

                    # ── Step 4: Gemini 生成后验推理输出 ──────────────────
                    t0 = time.time()
                    try:
                        result = call_gemini(
                            client,
                            prompt_dict["system"],
                            prompt_dict["user"],
                            args.model,
                        )
                    except Exception as e:
                        print(f"Gemini ERROR: {e}")
                        time.sleep(args.sleep * 5)
                        continue
                    elapsed = time.time() - t0

                    # ── Step 5: 校验输出包含答案 ──────────────────────────
                    answer_ok = _answer_has_content(result["output"], pt)
                    status = "ok" if answer_ok else "NO_ANSWER"
                    print(
                        f"output={result['output_tokens']} tok  "
                        f"[{status}]  ({elapsed:.1f}s)"
                    )
                    if not answer_ok:
                        time.sleep(args.sleep)
                        continue

                    # ── Step 6: 保存 ──────────────────────────────────────
                    record = {
                        "id":            sample_id,
                        "problem_type":  pt,
                        "n":             n,
                        "sample_idx":    i,
                        "prompt":        prompt_dict,    # 后验推理 prompt（含 LKH 答案）
                        "lkh_answer":    lkh_answer,     # LKH 原始答案（供参考/校验）
                        "output":        result["output"],  # <think>...</think>\n[answer]，直接用于 SFT
                        "output_tokens": result["output_tokens"],
                        "prompt_tokens": result["prompt_tokens"],
                        "total_tokens":  result["total_tokens"],
                        "timestamp":     datetime.now().isoformat(),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    time.sleep(args.sleep)

                print()

    print("全部完成！\n")
    print_stats(args.output)


if __name__ == "__main__":
    main()
