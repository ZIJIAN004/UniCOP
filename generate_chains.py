"""
generate_chains.py
调用 Gemini 2.5 Pro 为 COP 问题生成思维链蒸馏数据。

功能：
  - 对每个 (problem_type, n) 组合生成 num_samples 条样本
  - 提取 Gemini 的 thinking 链和最终 answer，分别保存
  - 统计 thinking/answer/prompt 的 token 数
  - 支持断点续跑（已生成的 sample 自动跳过）
  - 输出 JSONL，每行一条样本，便于后续 SFT 使用

输出格式（每行）：
  {
    "id":              "tsp_n5_s42_i0",
    "problem_type":    "tsp",
    "n":               5,
    "sample_idx":      0,
    "prompt":          {"system": "...", "user": "..."},
    "thinking":        "...",        <- Gemini 推理链原文
    "answer":          "...",        <- Gemini 最终答案
    "thinking_tokens": 312,
    "answer_tokens":   48,
    "prompt_tokens":   195,
    "total_tokens":    555,
    "timestamp":       "2026-03-23T10:00:00"
  }

运行示例：
  python generate_chains.py --credentials /path/to/key.json --project my-gcp-project
  python generate_chains.py --credentials /path/to/key.json --project my-gcp-project \
      --problems tsp tsptw --sizes 5 10 20
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

# ── UniCOP-Reason 的 problems/ 路径 ───────────────────────────────────────────
# 默认假设两个项目同级：../UniCOP-Reason
_DEFAULT_UNICOP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "UniCOP-Reason")

# ── 默认测试矩阵 ──────────────────────────────────────────────────────────────
PROBLEM_TYPES = ["tsp", "tsptw", "tspdl", "cvrp", "vrptw", "cvrptw"]
NODE_SIZES    = [5, 10, 20, 50]
GEMINI_MODEL  = "gemini-2.5-pro"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def setup_problems_path(unicop_path: str):
    """把 UniCOP-Reason 加入 sys.path，使 problems/ 可以 import。"""
    path = os.path.abspath(unicop_path)
    if not os.path.isdir(os.path.join(path, "problems")):
        raise FileNotFoundError(f"找不到 problems/ 目录：{path}\n请通过 --unicop_path 指定正确路径")
    if path not in sys.path:
        sys.path.insert(0, path)


def extract_system_user(prompt: list[dict]) -> tuple[str, str]:
    """从 build_prompt() 返回的消息列表中提取 system / user 内容。"""
    system, user = "", ""
    for msg in prompt:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            user = msg["content"]
    return system, user


def instance_to_serializable(instance: dict) -> dict:
    """将 instance 中 numpy 数组转为 list，使其可被 json.dumps 序列化。"""
    out = {}
    for k, v in instance.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def load_existing_ids(output_path: str) -> set:
    """读取已有 JSONL，返回已完成的 sample id 集合（支持断点续跑）。"""
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
# Gemini 调用
# ─────────────────────────────────────────────────────────────────────────────

def build_client(credentials_path: str, project: str, location: str):
    """
    使用 GCP 服务账号 JSON key 文件构造 Vertex AI 客户端。
    与 Evolve_VLM_DPO 保持相同认证方式。
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(credentials_path)
    return genai.Client(vertexai=True, project=project, location=location)


def call_gemini(client, system: str, user: str, model: str) -> dict:
    """
    调用 Gemini，开启 include_thoughts=True，分离 thinking 链和最终答案。

    返回：
      {
        "thinking":        str,   # <think> 推理链原文
        "answer":          str,   # 最终答案
        "thinking_tokens": int | None,
        "answer_tokens":   int | None,
        "prompt_tokens":   int | None,
        "total_tokens":    int | None,
      }
    """
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system if system else None,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            temperature=1.0,
        ),
    )

    thinking_parts = []
    answer_parts   = []
    for part in response.candidates[0].content.parts:
        if getattr(part, "thought", False):
            thinking_parts.append(part.text)
        else:
            answer_parts.append(part.text)

    usage = response.usage_metadata
    return {
        "thinking":        "\n".join(thinking_parts),
        "answer":          "\n".join(answer_parts),
        "thinking_tokens": getattr(usage, "thoughts_token_count",    None),
        "answer_tokens":   getattr(usage, "candidates_token_count",  None),
        "prompt_tokens":   getattr(usage, "prompt_token_count",      None),
        "total_tokens":    getattr(usage, "total_token_count",       None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 统计打印
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(output_path: str):
    """读取 JSONL，按 (problem_type, n) 打印 thinking token 统计。"""
    stats = defaultdict(list)
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("thinking_tokens") is not None:
                stats[(r["problem_type"], r["n"])].append(r["thinking_tokens"])

    if not stats:
        print("（暂无数据）")
        return

    col = 14
    print(f"\n{'Problem':<10} {'n':>5}  {'count':>6}  {'mean_think':>{col}}  {'max_think':>{col}}  {'min_think':>{col}}")
    print("-" * (10 + 5 + 6 + col * 3 + 10))
    for (pt, n), toks in sorted(stats.items()):
        arr = np.array(toks)
        print(f"{pt:<10} {n:>5}  {len(toks):>6}  {arr.mean():>{col}.0f}  {arr.max():>{col}.0f}  {arr.min():>{col}.0f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="用 Gemini 2.5 Pro 生成 COP 推理链蒸馏数据")
    parser.add_argument("--credentials",  type=str,
                        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
                        help="GCP 服务账号 JSON key 文件路径（也可通过环境变量 GOOGLE_APPLICATION_CREDENTIALS 设置）")
    parser.add_argument("--project",     type=str,
                        default=os.environ.get("GCP_PROJECT", ""),
                        help="GCP 项目 ID，如 keen-oasis-489308-m8")
    parser.add_argument("--location",    type=str,  default="us-central1",
                        help="Vertex AI 区域（默认 us-central1）")
    parser.add_argument("--model",       type=str,  default=GEMINI_MODEL,
                        help=f"Gemini 模型名称（默认 {GEMINI_MODEL}）")
    parser.add_argument("--unicop_path", type=str,  default=_DEFAULT_UNICOP_PATH,
                        help="UniCOP-Reason 项目路径（默认 ../UniCOP-Reason）")
    parser.add_argument("--problems",    type=str,  nargs="+", default=PROBLEM_TYPES,
                        choices=PROBLEM_TYPES,
                        help="要生成的问题类型（默认全部）")
    parser.add_argument("--sizes",       type=int,  nargs="+", default=NODE_SIZES,
                        help="节点规模列表（默认 5 10 20 50）")
    parser.add_argument("--num_samples", type=int,  default=20,
                        help="每个 (problem, n) 组合的样本数（默认 20）")
    parser.add_argument("--seed",        type=int,  default=42,
                        help="基础随机种子（每个 n 会在此基础上偏移）")
    parser.add_argument("--output",      type=str,  default="data/chains.jsonl",
                        help="输出 JSONL 文件路径（默认 data/chains.jsonl）")
    parser.add_argument("--sleep",       type=float, default=2.0,
                        help="每次 API 调用后的等待秒数，避免限速（默认 2s）")
    parser.add_argument("--stats_only",  action="store_true",
                        help="只打印已有数据的统计，不发起新 API 调用")
    args = parser.parse_args()

    # ── 仅统计模式 ────────────────────────────────────────────────────────────
    if args.stats_only:
        print_stats(args.output)
        return

    # ── 检查 Vertex AI 认证参数 ───────────────────────────────────────────────
    if not args.credentials:
        raise ValueError(
            "请通过 --credentials 或环境变量 GOOGLE_APPLICATION_CREDENTIALS 指定 GCP JSON key 文件路径"
        )
    if not os.path.isfile(args.credentials):
        raise FileNotFoundError(f"找不到 credentials 文件：{args.credentials}")
    if not args.project:
        raise ValueError(
            "请通过 --project 或环境变量 GCP_PROJECT 指定 GCP 项目 ID"
        )

    # ── 加载 problems ─────────────────────────────────────────────────────────
    setup_problems_path(args.unicop_path)
    from problems import get_problem  # noqa: E402（延迟 import）

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    client       = build_client(args.credentials, args.project, args.location)
    existing_ids = load_existing_ids(args.output)

    total    = len(args.problems) * len(args.sizes) * args.num_samples
    n_done   = len(existing_ids)
    n_remain = total - n_done
    print(f"\n{'='*60}")
    print(f"  Gemini COP 推理链数据生成")
    print(f"  模型:    {args.model}")
    print(f"  项目:    {args.project}  ({args.location})")
    print(f"  计划:    {total} 条  |  已有: {n_done} 条  |  剩余: {n_remain} 条")
    print(f"  输出:    {args.output}")
    print(f"{'='*60}\n")

    combo_total = len(args.problems) * len(args.sizes)
    combo_idx   = 0

    with open(args.output, "a", encoding="utf-8") as fout:
        for pt in args.problems:
            problem = get_problem(pt)
            for n in args.sizes:
                combo_idx += 1
                rng = np.random.default_rng(seed=args.seed + n)
                print(f"[{combo_idx}/{combo_total}] {pt}  n={n}")

                for i in range(args.num_samples):
                    sample_id = f"{pt}_n{n}_s{args.seed}_i{i}"
                    if sample_id in existing_ids:
                        print(f"    [skip] {sample_id}")
                        continue

                    instance        = problem.generate_instance(n, rng)
                    prompt          = problem.build_prompt(instance)
                    system, user    = extract_system_user(prompt)

                    print(f"    #{i+1:>2}/{args.num_samples} {sample_id} ... ", end="", flush=True)
                    t0 = time.time()
                    try:
                        result = call_gemini(client, system, user, args.model)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        time.sleep(args.sleep * 5)
                        continue
                    elapsed = time.time() - t0

                    print(
                        f"think={result['thinking_tokens']} tok  "
                        f"ans={result['answer_tokens']} tok  "
                        f"({elapsed:.1f}s)"
                    )

                    record = {
                        "id":              sample_id,
                        "problem_type":    pt,
                        "n":               n,
                        "sample_idx":      i,
                        "prompt":          {"system": system, "user": user},
                        "thinking":        result["thinking"],
                        "answer":          result["answer"],
                        "thinking_tokens": result["thinking_tokens"],
                        "answer_tokens":   result["answer_tokens"],
                        "prompt_tokens":   result["prompt_tokens"],
                        "total_tokens":    result["total_tokens"],
                        "timestamp":       datetime.now().isoformat(),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()

                    time.sleep(args.sleep)

                print()

    print("全部完成！\n")
    print_stats(args.output)


if __name__ == "__main__":
    main()
