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
      --lkh_bin /path/to/LKH3
  python generate_chains.py ... --problems tsp --sizes 5 10 --num_samples 5
  python generate_chains.py --stats_only
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from google import genai
from google.genai import types

from lkh_solver import solve as lkh_solve, LKH_BIN

# ── UniCOP-Reason 的 problems/ 路径 ───────────────────────────────────────────
_DEFAULT_UNICOP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "UniCOP-Reason")

# ── 默认测试矩阵 ──────────────────────────────────────────────────────────────
PROBLEM_TYPES = ["tsp", "cvrp", "tsptw", "vrptw"]
NODE_SIZES    = [20, 50, 100]
GEMINI_MODEL  = "gemini-2.5-pro"

# ── 后验推理的 system prompt 后缀（追加到各问题原有 system prompt 之后）────────
_POSTHOC_SUFFIX = (
    "\n\nYour output MUST start with <think> and follow this exact structure:\n\n"
    "<think>\n[your reasoning here]\n</think>\n[solution in required format]\n\n"
    "Rules:\n"
    "1. Your FIRST token MUST be '<think>'. Do NOT output anything before <think>.\n"
    "2. In <think>, show your step-by-step decision process for "
    "constructing the route from scratch. At each step, state where you are, "
    "which nearby nodes are candidates, and why you pick the next one "
    "(e.g. nearest distance, tightest time window, capacity constraint). "
    "Write as if you are solving this problem yourself for the first time.\n"
    "3. Keep <think> concise (a few hundred words at most). "
    "Do NOT mention that a solution was provided or given to you. "
    "Do NOT describe your task as 'reconstructing', 'explaining', or 'justifying' a solution. "
    "You are solving this problem from scratch — your reasoning should read as original problem-solving, "
    "not as post-hoc analysis of a known answer.\n"
    "4. After </think>, output the solution exactly in the required format.\n"
    "5. Do NOT output the solution before <think>. The solution ONLY appears after </think>."
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
        + f"\n\nTarget solution (you MUST output exactly this solution after </think>, but do NOT reveal it was given to you):\n{lkh_answer}"
        + "\n\nStart your response with <think> immediately. "
          "Solve this problem step by step, then output the target solution after </think>."
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


def _output_quality_check(output: str) -> tuple[bool, str]:
    """
    检查输出质量：
    1. 必须有 <think>...</think> 标签
    2. think 内不能复制路线（0 -> 出现 <= 1 次）
    返回 (通过, 原因)
    """
    if "<think>" not in output or "</think>" not in output:
        return False, "NO_THINK_TAGS"

    think_start = output.index("<think>") + 7
    think_end = output.index("</think>")
    think_content = output[think_start:think_end]

    if think_content.count("0 ->") > 1:
        return False, "ROUTE_IN_THINK"

    return True, "ok"


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


def count_valid_samples(output_path: str, max_output_tokens: int) -> dict:
    """
    统计已有数据中每个 (problem_type, n) 组合的合格样本数。
    合格条件：output_tokens <= max_output_tokens。
    返回 {(problem_type, n): count}
    """
    counts = defaultdict(int)
    if not os.path.exists(output_path):
        return counts
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("output_tokens") is not None and r["output_tokens"] <= max_output_tokens:
                    counts[(r["problem_type"], r["n"])] += 1
            except Exception:
                pass
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI 客户端 & Gemini 调用
# ─────────────────────────────────────────────────────────────────────────────

def build_client(credentials_path: str, project: str, location: str):
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(
        os.path.abspath(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(vertexai=True, project=project, location=location, credentials=creds)


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
    parser.add_argument("--lkh_bin",     type=str, default=os.environ.get("LKH_BIN", LKH_BIN),
                        help="LKH-3 二进制路径（处理 TSP 和 CVRP）")
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
    parser.add_argument("--sleep",       type=float, default=0.5,
                        help="每次 Gemini API 调用后的等待秒数（防限速）")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Gemini 并发请求数（默认 8）")
    parser.add_argument("--max_output_tokens", type=int, default=4096,
                        help="合格样本的最大 output token 数，超过则丢弃（默认 4096）")
    parser.add_argument("--stats_only",  action="store_true",
                        help="只打印已有数据的统计，不发起新请求")
    args = parser.parse_args()

    if args.stats_only:
        print_stats(args.output)
        return

    try:
        _run(args)
    except KeyboardInterrupt:
        print("\n用户中断")
        _notify_serverchan("generate_chains 被手动中断")
    except Exception:
        import traceback
        _notify_serverchan("generate_chains 异常退出", traceback.format_exc()[-400:])
        raise
    else:
        _notify_serverchan("generate_chains 正常完成",
                           f"output: {args.output}, num_samples: {args.num_samples}")


def _run(args):
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

    # 清理已有文件中超长样本（output_tokens > max_output_tokens）
    if os.path.exists(args.output):
        kept_lines = []
        purged = 0
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    r = json.loads(stripped)
                    ot = r.get("output_tokens")
                    if ot is not None and ot > args.max_output_tokens:
                        purged += 1
                        continue
                except json.JSONDecodeError:
                    pass
                kept_lines.append(stripped)
        if purged > 0:
            with open(args.output, "w", encoding="utf-8") as f:
                for kl in kept_lines:
                    f.write(kl + "\n")
            print(f"  清理超长样本: 删除 {purged} 条 (output_tokens > {args.max_output_tokens})")

    client       = build_client(args.credentials, args.project, args.location)
    existing_ids = load_existing_ids(args.output)
    valid_counts = count_valid_samples(args.output, args.max_output_tokens)

    target_per_combo = args.num_samples
    combo_total = len(args.problems) * len(args.sizes)

    # 统计需要补充的组合
    combos_need = []
    for pt in args.problems:
        for n in args.sizes:
            current = valid_counts.get((pt, n), 0)
            if current < target_per_combo:
                combos_need.append((pt, n, target_per_combo - current))

    total_need = sum(c[2] for c in combos_need)
    total_valid = sum(valid_counts.values())

    print(f"\n{'='*60}")
    print(f"  后验推理蒸馏数据生成（LKH + Gemini）")
    print(f"  模型:    {args.model}")
    print(f"  项目:    {args.project}  ({args.location})")
    print(f"  目标:    每组合 {target_per_combo} 条合格样本（output_tokens <= {args.max_output_tokens}）")
    print(f"  现有合格: {total_valid} 条  |  需补充: {total_need} 条  |  涉及 {len(combos_need)} 个组合")
    print(f"  输出:    {args.output}")
    print(f"{'='*60}")

    # 打印各组合现有合格数
    print(f"\n{'组合':<15} {'合格':>5} {'目标':>5} {'差额':>5}")
    print("-" * 35)
    for pt in args.problems:
        for n in args.sizes:
            current = valid_counts.get((pt, n), 0)
            gap = max(0, target_per_combo - current)
            marker = " ← 需补充" if gap > 0 else ""
            print(f"{pt}_n{n:<7} {current:>5} {target_per_combo:>5} {gap:>5}{marker}")
    print()

    if total_need == 0:
        print("所有组合已达标，无需生成！\n")
        print_stats(args.output)
        return

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    write_lock = threading.Lock()

    def _call_gemini_with_retry(client, system, user, model, sleep):
        """带限流重试的 Gemini 调用（线程安全）"""
        max_retries = 10
        base_wait = 30
        max_wait_time = 300
        for attempt in range(max_retries):
            try:
                return call_gemini(client, system, user, model)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in [
                    "429", "rate limit", "resource exhausted",
                    "quota", "too many requests",
                ])
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = min(base_wait * (2 ** attempt), max_wait_time)
                    sleep_time += random.uniform(0, sleep_time * 0.3)
                    print(f"  RATE LIMIT, 等待 {sleep_time:.0f}s 重试 ({attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    print(f"  Gemini ERROR: {e}")
                    return None
        return None

    def _process_single(task_info):
        """处理单条数据：LKH 求解 + Gemini 调用 + 校验 + 长度检查"""
        pt, n, i, sample_id, instance, problem, max_tok = task_info

        # Step 2: LKH 求解
        lkh_answer = lkh_solve(
            pt, instance,
            lkh_bin=args.lkh_bin,
            runs=args.lkh_runs, seed=args.seed, timeout=args.lkh_timeout
        )
        if lkh_answer is None:
            print(f"    {sample_id}  LKH FAILED, skip")
            return None

        # Step 3: 构建后验推理 prompt
        orig_prompt = problem.build_prompt(instance)
        prompt_dict = build_posthoc_prompt(lkh_answer, orig_prompt)

        # Step 4: Gemini 调用
        t0 = time.time()
        result = _call_gemini_with_retry(
            client, prompt_dict["system"], prompt_dict["user"],
            args.model, args.sleep
        )
        if result is None:
            return None
        elapsed = time.time() - t0

        output_tokens = result.get("output_tokens") or 0

        # Step 5: 长度检查
        if output_tokens > max_tok:
            print(f"    {sample_id}  output={output_tokens} tok  [TOO_LONG > {max_tok}]  ({elapsed:.1f}s)")
            return None

        # Step 6: 内容校验
        answer_ok = _answer_has_content(result["output"], pt)
        status = "ok" if answer_ok else "NO_ANSWER"
        print(f"    {sample_id}  output={output_tokens} tok  [{status}]  ({elapsed:.1f}s)")

        if not answer_ok:
            return None

        # Step 7: 构建 record
        return {
            "id":            sample_id,
            "problem_type":  pt,
            "n":             n,
            "sample_idx":    i,
            "prompt":        prompt_dict,
            "lkh_answer":    lkh_answer,
            "output":        result["output"],
            "output_tokens": result["output_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "total_tokens":  result["total_tokens"],
            "timestamp":     datetime.now().isoformat(),
        }

    # 对每个需要补充的组合，持续生成直到达标
    with open(args.output, "a", encoding="utf-8") as fout:
        for pt, n, gap in combos_need:
            problem = get_problem(pt)
            current_valid = valid_counts.get((pt, n), 0)
            # 从已有最大 sample_idx 之后开始编号，避免 ID 冲突
            start_idx = args.num_samples  # 安全起点：原始 num_samples 之后
            for existing_id in existing_ids:
                if existing_id.startswith(f"{pt}_n{n}_s{args.seed}_i"):
                    try:
                        idx = int(existing_id.split("_i")[-1])
                        start_idx = max(start_idx, idx + 1)
                    except ValueError:
                        pass

            print(f"[{pt}_n{n}] 合格 {current_valid}/{target_per_combo}，需补充 {gap} 条")

            batch_idx = start_idx
            consecutive_failures = 0
            max_consecutive_failures = 50  # 连续失败上限，防止死循环

            while current_valid < target_per_combo:
                # 每轮生成一批任务（批大小 = 剩余缺口 × 2，留余量应对丢弃）
                remaining = target_per_combo - current_valid
                batch_size = min(remaining * 2, args.concurrency * 4)

                pt_offset = sum(ord(c) for c in pt)
                rng = np.random.default_rng(seed=args.seed + n + pt_offset + batch_idx)

                tasks = []
                for j in range(batch_size):
                    sample_id = f"{pt}_n{n}_s{args.seed}_i{batch_idx + j}"
                    instance = problem.generate_instance(n, rng)
                    if sample_id in existing_ids:
                        continue
                    tasks.append((pt, n, batch_idx + j, sample_id, instance, problem, args.max_output_tokens))

                batch_idx += batch_size

                if not tasks:
                    continue

                # 并发调用 Gemini
                saved_this_batch = 0
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    futures = {pool.submit(_process_single, t): t for t in tasks}
                    for future in as_completed(futures):
                        record = future.result()
                        if record is not None:
                            with write_lock:
                                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                                fout.flush()
                                existing_ids.add(record["id"])
                                current_valid += 1
                                saved_this_batch += 1

                print(f"    本轮: 生成 {len(tasks)} 条，合格 {saved_this_batch} 条，"
                      f"累计合格 {current_valid}/{target_per_combo}")

                if saved_this_batch == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"    连续 {max_consecutive_failures} 轮无合格样本，跳过该组合")
                        break
                else:
                    consecutive_failures = 0

            print()

    print("全部完成！\n")
    print_stats(args.output)


_SCKEY = "SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"


def _notify_serverchan(title: str, desp: str = ""):
    import urllib.request
    import urllib.parse
    try:
        data = urllib.parse.urlencode({"title": title[:100], "desp": desp[:500]}).encode()
        req = urllib.request.Request(f"https://sctapi.ftqq.com/{_SCKEY}.send", data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
