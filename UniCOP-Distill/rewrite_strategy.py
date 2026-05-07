"""将模板思维链的 Strategy 段用 LLM 改写为自然语言。

保留 step-by-step / verification / final routes 不动，
只替换 section 1 的模板统计为自然语言策略分析。

用法：
    # 预览 3 条
    python rewrite_strategy.py \
        --input data/chains_template_cvrp20.jsonl \
        --api_key sk-xxx \
        --preview 3

    # 正式生成
    python rewrite_strategy.py \
        --input data/chains_template_cvrp20.jsonl \
        --api_key sk-xxx \
        --num_samples 2000 \
        --output data/chains_hybrid_cvrp20.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI


STRATEGY_SYSTEM = (
    "You are a logistics route planning expert solving a CVRP. "
    "Write a concise strategy analysis (100-150 words) in first person, "
    "describing how you would plan the routes for this problem.\n\n"
    "Cover: the geographic layout of nodes, how they naturally cluster, "
    "demand balancing across routes, and the visit order logic within each "
    "group (e.g. sweep direction, nearest-neighbor).\n\n"
    "Write as original planning (\"I notice...\", \"I'll group...\", "
    "\"Starting from depot...\"). Be specific about node IDs and positions. "
    "Do NOT use bullet points, numbered lists, or markdown headers — write "
    "flowing prose. Your output must read as if you are deciding the routes "
    "yourself, not describing or referencing any pre-existing solution."
)


def parse_template_sections(output: str) -> dict | None:
    """拆分模板链的 <think> 为 4 个 section。"""
    if "<think>" not in output or "</think>" not in output:
        return None

    think_start = output.index("<think>") + 7
    think_end = output.index("</think>")
    think = output[think_start:think_end]
    after_think = output[think_end + 8:]

    m_step = re.search(r'\n2\.\s*\*\*Step-by-step construction\*\*:', think)
    if not m_step:
        return None

    strategy = think[:m_step.start()].strip()
    rest = think[m_step.start():]

    # strategy 里去掉开头的 "1. **Strategy**: "
    strategy = re.sub(r'^1\.\s*\*\*Strategy\*\*:\s*', '', strategy).strip()

    return {
        "strategy": strategy,
        "rest": rest,
        "after_think": after_think.strip(),
    }


def build_strategy_prompt(user_prompt: str, sections: dict, problem_type: str) -> dict:
    """构建让 LLM 改写 strategy 的 prompt。"""
    # 从 rest 中提取路线分组信息（从 verification 或 final routes）
    rest = sections["rest"]

    # 从 final routes 提取
    route_lines = []
    for line in rest.split("\n"):
        line = line.strip()
        if re.match(r'Route\s+\d+\s*:', line, re.I):
            route_lines.append(line)

    # 从模板 strategy 提取 demand 信息
    template_strategy = sections["strategy"]

    user_content = (
        f"{user_prompt}\n\n"
        f"######\n"
        + "\n".join(route_lines)
        + f"\n######"
    )

    return {"system": STRATEGY_SYSTEM, "user": user_content}


def call_llm(client: OpenAI, system: str, user: str,
             model: str, max_tokens: int,
             max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content or ""
            content = content.lstrip()
            if content.startswith("</think>"):
                content = content[len("</think>"):].lstrip()

            usage = response.usage
            return {
                "text": content.strip(),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            wait = min(2 ** attempt * 5, 60)
            print(f"  API error (retry {attempt+1}/{max_retries}, wait {wait}s): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None


def reassemble(new_strategy: str, sections: dict) -> str:
    """用新 strategy 替换模板 strategy，重新组装 output。"""
    think_content = (
        f"1. **Strategy**: {new_strategy}\n"
        f"{sections['rest']}"
    )
    return f"<think>\n{think_content}\n</think>\n{sections['after_think']}"


def quality_check_strategy(text: str) -> tuple[bool, str]:
    """检查 LLM 生成的 strategy 质量。"""
    if len(text.strip()) < 50:
        return False, "TOO_SHORT"
    if len(text.strip()) > 2000:
        return False, "TOO_LONG"

    lower = text.lower()
    leak_words = [
        "given", "provided", "assigned", "expected",
        "the routes are", "the solution is",
        "pre-existing", "above routes", "these routes are",
    ]
    for w in leak_words:
        if w in lower:
            return False, f"LEAK:{w}"

    if any(w in lower for w in ["bullet", "numbered list", "markdown"]):
        return False, "META_REFERENCE"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite template chain strategy sections with LLM")
    parser.add_argument("--input", required=True,
                        help="Template chains JSONL file")
    parser.add_argument("--output", default="data/chains_hybrid_cvrp20.jsonl")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 读取模板链
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} template chains")

    # 过滤可解析的
    valid = []
    for r in records:
        sections = parse_template_sections(r["output"])
        if sections:
            valid.append((r, sections))
    print(f"Parseable: {len(valid)} / {len(records)}")

    if args.num_samples > 0:
        valid = random.sample(valid, min(args.num_samples, len(valid)))
    print(f"Sampled: {len(valid)}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # 断点续跑
    existing_ids = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line.strip())["id"])
                    except Exception:
                        pass
    if existing_ids:
        valid = [(r, s) for r, s in valid if r["id"] not in existing_ids]
        print(f"After skipping existing: {len(valid)}")

    if not valid:
        print("All done!")
        return

    # 处理函数
    stats_lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}

    def process_one(item):
        rec, sections = item
        prompt = build_strategy_prompt(
            rec["prompt"]["user"], sections, rec.get("problem_type", "cvrp")
        )

        for attempt in range(3):
            result = call_llm(
                client, prompt["system"], prompt["user"],
                args.model, args.max_tokens
            )
            if result is None:
                continue

            ok, reason = quality_check_strategy(result["text"])
            if not ok:
                print(f"    [{rec['id']}] Strategy quality fail: {reason}")
                continue

            new_output = reassemble(result["text"], sections)
            with stats_lock:
                stats["ok"] += 1

            out_rec = dict(rec)
            out_rec["output"] = new_output
            out_rec["strategy_tokens"] = {
                "prompt": result["prompt_tokens"],
                "output": result["output_tokens"],
            }
            out_rec["original_strategy"] = sections["strategy"]
            out_rec["timestamp"] = datetime.now().isoformat()
            return out_rec

        with stats_lock:
            stats["fail"] += 1
        return None

    # 预览模式
    if args.preview > 0:
        preview_items = valid[:args.preview]
        print(f"\nPreview: {len(preview_items)} samples\n")

        with ThreadPoolExecutor(max_workers=min(args.preview, args.concurrency)) as pool:
            futures = {pool.submit(process_one, item): item for item in preview_items}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    item = futures[future]
                    print(f"  [{item[0]['id']}] FAILED\n")
                    continue

                rid = result["id"]
                sections_new = parse_template_sections(result["output"])
                print(f"  [{rid}]  strategy_tokens: "
                      f"in={result['strategy_tokens']['prompt']} "
                      f"out={result['strategy_tokens']['output']}")
                print(f"  Original strategy:")
                for line in result["original_strategy"].split("\n")[:5]:
                    print(f"    {line}")
                print(f"  New strategy:")
                if sections_new:
                    for line in sections_new["strategy"].split("\n"):
                        print(f"    {line}")
                print()
        print(f"Preview done: {stats['ok']}/{len(preview_items)} passed")
        return

    # 全量生成
    t_start = time.time()
    write_lock = threading.Lock()
    total = len(valid)
    print(f"\nGenerating {total} samples (concurrency={args.concurrency})...\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process_one, item): item for item in valid}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result is not None:
                    with write_lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                if done % 100 == 0 or done == total:
                    elapsed = (time.time() - t_start) / 60
                    speed = done / max(elapsed, 0.01)
                    eta = (total - done) / max(speed, 0.01)
                    print(f"  {done}/{total}  ok={stats['ok']} fail={stats['fail']}  "
                          f"speed={speed:.0f}/min  ETA={eta:.0f}min")

    elapsed = (time.time() - t_start) / 60
    total_done = stats["ok"] + stats["fail"]
    print(f"\nDone: {stats['ok']}/{total_done} passed "
          f"({stats['ok']/max(total_done,1)*100:.0f}%)  time={elapsed:.1f}min")


if __name__ == "__main__":
    main()
