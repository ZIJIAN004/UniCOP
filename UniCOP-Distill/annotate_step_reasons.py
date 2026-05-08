"""为 think chain 的关键决策步骤添加简短推理标注。

在 step-by-step 构造部分，对非最近选择、主动返回 depot、新路线首步
等关键步骤，通过 DeepSeek API 生成 15-30 词的理由，插入在 → 决策前。

用法：
    # 预览 3 条
    python annotate_step_reasons.py \
        --input data/chains_hybrid_cvrp20.jsonl \
        --api_key sk-xxx \
        --preview 3

    # 正式生成
    python annotate_step_reasons.py \
        --input data/chains_hybrid_cvrp20.jsonl \
        --api_key sk-xxx \
        --output data/chains_hybrid_cvrp20_annotated.jsonl
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI


REASON_SYSTEM = (
    "Write a SHORT reason (10-20 words) for a vehicle routing decision. "
    "Rules:\n"
    "- Do NOT start with \"Node X was chosen/selected\" — start with the reason directly\n"
    "- Use lowercase, no period at end\n"
    "- Reference specific numbers (distances, demands, capacity) only when they drive the decision\n"
    "- Examples of good style:\n"
    "  \"closer to remaining southwest cluster, preserving capacity for high-demand nodes\"\n"
    "  \"demand 0.30 fits tightly, using capacity before starting new route\"\n"
    "  \"nearest node 14 belongs to a different cluster, so skip to node 3\"\n"
    "  \"low remaining cap 0.04 makes serving node 1 wasteful, better restart\"\n"
)


# ── 问题解析 ─────────────────────────────────────────────────────────────────

def parse_problem(user_prompt: str) -> dict | None:
    cap_m = re.search(r'vehicle capacity=([\d.]+)', user_prompt)
    if not cap_m:
        return None
    capacity = float(cap_m.group(1))

    nodes = {}
    for m in re.finditer(
        r'Node (\d+)(?:\s*\(depot\))?:\s*\(([\d.]+),\s*([\d.]+)\)\s*demand=([\d.]+)',
        user_prompt,
    ):
        nid = int(m.group(1))
        nodes[nid] = {
            "x": float(m.group(2)),
            "y": float(m.group(3)),
            "demand": float(m.group(4)),
        }
    if not nodes:
        return None
    return {"capacity": capacity, "nodes": nodes}


# ── think 结构解析 ────────────────────────────────────────────────────────────

def parse_think_sections(output: str) -> dict | None:
    if "<think>" not in output or "</think>" not in output:
        return None

    think_start = output.index("<think>") + 7
    think_end = output.index("</think>")
    think = output[think_start:think_end]

    m_step = re.search(r'\n2\.\s*\*\*Step-by-step construction\*\*:', think)
    m_verif = re.search(r'\n3\.\s*\*\*Verification\*\*:', think)
    if not m_step or not m_verif:
        return None

    return {
        "prefix": output[:think_start],
        "before_steps": think[:m_step.end()],
        "steps_text": think[m_step.end():m_verif.start()],
        "after_steps": think[m_verif.start():],
        "suffix": output[think_end:],
    }


# ── step 行解析与分类 ────────────────────────────────────────────────────────

def parse_feasible(line: str) -> list:
    feas_idx = line.find("| feasible:")
    if feas_idx == -1:
        return []

    decision_idx = -1
    for pat in (" → select ", " → remaining nodes"):
        idx = line.find(pat)
        if idx != -1:
            decision_idx = idx
            break
    if decision_idx == -1:
        return []

    feas_str = line[feas_idx + len("| feasible:"):decision_idx]
    candidates = []
    for m in re.finditer(
        r'(\d+)\(d=([\d.]+),dem=([\d.]+),cap→([-\d.]+)\)', feas_str
    ):
        candidates.append({
            "node": int(m.group(1)),
            "dist": float(m.group(2)),
            "demand": float(m.group(3)),
            "cap_after": float(m.group(4)),
        })
    return candidates


def parse_step_line(line: str) -> dict | None:
    line = line.strip()
    if not line.startswith("[R"):
        return None

    label_m = re.match(r'\[R(\d+),(\d+)\]', line)
    if not label_m:
        return None

    route_idx = int(label_m.group(1))
    step_num = int(label_m.group(2))

    if "→ all customers served" in line:
        return {"line": line, "route": route_idx, "step": step_num,
                "type": "all_served", "feasible": [], "selected": None}

    if "→ no feasible → return depot" in line:
        return {"line": line, "route": route_idx, "step": step_num,
                "type": "no_feasible", "feasible": [], "selected": None}

    if "→ remaining nodes better served by new route" in line:
        feasible = parse_feasible(line)
        return {"line": line, "route": route_idx, "step": step_num,
                "type": "voluntary_return", "feasible": feasible, "selected": None}

    sel_m = re.search(r'→ select (\d+)', line)
    if not sel_m:
        return None

    selected = int(sel_m.group(1))
    feasible = parse_feasible(line)

    if step_num == 1:
        step_type = "route_start"
    elif feasible:
        nearest = min(feasible, key=lambda c: c["dist"])
        step_type = "non_nearest" if nearest["node"] != selected else "normal"
    else:
        step_type = "normal"

    return {"line": line, "route": route_idx, "step": step_num,
            "type": step_type, "feasible": feasible, "selected": selected}


def classify_steps(steps_text: str) -> list[dict]:
    lines = steps_text.strip().split("\n")
    result = []
    current_node = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Unvisited:"):
            current_node = 0
            result.append({"line": line, "type": "unvisited"})
            continue

        parsed = parse_step_line(line)
        if parsed:
            if parsed["step"] == 1:
                current_node = 0
            parsed["current_node"] = current_node
            if parsed["selected"] is not None:
                current_node = parsed["selected"]
            else:
                current_node = 0
            result.append(parsed)
        else:
            result.append({"line": line, "type": "other"})

    return result


# ── LLM prompt 构建 ──────────────────────────────────────────────────────────

def build_reason_prompt(step: dict, problem: dict) -> str:
    nodes = problem["nodes"]
    cap = problem["capacity"]
    step_type = step["type"]

    cap_m = re.search(r'cap=[\d.]+-[\d.]+=([\d.]+)', step["line"])
    if not cap_m:
        cap_m = re.search(r'cap=([\d.]+)', step["line"])
    current_cap = float(cap_m.group(1)) if cap_m else cap

    curr = step["current_node"]
    curr_coord = (f"({nodes[curr]['x']:.3f}, {nodes[curr]['y']:.3f})"
                  if curr in nodes else "")

    feas_lines = []
    for c in step["feasible"]:
        n = c["node"]
        coord = (f"({nodes[n]['x']:.3f}, {nodes[n]['y']:.3f})"
                 if n in nodes else "")
        feas_lines.append(
            f"  node {n} {coord}: dist={c['dist']:.3f}, "
            f"demand={c['demand']:.2f}, cap_after={c['cap_after']:.2f}"
        )
    feas_str = "\n".join(feas_lines)

    if step_type == "route_start":
        selected = step["selected"]
        return (
            f"CVRP, {len(nodes)-1} customers, capacity={cap}.\n"
            f"Starting new route from depot (node 0) {curr_coord}.\n"
            f"Remaining capacity: {current_cap:.2f}\n"
            f"Nearest feasible candidates:\n{feas_str}\n"
            f"Selected: node {selected}\n\n"
            f"Why was node {selected} chosen to start this route?"
        )

    if step_type == "non_nearest":
        selected = step["selected"]
        nearest = min(step["feasible"], key=lambda c: c["dist"])
        return (
            f"CVRP, {len(nodes)-1} customers, capacity={cap}.\n"
            f"Current position: node {curr} {curr_coord}, "
            f"remaining cap: {current_cap:.2f}\n"
            f"Feasible candidates (by distance):\n{feas_str}\n"
            f"Nearest is node {nearest['node']} (dist={nearest['dist']:.3f}), "
            f"but node {selected} was selected.\n\n"
            f"Why was node {selected} chosen over node {nearest['node']}?"
        )

    if step_type == "voluntary_return":
        return (
            f"CVRP, {len(nodes)-1} customers, capacity={cap}.\n"
            f"Current position: node {curr} {curr_coord}, "
            f"remaining cap: {current_cap:.2f}\n"
            f"Feasible candidates still available:\n{feas_str}\n"
            f"Decision: return to depot instead of continuing.\n\n"
            f"Why start a new route rather than serve remaining feasible nodes?"
        )

    return ""


# ── reason 插入与质量检查 ─────────────────────────────────────────────────────

def insert_reason(line: str, reason: str) -> str:
    for pat in (" → select ", " → remaining nodes"):
        idx = line.find(pat)
        if idx != -1:
            return line[:idx] + f" — {reason}" + line[idx:]
    return line


def quality_check_reason(text: str) -> tuple[bool, str]:
    text = text.strip()
    if not text:
        return False, "EMPTY"

    words = text.split()
    if len(words) < 5:
        return False, "TOO_SHORT"
    if len(words) > 35:
        return False, "TOO_LONG"

    for c in ("→", "|", "\n", "\r"):
        if c in text:
            return False, f"BAD_CHAR:{repr(c)}"

    if re.search(r'\[R\d', text):
        return False, "CONTAINS_STEP_LABEL"

    sentence_ends = len(re.findall(r'\.\s+[A-Z]', text)) + (1 if text.endswith(".") else 0)
    if sentence_ends > 2:
        return False, "TOO_MANY_SENTENCES"

    return True, "ok"


# ── LLM 调用 ─────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system: str, user: str,
             model: str, temperature: float,
             max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2048,
                extra_body={"thinking": {"type": "enabled"}},
            )
            msg = response.choices[0].message
            content = msg.content or ""
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if content.startswith("</think>"):
                content = content[len("</think>"):].strip()

            if not content:
                reasoning = getattr(msg, 'reasoning_content', '') or ""
                if reasoning:
                    lines = [l.strip() for l in reasoning.strip().split('\n') if l.strip()]
                    content = lines[-1] if lines else ""

            usage = response.usage
            return {
                "text": content,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            wait = min(2 ** attempt * 5, 60)
            print(f"  API error (retry {attempt+1}/{max_retries}, wait {wait}s): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None


# ── 单条记录处理 ──────────────────────────────────────────────────────────────

def process_one(rec: dict, client: OpenAI, model: str, temperature: float,
                stats: dict, stats_lock: threading.Lock) -> dict | None:
    problem = parse_problem(rec["prompt"]["user"])
    if not problem:
        with stats_lock:
            stats["fail"] += 1
        return None

    sections = parse_think_sections(rec["output"])
    if not sections:
        with stats_lock:
            stats["fail"] += 1
        return None

    steps = classify_steps(sections["steps_text"])
    key_types = {"route_start", "non_nearest", "voluntary_return"}
    key_steps = [(i, s) for i, s in enumerate(steps) if s.get("type") in key_types]

    local_total = sum(1 for s in steps if s.get("type") not in ("unvisited", "other"))
    local_annotated = 0

    for idx, step in key_steps:
        step_type = step["type"]
        user_prompt = build_reason_prompt(step, problem)
        if not user_prompt:
            continue

        reason_text = None
        for attempt in range(2):
            result = call_llm(client, REASON_SYSTEM, user_prompt,
                              model, temperature)
            if result is None:
                continue

            ok, check_msg = quality_check_reason(result["text"])
            if ok:
                reason_text = result["text"]
                break
            if attempt == 0:
                print(f"    [{rec['id']}] R{step['route']},{step['step']} "
                      f"quality fail: {check_msg}, retrying...")

        if reason_text:
            reason_text = reason_text.rstrip(".")
            steps[idx]["line"] = insert_reason(step["line"], reason_text)
            local_annotated += 1
            with stats_lock:
                stats["by_type"][step_type] += 1

    new_steps_lines = [s["line"] for s in steps]
    new_steps_text = "\n" + "\n".join(new_steps_lines) + "\n"
    new_think = sections["before_steps"] + new_steps_text + sections["after_steps"]
    new_output = sections["prefix"] + new_think + sections["suffix"]

    with stats_lock:
        stats["ok"] += 1
        stats["total_steps"] += local_total
        stats["annotated_steps"] += local_annotated

    out_rec = dict(rec)
    out_rec["output"] = new_output
    out_rec["annotation_stats"] = {
        "total_steps": local_total,
        "annotated": local_annotated,
    }
    out_rec["annotation_timestamp"] = datetime.now().isoformat()
    return out_rec


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotate key decision steps with brief reasons via LLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_annotated{ext}"

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} chains")

    if args.num_samples > 0:
        records = records[:args.num_samples]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    existing_ids: set[str] = set()
    if not args.preview and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line.strip())["id"])
                    except Exception:
                        pass
        if existing_ids:
            records = [r for r in records if r["id"] not in existing_ids]
            print(f"After skipping existing: {len(records)}")

    if not records:
        print("All done!")
        return

    stats_lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "total_steps": 0, "annotated_steps": 0,
             "by_type": {"route_start": 0, "non_nearest": 0, "voluntary_return": 0}}

    # ── 预览模式 ──────────────────────────────────────────────────────────
    if args.preview > 0:
        preview_items = records[:args.preview]
        print(f"\nPreview: {len(preview_items)} samples\n")

        with ThreadPoolExecutor(
            max_workers=min(args.preview, args.concurrency)
        ) as pool:
            futures = {
                pool.submit(process_one, rec, client, args.model,
                            args.temperature, stats, stats_lock): rec
                for rec in preview_items
            }
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    rec = futures[future]
                    print(f"  [{rec['id']}] FAILED\n")
                    continue

                ann = result["annotation_stats"]
                print(f"  [{result['id']}] annotated "
                      f"{ann['annotated']}/{ann['total_steps']} steps")

                sections = parse_think_sections(result["output"])
                if sections:
                    for sline in sections["steps_text"].strip().split("\n"):
                        sline = sline.strip()
                        if "—" in sline and sline.startswith("[R"):
                            print(f"    {sline}")
                print()

        print(f"Preview done: {stats['ok']}/{len(preview_items)} processed")
        print(f"Steps: {stats['annotated_steps']}/{stats['total_steps']} annotated")
        print(f"By type: {stats['by_type']}")
        return

    # ── 全量生成 ──────────────────────────────────────────────────────────
    t_start = time.time()
    write_lock = threading.Lock()
    total = len(records)
    print(f"\nAnnotating {total} chains (concurrency={args.concurrency})...\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(process_one, rec, client, args.model,
                            args.temperature, stats, stats_lock): rec
                for rec in records
            }
            done = 0
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result is not None:
                    with write_lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                if done % 50 == 0 or done == total:
                    elapsed = (time.time() - t_start) / 60
                    speed = done / max(elapsed, 0.01)
                    eta = (total - done) / max(speed, 0.01)
                    print(f"  {done}/{total}  ok={stats['ok']} fail={stats['fail']}  "
                          f"annotated={stats['annotated_steps']}  "
                          f"speed={speed:.0f}/min  ETA={eta:.0f}min")

    elapsed = (time.time() - t_start) / 60
    print(f"\nDone in {elapsed:.1f}min")
    print(f"Chains: {stats['ok']}/{stats['ok']+stats['fail']} processed")
    print(f"Steps: {stats['annotated_steps']}/{stats['total_steps']} annotated")
    print(f"By type: {stats['by_type']}")


if __name__ == "__main__":
    main()
