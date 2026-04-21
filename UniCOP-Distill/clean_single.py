"""
clean_chains.py
读取单个 chains jsonl 文件，清洗后写回（或输出到指定路径）。

逻辑：
  1. 读取指定的单个 jsonl 文件
  2. 同 ID 去重：保留最后出现的那条
  3. 格式修复：无 <think> 标签但有推理内容的，自动补标签
  4. 质量过滤：无推理内容或 <think> 内容太短的，废弃
  5. 输出到原文件（覆盖）或指定路径

运行：
  python clean_chains.py --input data/chains.jsonl
  python clean_chains.py --input data/chains.jsonl --output data/chains_clean.jsonl
"""

import argparse
import json
import os
from collections import OrderedDict


def find_answer_start(output: str) -> int:
    """找到答案的起始位置（Route: 或 0 ->）。"""
    lower = output.lower()
    # 先找 </think> 后面的答案（有标签的情况）
    think_end = output.find("</think>")
    if think_end != -1:
        return think_end + len("</think>")

    # 无标签：找 Route: 或 0 ->
    route_idx = lower.find("route:")
    arrow_idx = output.find("0 ->")

    candidates = [i for i in [route_idx, arrow_idx] if i != -1]
    if candidates:
        return min(candidates)
    return -1


def _contains_full_route(think: str, lkh_answer: str) -> bool:
    """
    检查 think 开头是否直接抄了完整路线。
    只看 think 前 300 字符，如果其中按顺序包含 >= 80% 的答案节点，判定为抄答案。
    后半段出现路线总结不算。
    """
    import re
    nodes = re.findall(r"\d+", lkh_answer)
    if len(nodes) < 5:
        return False

    think_head = think[:300]
    head_numbers = re.findall(r"\d+", think_head)

    match_count = 0
    head_idx = 0
    for node in nodes:
        while head_idx < len(head_numbers):
            if head_numbers[head_idx] == node:
                match_count += 1
                head_idx += 1
                break
            head_idx += 1
        else:
            break

    return match_count >= len(nodes) * 0.8


def clean_single(record: dict) -> dict | None:
    """
    清洗单条记录。返回清洗后的 record，或 None 表示废弃。
    """
    output = record.get("output", "")
    if not output or not output.strip():
        return None

    # 检查是否有答案（只看 </think> 之后的部分，避免 think 中提及 route 被误判）
    think_end = output.find("</think>")
    answer_part = output[think_end:].lower() if think_end != -1 else output.lower()
    has_answer = "route:" in answer_part or "0 ->" in answer_part
    if not has_answer:
        return None

    # 已有完整 <think>...</think> 标签
    if "<think>" in output and "</think>" in output:
        think_start = output.index("<think>") + 7
        think_end = output.index("</think>")
        think_content = output[think_start:think_end].strip()
        if len(think_content) < 200:
            return None  # think 内容太短
        # think 里包含完整答案路线 → 废弃
        lkh = record.get("lkh_answer", "")
        if lkh and _contains_full_route(think_content, lkh):
            return None
        return record

    # 有 <think> 但无 </think>：在答案前补 </think>
    if "<think>" in output and "</think>" not in output:
        think_start = output.index("<think>") + 7
        remaining = output[think_start:]
        # 在 remaining 中找答案起始位置
        remaining_lower = remaining.lower()
        route_idx = remaining_lower.find("route:")
        arrow_idx = remaining.find("0 ->")
        candidates = [i for i in [route_idx, arrow_idx] if i != -1]
        if not candidates:
            return None  # 无答案，废弃
        ans_pos = min(candidates)
        think_content = remaining[:ans_pos].strip()
        answer = remaining[ans_pos:].strip()
        if len(think_content) < 200:
            return None
        lkh = record.get("lkh_answer", "")
        if lkh and _contains_full_route(think_content, lkh):
            return None
        record = dict(record)
        record["output"] = f"<think>\n{think_content}\n</think>\n{answer}"
        return record

    # 无 <think> 标签：尝试补标签
    answer_start = find_answer_start(output)
    if answer_start <= 50:
        return None  # 答案在开头，无推理内容

    reasoning = output[:answer_start].strip()
    answer = output[answer_start:].strip()

    if len(reasoning) < 50:
        return None

    # 补标签前也检查是否开头抄路线
    lkh = record.get("lkh_answer", "")
    if lkh and _contains_full_route(reasoning, lkh):
        return None

    record = dict(record)
    record["output"] = f"<think>\n{reasoning}\n</think>\n{answer}"
    return record


def main():
    parser = argparse.ArgumentParser(description="清洗单个 chains jsonl 文件")
    parser.add_argument("--input", type=str, required=True,
                        help="输入 jsonl 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认覆盖原文件）")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    if not os.path.exists(args.input):
        print(f"文件不存在: {args.input}")
        return

    print(f"输入文件: {args.input}")

    # 读取记录，同 ID 去重
    all_records = OrderedDict()
    total_read = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = r.get("id", f"unknown_{total_read}")
            all_records[record_id] = r
            total_read += 1

    deduped = len(all_records)
    duplicates = total_read - deduped

    print(f"\n读取总条数: {total_read}")
    print(f"去重后: {deduped} (去除 {duplicates} 条重复 ID)")

    # 清洗
    cleaned = []
    discarded_no_answer = 0
    discarded_short_think = 0
    discarded_no_reasoning = 0
    discarded_route_in_think = 0
    fixed_tags = 0

    for record_id, record in all_records.items():
        output = record.get("output", "")
        has_think = "<think>" in output and "</think>" in output

        # 先检测是否在 think 里抄了完整路线（用于统计）
        is_route_in_think = False
        if has_think:
            ts = output.index("<think>") + 7
            te = output.index("</think>")
            think_text = output[ts:te].strip()
            lkh = record.get("lkh_answer", "")
            if lkh and _contains_full_route(think_text, lkh):
                is_route_in_think = True

        result = clean_single(record)
        if result is None:
            if not ("route:" in output.lower() or "0 ->" in output):
                discarded_no_answer += 1
            elif is_route_in_think:
                discarded_route_in_think += 1
            elif has_think:
                discarded_short_think += 1
            else:
                discarded_no_reasoning += 1
            continue

        if not has_think and "<think>" in result["output"]:
            fixed_tags += 1

        cleaned.append(result)

    print(f"\n清洗结果:")
    print(f"  保留: {len(cleaned)}")
    print(f"  补标签: {fixed_tags}")
    print(f"  废弃 - 无答案: {discarded_no_answer}")
    print(f"  废弃 - think太短: {discarded_short_think}")
    print(f"  废弃 - think开头抄路线: {discarded_route_in_think}")
    print(f"  废弃 - 无推理: {discarded_no_reasoning}")

    # 统计
    from collections import Counter
    type_counts = Counter(r["problem_type"] for r in cleaned)
    size_counts = Counter(r["n"] for r in cleaned)
    print(f"\n问题类型分布: {dict(sorted(type_counts.items()))}")
    print(f"规模分布:     {dict(sorted(size_counts.items()))}")

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n已保存到: {args.output} ({len(cleaned)} 条)")


if __name__ == "__main__":
    main()
