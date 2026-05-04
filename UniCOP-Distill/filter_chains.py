"""过滤 chains JSONL，只保留 think chain 中节点覆盖率高于阈值的样本。

覆盖率 = think 中以各种形式提及的答案节点数 / 答案总节点数。
同时输出保留样本的 ID 列表，供 Stage 1 做数据去重。
"""

import argparse
import json
import re


def parse_routes(text: str) -> list[list[int]]:
    routes = []
    for m in re.finditer(r"Route\s*\d*\s*:\s*([\d\s\->]+)", text):
        nodes = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if nodes:
            routes.append(nodes)
    return routes


def get_customers(routes: list[list[int]]) -> set[int]:
    return {n for r in routes for n in r if n != 0}


_RE_NODE_LIST = re.compile(
    r"[Nn]odes?\s+([\d]+(?:\s*[,，]\s*\d+)+(?:\s*(?:and|&)\s*\d+)?)")
_RE_NODE_RANGE = re.compile(r"[Nn]odes?\s+(\d+)\s*[-–—]+\s*(\d+)")
_RE_NODE_RANGE2 = re.compile(r"[Nn]odes?\s+(\d+)\s+to\s+(\d+)")


def count_mentioned_nodes(think: str, answer_nodes: set[int]) -> set[int]:
    """检查 think chain 中提到了哪些答案节点（Route 格式 + 列表/范围 + 文字提及）。"""
    mentioned = set()

    route_nodes = get_customers(parse_routes(think))
    mentioned |= (route_nodes & answer_nodes)

    for m in _RE_NODE_LIST.finditer(think):
        nums = {int(x) for x in re.findall(r"\d+", m.group(0))}
        mentioned |= (nums & answer_nodes)

    for pat in (_RE_NODE_RANGE, _RE_NODE_RANGE2):
        for m in pat.finditer(think):
            start, end = int(m.group(1)), int(m.group(2))
            if 0 < end - start <= 25:
                mentioned |= (set(range(start, end + 1)) & answer_nodes)

    for n in answer_nodes - mentioned:
        patterns = [
            rf"Node\s+{n}\b",
            rf"\b{n}\s*->",
            rf"->\s*{n}\b",
            rf"nodes?\s+{n}\b",
            rf"visit.*\b{n}\b",
        ]
        for p in patterns:
            if re.search(p, think, re.IGNORECASE):
                mentioned.add(n)
                break
    return mentioned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids_output", default=None,
                        help="保留样本的 ID 列表输出路径（每行一个 ID）")
    parser.add_argument("--min_coverage", type=float, default=0.8,
                        help="think 中节点覆盖答案节点的最低比例")
    args = parser.parse_args()

    kept, dropped = 0, 0
    kept_ids = []

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            output = r.get("output", "")

            m = re.search(r"<think>(.*?)</think>(.*)", output, re.DOTALL)
            if not m:
                dropped += 1
                continue

            think, answer = m.group(1), m.group(2)
            answer_routes = parse_routes(answer)
            answer_nodes = get_customers(answer_routes)

            if not answer_nodes:
                dropped += 1
                continue

            mentioned = count_mentioned_nodes(think, answer_nodes)
            coverage = len(mentioned) / len(answer_nodes)

            if coverage >= args.min_coverage:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
                kept_ids.append(r.get("id", ""))
            else:
                dropped += 1

    total = kept + dropped
    print(f"总计: {total}  保留: {kept} ({kept/max(total,1):.1%})  过滤: {dropped}")
    print(f"输出: {args.output}")

    if args.ids_output:
        with open(args.ids_output, "w", encoding="utf-8") as f:
            for sid in kept_ids:
                f.write(sid + "\n")
        print(f"ID 列表: {args.ids_output} ({len(kept_ids)} 条)")


if __name__ == "__main__":
    main()
