"""
定量校验 build_reverse_chains.py 产出的 reverse chain 是否正确。
独立从样本文本重算所有不变量，按条统计通过率（禁止肉眼抽查下结论）。

检查项：
  C1 覆盖性     : final routes 恰好覆盖客户 1..n 各一次
  C2 容量可行   : 每条路线 load <= capacity
  C3 标签一致   : <think> 内 final routes == </think> 后答案
  C4 giant 一致 : user prompt 的 "Fixed visit order" == final routes 客户顺序拼接
  C5 keep 算术  : 每个 keep 步 cap_before - dem == cap_after，且 dem == 真实 demand
  C6 切分原因   : 关闭步的原因与"后续剩余客户 vs cap"的可行性判据一致
  C7 depot 距离 : 关闭步 d= 等于 last_node 到 depot 的欧氏距离

用法： python validate_reverse_chains.py --input data/chains_reverse_cvrp20_val.jsonl
"""

import argparse
import json
import math
import re

EPS = 5e-3  # 文本保留 2~3 位小数，比较留余量


def parse_nodes(user_prompt):
    coords, demands = {}, {}
    for m in re.finditer(
        r"Node\s+(\d+).*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s+demand=([\d.]+)",
        user_prompt,
    ):
        nid = int(m.group(1))
        coords[nid] = (float(m.group(2)), float(m.group(3)))
        demands[nid] = float(m.group(4))
    cap = float(re.search(r"capacity=([\d.]+)", user_prompt).group(1))
    return coords, demands, cap


def parse_final_routes(output):
    after = output.split("</think>")[1]
    routes = []
    for line in after.strip().splitlines():
        m = re.match(r"Route\s*\d*\s*:\s*(.+)", line.strip())
        if m:
            routes.append([int(x) for x in re.split(r"\s*->\s*", m.group(1))])
    return routes


def parse_think_routes(output):
    seg = output.split("4. **Final routes**:")[1].split("</think>")[0]
    routes = []
    for line in seg.strip().splitlines():
        m = re.match(r"Route\s*\d*\s*:\s*(.+)", line.strip())
        if m:
            routes.append([int(x) for x in re.split(r"\s*->\s*", m.group(1))])
    return routes


def dist(coords, a, b):
    return math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])


def validate_record(rec):
    """返回 dict: 各检查项 True/False。"""
    out = rec["output"]
    user = rec["prompt"]["user"]
    coords, demands, cap = parse_nodes(user)
    n = rec["n"]

    final_routes = parse_final_routes(out)
    think_routes = parse_think_routes(out)
    res = {}

    # C1 覆盖性
    custs = sorted(v for r in final_routes for v in r if v != 0)
    res["C1_cover"] = (custs == list(range(1, n + 1)))

    # C2 容量
    res["C2_cap"] = all(
        sum(demands[v] for v in r if v != 0) <= cap + EPS for r in final_routes
    )

    # C3 标签一致
    res["C3_label"] = (final_routes == think_routes)

    # C4 giant 一致
    m = re.search(r"Fixed visit order:\s*([0-9 ->]+)", user)
    giant = [int(x) for x in re.split(r"\s*->\s*", m.group(1).strip())]
    giant_from_routes = [v for r in final_routes for v in r if v != 0]
    res["C4_giant"] = (giant == giant_from_routes)

    # C5 keep 算术 + C7 depot 距离 + C6 切分原因：逐步重放第 2 段
    step_seg = out.split("2. **Step-by-step splitting**:")[1].split("3. **Verification**:")[0]
    keep_ok, close_ok, dist_ok = True, True, True

    # 重建 giant 顺序下"已放置数"，用于判断每个关闭步的剩余客户
    placed = 0
    cur_route_idx = 0
    for line in step_seg.strip().splitlines():
        km = re.search(r"keep (\d+) \(dem=([\d.]+)\) cap→([-\d.]+)", line)
        capm = re.search(r"cap=([-\d.]+)", line)
        if km:
            v = int(km.group(1)); dem_txt = float(km.group(2)); after_txt = float(km.group(3))
            before = float(capm.group(1))
            # 算术：before - dem == after
            if abs((before - dem_txt) - after_txt) > EPS:
                keep_ok = False
            # demand 与真实一致
            if abs(dem_txt - demands[v]) > EPS:
                keep_ok = False
            placed += 1
            continue
        if "return depot" in line:
            capc = float(capm.group(1))
            dm = re.search(r"d=([\d.]+)", line)
            last = final_routes[cur_route_idx][-2]  # 倒数第二个=最后一个客户
            if dm and abs(float(dm.group(1)) - dist(coords, last, 0)) > EPS:
                dist_ok = False
            remaining = giant[placed:]
            feasible = [u for u in remaining if demands[u] <= capc + EPS]
            if "all customers placed" in line:
                if remaining:
                    close_ok = False
            elif "no room" in line:
                # 必须确实无人可装
                if feasible:
                    close_ok = False
            elif "feasible:" in line:
                # 必须确有可装者，且列出的节点都真的 <= cap
                if not feasible:
                    close_ok = False
                for fu in re.findall(r"(\d+)\(dem=([\d.]+)<=([\d.]+)\)", line):
                    if float(fu[1]) > float(fu[2]) + EPS:
                        close_ok = False
            else:
                close_ok = False
            cur_route_idx += 1

    res["C5_keep_arith"] = keep_ok
    res["C6_split_reason"] = close_ok
    res["C7_depot_dist"] = dist_ok
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    checks = ["C1_cover", "C2_cap", "C3_label", "C4_giant",
              "C5_keep_arith", "C6_split_reason", "C7_depot_dist"]
    counts = {c: 0 for c in checks}
    total = 0
    all_pass = 0
    fail_examples = {c: None for c in checks}

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            res = validate_record(rec)
            ok_all = True
            for c in checks:
                if res[c]:
                    counts[c] += 1
                else:
                    ok_all = False
                    if fail_examples[c] is None:
                        fail_examples[c] = rec["id"]
            if ok_all:
                all_pass += 1

    print(f"校验样本数: {total}")
    for c in checks:
        pct = counts[c] / total * 100
        flag = "" if counts[c] == total else f"  <-- 失败例: {fail_examples[c]}"
        print(f"  {c:16s}: {counts[c]}/{total} ({pct:.2f}%){flag}")
    print(f"全部通过: {all_pass}/{total} ({all_pass/total*100:.2f}%)")


if __name__ == "__main__":
    main()
