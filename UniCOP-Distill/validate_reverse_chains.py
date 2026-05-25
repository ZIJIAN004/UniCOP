"""
定量校验 build_reverse_chains.py 产出的 reverse chain 是否正确。
独立从样本重算所有不变量，按条统计通过率（禁止肉眼抽查下结论）。

检查项：
  C1 覆盖性     : final routes 恰好覆盖客户 1..n 各一次
  C2 容量可行   : 每条路线 load <= capacity
  C3 标签一致   : <think> 内 final routes == </think> 后答案
  C4 giant 一致 : user prompt 的 "Fixed visit order" == final routes 客户顺序拼接
  C5 第2段同源  : 反向第 2 段 == 正向 build_steps_cvrp(routes) 逐字节相等
  C6 第3段同源  : 反向第 3 段 == 正向 _build_verification(routes) 逐字节相等

C5/C6 直接证明反向与正向**逐字节同格式**（Unvisited 选择池 / cap=X-X=X /
feasible 候选池 / → select M / 两种 return depot），不引入任何新步骤模式。

用法： python validate_reverse_chains.py --input data/chains_reverse_cvrp20_val.jsonl
"""

import argparse
import json
import re

from build_think_chains import (
    parse_instance_from_prompt,
    build_steps_cvrp,
    _build_verification,
)


def parse_routes_from_text(text):
    routes = []
    for line in text.strip().splitlines():
        m = re.match(r"Route\s*\d*\s*:\s*(.+)", line.strip())
        if m:
            routes.append([int(x) for x in re.split(r"\s*->\s*", m.group(1))])
    return routes


def section(output, start_tag, end_tag):
    """取出 output 中 start_tag 与 end_tag 之间的正文（去首尾空行）。"""
    return output.split(start_tag)[1].split(end_tag)[0].strip("\n")


def validate_record(rec):
    out = rec["output"]
    user = rec["prompt"]["user"]
    n = rec["n"]
    instance = parse_instance_from_prompt(user, "cvrp")
    coords, demands, cap = instance["coords"], instance["demands"], instance["capacity"]

    final_routes = parse_routes_from_text(out.split("</think>")[1])
    think_routes = parse_routes_from_text(
        out.split("4. **Final routes**:")[1].split("</think>")[0]
    )
    res = {}

    custs = sorted(v for r in final_routes for v in r if v != 0)
    res["C1_cover"] = (custs == list(range(1, n + 1)))
    res["C2_cap"] = all(
        sum(demands[v] for v in r if v != 0) <= cap + 5e-3 for r in final_routes
    )
    res["C3_label"] = (final_routes == think_routes)

    m = re.search(r"Fixed visit order:\s*([0-9 ->]+)", user)
    giant = [int(x) for x in re.split(r"\s*->\s*", m.group(1).strip())]
    res["C4_giant"] = (giant == [v for r in final_routes for v in r if v != 0])

    # C5/C6：与正向生成器逐字节比对
    seg2 = section(out, "2. **Step-by-step construction**:\n", "\n\n3. **Verification**:")
    seg3 = section(out, "3. **Verification**:\n", "\n\n4. **Final routes**:")
    exp2 = "\n".join(build_steps_cvrp(final_routes, coords, demands, cap, stride=1))
    exp3 = "\n".join(_build_verification(final_routes, n, multi_route=True))
    res["C5_steps_same"] = (seg2 == exp2)
    res["C6_verify_same"] = (seg3 == exp3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    checks = ["C1_cover", "C2_cap", "C3_label", "C4_giant",
              "C5_steps_same", "C6_verify_same"]
    counts = {c: 0 for c in checks}
    total, all_pass = 0, 0
    fail_ex = {c: None for c in checks}

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            res = validate_record(rec)
            ok = True
            for c in checks:
                if res[c]:
                    counts[c] += 1
                else:
                    ok = False
                    if fail_ex[c] is None:
                        fail_ex[c] = rec["id"]
            if ok:
                all_pass += 1

    print(f"校验样本数: {total}")
    for c in checks:
        pct = counts[c] / total * 100
        flag = "" if counts[c] == total else f"  <-- 失败例: {fail_ex[c]}"
        print(f"  {c:16s}: {counts[c]}/{total} ({pct:.2f}%){flag}")
    print(f"全部通过: {all_pass}/{total} ({all_pass/total*100:.2f}%)")


if __name__ == "__main__":
    main()
