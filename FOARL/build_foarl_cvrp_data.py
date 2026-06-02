#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_foarl_cvrp_data.py — 把 UniCOP 的 solutions_cvrp20.jsonl 转成 FOARL 原版内容格式。

复现 FOARL (LLMCoSolver, NeurIPS 2025) 的 CVRP 数据"内容"格式 (instruction / input / output),
但底层尺度沿用我们自己的数据 (坐标 [0,1] 浮点, demand 浮点, capacity=1.0), 而非 FOARL 的
×1000 整数坐标 + 整数 demand —— 这样 prompt↔解↔solver_distance 自洽, 可直接复用我们的 eval。

输出每条: {id, problem_type, n, instruction, input, output, instance:[coords, demands, capacity]}
  - instruction: FOARL CVRPEnv 原版指令文本 (客户数 + 容量 + 输出格式说明)
  - input:       FOARL 原版节点描述 "Node i, coordinates: [x,y], demand: d, neighbors: [j: dist;...]"
                 (含每点 k 近邻, 与 FOARL 一致; 距离按 [0,1] 尺度用 .3f)
  - output:      FOARL 原版 "Routes: [[0,..,0],..], Objective: X.XX"
  - instance:    原始数值, 供 eval / gap 计算 (FOARL utils.compute_metric_cop 需要)

chat 外壳 (system/user/assistant) 由 train_sft_foarl.py 用 Instruct 的 chat_template 套, 本脚本不管。

源文件兼容两种 (自动识别):
  - chains_template_cvrp20.jsonl (默认, 与思维臂同源): 答案 = output 里 </think> 之后那段
  - solutions_cvrp20.jsonl:                            答案 = solution 字段
默认用 template, 保证无思维臂与思维臂的实例集逐一一致 (最干净的消融)。

用法:
  python build_foarl_cvrp_data.py \
    --src ../UniCOP-Distill/data/chains_template_cvrp20.jsonl \
    --out data/foarl_cvrp20.jsonl --k_nn 2
"""
import argparse
import json
import re
import numpy as np

# ── prompt.user 解析正则 (匹配 solutions_cvrp20.jsonl 的格式) ──────────────────
_RE_CAP   = re.compile(r"vehicle capacity\s*=\s*([-\d.]+)")
_RE_NCUST = re.compile(r"\((\d+)\s+customer nodes")
# 例: "  Node 0 (depot): (0.301, 0.387)  demand=0.0000"
#     "  Node 1: (0.459, 0.324)  demand=0.2000"
_RE_NODE  = re.compile(
    r"Node\s+(\d+)\s*(?:\(depot\))?\s*:\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s+demand\s*=\s*([-\d.]+)"
)
# 例: "Route 1: 0 -> 5 -> 2 -> 0"
_RE_ROUTE = re.compile(r"Route\s+\d+\s*:\s*(.+)")


def parse_instance(user_text: str):
    """从 prompt.user 解析 (coords[N+1,2], demands[N+1], capacity)。失败返回 None。"""
    cap_m = _RE_CAP.search(user_text)
    if not cap_m:
        return None
    capacity = float(cap_m.group(1))

    nodes = {}  # id -> (x, y, demand)
    for m in _RE_NODE.finditer(user_text):
        idx = int(m.group(1))
        nodes[idx] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    if not nodes or 0 not in nodes:
        return None
    # 节点 id 必须是 0..N 连续
    n_total = max(nodes) + 1
    if set(nodes) != set(range(n_total)):
        return None

    coords  = np.array([[nodes[i][0], nodes[i][1]] for i in range(n_total)], dtype=float)
    demands = np.array([nodes[i][2] for i in range(n_total)], dtype=float)
    return coords, demands, capacity


def parse_routes(solution_text: str):
    """从 solution 文本解析路线列表 [[0,..,0], ...]。失败返回 None。"""
    routes = []
    for line in solution_text.splitlines():
        m = _RE_ROUTE.search(line)
        if not m:
            continue
        try:
            seq = [int(tok.strip()) for tok in m.group(1).split("->") if tok.strip() != ""]
        except ValueError:
            return None
        if seq:
            routes.append(seq)
    return routes if routes else None


def get_solution_text(rec) -> str:
    """兼容两种源: solutions(有 'solution') / template(output 里 </think> 之后是答案)。"""
    if rec.get("solution"):
        return rec["solution"]
    out = rec.get("output", "") or ""
    if "</think>" in out:
        return out.split("</think>", 1)[1]
    return out


def dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def route_distance(route, D) -> float:
    return float(sum(D[route[i], route[i + 1]] for i in range(len(route) - 1)))


def knn_str(i: int, D: np.ndarray, k: int) -> str:
    """节点 i 的 k 近邻 (排除自身), FOARL 风格 "[j: dist; ...]"; [0,1] 尺度用 .3f。"""
    order = np.argsort(D[i])
    neigh = [j for j in order if j != i][:k]
    return "[" + ", ".join(f"{j}: {D[i, j]:.3f}" for j in neigh) + "]"


def build_instruction(n_cust: int, capacity: float, k_nn: int) -> str:
    # FOARL CVRPEnv.tag_prompt_and_transform_to_json_cvrp 原版指令
    return (
        f"Solve the Capacitated Vehicle Routing Problem (CVRP) with {n_cust} customers "
        "and 1 depot (node 0). Each customer node has a demand. "
        f"All vehicles have the same capacity {capacity}. You must assign each customer to exactly one route "
        "and ensure that the sum of demands on each route does not exceed the vehicle capacity. "
        "Minimize the total distance traveled.\n\n"
        f"The input includes city coordinates, the {k_nn} nearest neighbors for each city, and their respective distances. "
        "Provide the solution in the following format:\n"
        "1. A list of routes, each route as an ordered list of visited nodes (start/end at the depot).\n"
        "2. Objective: The total distance of all routes."
    )


def build_input(coords: np.ndarray, demands: np.ndarray, D: np.ndarray, k_nn: int) -> str:
    p_size = coords.shape[0]
    descs = []
    for i in range(p_size):
        coord_list = [round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3)]
        dem = round(float(demands[i]), 4)
        node_desc = (
            f"Node {i}, coordinates: {coord_list}, "
            f"demand: {dem}, "
            f"neighbors: {knn_str(i, D, k_nn)};"
        ).replace("'", "")
        descs.append(node_desc)
    inp = "".join(descs)
    # FOARL 收尾: 把最后一个 ';' 换成 '.'
    inp = ".".join(inp.rsplit(";", 1))
    return inp


def check_feasible(routes, demands, capacity, tol: float = 1e-9) -> bool:
    """FOARL utils.compute_metric_cop 的可行性判据 (start/end depot + 容量 + 全客户恰好一次)。
    tol: 容量超限容忍 (template demand 仅 2 位小数, 舍入会让极少解严格判超载, 放宽可吸收)。"""
    n_cust = len(demands) - 1
    visited = set()
    for r in routes:
        if not r or r[0] != 0 or r[-1] != 0:
            return False
        if sum(demands[node] for node in r if node != 0) > capacity + tol:
            return False
        visited.update(node for node in r if node != 0)
    return visited == set(range(1, n_cust + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../UniCOP-Distill/data/chains_template_cvrp20.jsonl",
                    help="默认 template (与思维臂同源, 取 </think> 后答案); 也兼容 solutions_*.jsonl")
    ap.add_argument("--out", default="data/foarl_cvrp20.jsonl")
    ap.add_argument("--k_nn", type=int, default=2)
    ap.add_argument("--obj_decimals", type=int, default=2, help="Objective 小数位 (FOARL 用 2)")
    ap.add_argument("--max_records", type=int, default=0, help="只转前 N 条 (验证用), 0=全量")
    ap.add_argument("--drop_infeasible", action="store_true",
                    help="丢弃(按展示 demand)严格不可行样本; 默认保留, 与思维臂同实例集")
    args = ap.parse_args()

    n_in = n_out = 0
    skip_parse_inst = skip_parse_route = skip_infeasible = 0
    # 定量自检累加器
    feas_strict = 0        # 严格 tol=0 可行
    feas_tol = 0           # 容 0.05 舍入后可行
    dist_err = []          # |recompute_dist - solver_distance|
    obj_vals = []          # solver_distance, 算相对误差用
    regex_parse_ok = 0     # 产出的 output 能否被 FOARL 的 Routes/Objective 正则解析回
    _re_routes = re.compile(r"Routes:\s*\[\s*(.*)\]", re.DOTALL)
    _re_obj    = re.compile(r"Objective:\s*([\d.]+)")

    import ast, os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.src, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.max_records and n_in > args.max_records:
                n_in -= 1
                break
            rec = json.loads(line)
            parsed = parse_instance(rec["prompt"]["user"])
            if parsed is None:
                skip_parse_inst += 1
                continue
            coords, demands, capacity = parsed
            routes = parse_routes(get_solution_text(rec))
            if routes is None:
                skip_parse_route += 1
                continue
            feas = check_feasible(routes, demands, capacity)             # 严格
            if feas:
                feas_strict += 1
            if check_feasible(routes, demands, capacity, tol=0.05):      # 容舍入
                feas_tol += 1
            if args.drop_infeasible and not feas:
                skip_infeasible += 1
                continue

            n_cust = coords.shape[0] - 1
            D = dist_matrix(coords)
            obj = float(rec.get("solver_distance", 0.0))
            # 自检: 用解析出的路线重算距离, 与 solver_distance 比对
            recomputed = sum(route_distance(r, D) for r in routes)
            dist_err.append(abs(recomputed - obj))
            obj_vals.append(obj)

            instruction = build_instruction(n_cust, capacity, args.k_nn)
            inp         = build_input(coords, demands, D, args.k_nn)
            output      = f"Routes: {routes}, Objective: {obj:.{args.obj_decimals}f}"

            # 自检: 产出的 output 能否被 FOARL 正则解析回
            rm = _re_routes.search(output)
            om = _re_obj.search(output)
            if rm and om:
                try:
                    back = ast.literal_eval(f"[{rm.group(1).strip()}]")
                    if all(isinstance(r, list) for r in back):
                        regex_parse_ok += 1
                except (SyntaxError, ValueError):
                    pass

            fout.write(json.dumps({
                "id":           rec.get("id"),
                "problem_type": "cvrp",
                "n":            rec.get("n", n_cust),
                "instruction":  instruction,
                "input":        inp,
                "output":       output,
                "instance":     [coords.tolist(), demands.tolist(), capacity],
            }, ensure_ascii=False) + "\n")
            n_out += 1

    # ── 定量报告 ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  FOARL CVRP 数据转换报告")
    print(f"  源:   {args.src}")
    print(f"  出:   {args.out}")
    print("-" * 60)
    print(f"  读入:            {n_in}")
    print(f"  成功写出:        {n_out}  ({n_out / max(n_in,1):.1%})")
    print(f"  跳过-实例解析失败: {skip_parse_inst}")
    print(f"  跳过-路线解析失败: {skip_parse_route}")
    print(f"  跳过-不可行(仅 --drop_infeasible 时): {skip_infeasible}")
    print("-" * 60)
    _denom = n_out + skip_infeasible
    print(f"  [自检1] 可行(严格 tol=0):  {feas_strict}/{_denom} = {feas_strict/max(_denom,1):.2%}")
    print(f"          可行(容 0.05 舍入): {feas_tol}/{_denom} = {feas_tol/max(_denom,1):.2%}")
    print(f"          (template demand 仅 2 位小数, 舍入会让少量解严格判超载, 容忍后应≈100%)")
    if dist_err:
        de = np.array(dist_err); ov = np.array(obj_vals)
        print(f"  [自检2] 重算距离 vs solver_distance: "
              f"mean|Δ|={de.mean():.4f}, max|Δ|={de.max():.4f}, "
              f"相对误差 mean|Δ|/obj={de.mean()/max(ov.mean(),1e-9):.2%}")
        print(f"          (Δ 应≈0; 偏大说明坐标/路线解析或距离口径不一致)")
    print(f"  [自检3] output 能被 FOARL Routes/Objective 正则解析: "
          f"{regex_parse_ok}/{n_out} = {regex_parse_ok / max(n_out,1):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
