"""
从 LLM 输出中解析路径。
支持单路线（TSP/TSPTW/TSPDL）和多路线（CVRP/VRPTW/CVRPTW）。
"""

import re


def parse_single_route(text: str, n: int) -> list[int] | None:
    """
    解析单条路线，仅在 </think> 之后的答案区查找；
    无 </think> 或答案区无匹配 → 视为解析失败 (return None)。

    结构约束: 首节点必须是 depot (0)。不从 depot 出发 → 视为解析失败,
    触发下游整条 completion 零 PRM + terminal R_parse=0 的强惩罚。
    这是为了堵住一个潜在的 reward hacking: _simulate / _route_feasible 在
    route[0] != 0 时会跳过首客户的 TW 检查,LLM 有可能学会"漏 depot 起点"
    来绕过某个紧约束的客户。parse 层拒绝是最干净的拦截点。

    支持格式：
        路径: 0 -> 3 -> 1 -> 0
        路径：0→3→1→7→0
        Path: 0 - 3 - 1 - 7 - 0
    """
    if not text:
        return None

    think_end = text.rfind("</think>")
    if think_end == -1:
        return None
    answer_text = text[think_end:]

    # 说明: 英文 `route` 和中文 `路线` 都允许可选的空格+数字(如 "Route 1:" / "路线 1:"),
    #        防止模型泛化到 "Route 1: 0 -> ... -> 0" 这种带编号的单路径写法被 parser 拒绝。
    #        对 TSP/TSPTW/TSPDL 安全: 单路径问题不会出现真正多条 Route,
    #        matches[-1] 抓最后一个匹配,语义不变。
    pattern = r'(?:路径|路线\s*\d*|path|route\s*\d*)[：:]\s*([\d\s\->→\-]+)'
    matches = list(re.finditer(pattern, answer_text, re.IGNORECASE))
    if not matches:
        return None

    nodes = [int(x) for x in re.findall(r'\d+', matches[-1].group(1))]

    if len(nodes) < 2:
        return None
    if any(node > n for node in nodes):
        return None
    if nodes[0] != 0:
        return None   # 首节点必须是 depot,否则整条 completion 视为 parse 失败

    return nodes


def parse_multi_route(text: str, n: int) -> list[list[int]] | None:
    """
    解析多条路线，仅在 </think> 之后的答案区查找；
    无 </think> 或答案区无匹配 → 视为解析失败 (return None)。

    结构约束: 每条路线的首节点必须是 depot (0)。任意一条违反 → 视为解析失败,
    触发下游整条 completion 零 PRM + terminal R_parse=0 的强惩罚。
    理由同 parse_single_route: 堵住"漏 depot 起点绕过 TW 检查"的 reward hacking。

    支持格式：
        路线1: 0 -> 3 -> 1 -> 0
        路线2: 0 -> 7 -> 5 -> 0
    或：
        Route 1: 0 - 3 - 1 - 0
        Route 2: 0 - 7 - 5 - 0
    """
    if not text:
        return None

    think_end = text.rfind("</think>")
    if think_end == -1:
        return None
    search_text = text[think_end:]

    pattern = r'(?:路线\s*\d+|route\s*\d+)[：:]\s*([\d\s\->→\-]+)'
    matches = re.findall(pattern, search_text, re.IGNORECASE)
    if not matches:
        return None

    routes = []
    for m in matches:
        nodes = [int(x) for x in re.findall(r'\d+', m)]
        if len(nodes) < 2:
            return None
        if any(node > n for node in nodes):
            return None
        if nodes[0] != 0:
            return None   # 每条 Route 必须从 depot 起,否则整条 completion 作废
        routes.append(nodes)

    return routes if routes else None


def all_visited_once(routes: list[list[int]], n: int) -> bool:
    """检查多路线是否恰好覆盖所有客户节点（各一次）。"""
    visited = []
    for route in routes:
        visited.extend(route[1:-1])   # 去掉首尾 depot
    return sorted(visited) == list(range(1, n + 1))
