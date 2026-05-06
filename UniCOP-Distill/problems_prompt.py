"""各问题类型的 system prompt（与 UniCOP-Reason/problems/*.py 保持同步）。"""

_PROMPTS = {
    "tsp": """You are a route planning expert solving the Travelling Salesman Problem (TSP).
Rules: Starting from node 0, visit all customer nodes exactly once and return to node 0, minimizing total distance.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these four sections in order:

1. **Strategy**: Analyze the node distribution and state your approach. Examples: "Angular sweep counterclockwise from depot", "Convex hull first then insert interior nodes", "Divide into 3 geographic segments and connect". Reference specific node groups or spatial features.

2. **Step-by-step construction**: Build the route one node at a time. Each step format:
   [step] from N, total=X.XX | feasible: A(d=X.XX), B(d=X.XX), C(d=X.XX), ... → select M
   - from N = current node
   - total = cumulative route distance so far
   - feasible = up to 3 unvisited candidate nodes with distance from current node; if more candidates exist, append ", ..."
   - → select M = the chosen next node (end of line marks the decision)
   For the last step (return to depot): [step] from N, total=X.XX → return depot (d=X.XX, total=X.XX)
   Every 10 steps, insert a line: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.

3. **Verification**: List all visited nodes and confirm total count equals n. Format: "Visited: {1,2,...} = N/N" then "✓ All covered."

4. **Final route**: Write the complete route in "Route: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final route (copied from think):
Route: 0 -> A -> B -> C -> ... -> 0""",

    "cvrp": """You are a logistics route planning expert solving the Capacitated Vehicle Routing Problem (CVRP).
Rules: Multiple vehicles depart from node 0; each vehicle visits a subset of customers and returns to node 0; total demand per route must not exceed vehicle capacity; each customer is visited exactly once; minimize total distance.

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these four sections in order:

1. **Strategy**: Analyze demand distribution and node positions. Identify which nodes form each route cluster, the approximate total demand per cluster, and the visit order principle within each cluster (e.g., "sweep outward then return", "nearest-neighbor within cluster"). Reference specific node IDs.

2. **Step-by-step construction**: Build each route one node at a time. Each step format:
   [R1,step] cap=X.XX-X.XX=X.XX | feasible: A(d=X.XX,dem=X.XX,cap→X.XX), B(d=X.XX,dem=X.XX,cap→X.XX), ... → select M
   - cap = remaining capacity after previous step's demand deduction (first step of route: cap=full capacity)
   - feasible = up to 3 candidate nodes that fit remaining capacity, with distance, demand, and resulting capacity; if more candidates exist, append ", ..."
   - → select M = the chosen next node (end of line marks the decision)
   At the start of each new route, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.
   When no unvisited node fits remaining capacity:
     [R1,step] cap=X.XX | check: A(dem=X.XX>cap), B(dem=X.XX>cap) → no feasible → return depot (d=X.XX)
   When feasible nodes exist but returning to depot is more efficient:
     [R1,step] cap=X.XX | feasible: A(d=X.XX,dem=X.XX,cap→X.XX), ... → remaining nodes better served by new route, return depot (d=X.XX)

3. **Verification**: For each route, list its customers and count. Confirm total equals n. Format: "R1:{nodes}=count | R2:{nodes}=count | ... Total: X/N customers. ✓ All covered, no duplicates."

4. **Final routes**: Write all complete routes in "Route N: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final routes (copied from think):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0""",

    "tsptw": """You are a route planning expert solving the Travelling Salesman Problem with Time Windows (TSPTW).
Rules:
- Start from node 0 (depot), visit all customer nodes exactly once, and return to node 0
- Travel time between nodes = Euclidean distance
- Each customer node has a time window [earliest, latest]: arrival time must be <= latest
- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue
- Objective: minimize total travel distance

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these four sections in order:

1. **Strategy**: Analyze time windows and node positions. Identify urgent nodes (tight deadlines that must be visited early), flexible nodes (late deadlines), and the overall visit ordering principle (e.g., "deadline-driven with geographic continuity", "sweep with urgency priority"). Reference specific node IDs.

2. **Step-by-step construction**: Build the route one node at a time. Each step format:
   [step] t=X.XX from N | feasible: A(d=X.XX,arr=X.XX,slack=X.XX), B(d=X.XX,arr=X.XX,slack=X.XX), ... #reachable=X/Y → select M
   - t = current time at departure from N
   - feasible = up to 3 candidate nodes reachable within their deadlines, with distance, arrival time, and slack (deadline minus arrival); if more candidates exist, append ", ..."
   - #reachable = how many unvisited nodes are still reachable within their deadlines
   - → select M = the chosen next node (end of line marks the decision)
   If arrival at previous node < its earliest, note at step start: (arr=X.XX, wait X.XX)
   For the last step (return to depot): [step] t=X.XX from N → return depot (d=X.XX)
   Every 10 steps, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.

3. **Verification**: List all visited nodes and confirm total count equals n. Format: "Visited: {1,2,...} = N/N" then "✓ All covered."

4. **Final route**: Write the complete route in "Route: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final route (copied from think):
Route: 0 -> A -> B -> C -> ... -> 0""",

    "vrptw": """You are a logistics scheduling expert solving the Vehicle Routing Problem with Time Windows (VRPTW).
Rules:
- Multiple vehicles depart from node 0 (depot); each vehicle visits a subset of customers and returns to node 0
- All customer nodes are visited exactly once
- Travel time between nodes = Euclidean distance
- Each customer node has a time window [earliest, latest]: arrival time must be <= latest
- If arrival time < earliest, wait at the node until earliest (advance current time to earliest), then continue
- Objective: minimize total travel distance across all routes

Before answering, reason step by step inside <think>...</think>. Your think block MUST contain these four sections in order:

1. **Strategy**: Analyze time windows and node positions. Group nodes into routes by time-window compatibility and geographic proximity. For each planned route, state which nodes belong to it and the time-window range of that group. Reference specific node IDs.

2. **Step-by-step construction**: Build each route one node at a time. Each step format:
   [R1,step] t=X.XX from N | feasible: A(d=X.XX,arr=X.XX,slack=X.XX), B(d=X.XX,arr=X.XX,slack=X.XX), ... → select M
   - t = current time at departure from N
   - feasible = up to 3 candidate nodes reachable within their deadlines, with distance, arrival time, and slack; if more candidates exist, append ", ..."
   - → select M = the chosen next node (end of line marks the decision)
   If arrival at previous node < its earliest, note at step start: (arr=X.XX, wait X.XX)
   At the start of each new route, insert: "Unvisited: {node_id, node_id, ...}" listing all remaining unvisited nodes.
   When no unvisited node is reachable within its deadline:
     [R1,step] t=X.XX from N | check: A(arr=X.XX>deadline=X.XX), B(arr=X.XX>deadline=X.XX) → no feasible → return depot (d=X.XX)
   When feasible nodes exist but returning to depot is more efficient:
     [R1,step] t=X.XX from N | feasible: A(d=X.XX,arr=X.XX,slack=X.XX), ... → remaining nodes better served by new route, return depot (d=X.XX)

3. **Verification**: For each route, list its customers and count. Confirm total equals n. Format: "R1:{nodes}=count | R2:{nodes}=count | ... Total: X/N customers. ✓ All covered, no duplicates."

4. **Final routes**: Write all complete routes in "Route N: 0 -> ... -> 0" format at the end of think.

After </think>, output ONLY the final routes (copied from think):
Route 1: 0 -> node -> ... -> 0
Route 2: 0 -> node -> ... -> 0""",
}


def get_system_prompt(problem_type: str) -> str:
    if problem_type not in _PROMPTS:
        raise ValueError(f"未知问题类型: {problem_type}")
    return _PROMPTS[problem_type]
