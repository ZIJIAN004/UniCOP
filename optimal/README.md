# optimal/ — LKH / HGS 近最优基线

为 UniCOP-Reason 的 LLM 求解器提供 **optimality gap** 的分母：用传统求解器对**同一批测试实例**求近最优解，缓存后供 `evaluate.py` 计算 `gap = model_dist / optimal_cost − 1`。

现有 `ref_solver.py` 只是启发式（NN+2opt，注释明确写了 "not meant to be optimal"），仅用于训练 reward 归一化；本模块才是评测用的近最优基线。

> **位置**：本模块在 **UniCOP 仓库根目录**（`UniCOP/optimal/`，与 UniCOP-Reason / UniCOP-Distill 平级），复用 `UniCOP-Reason/problems` 的统一问题生成逻辑（脚本内已自动把 `UniCOP-Reason` 加入 `sys.path`）。

## 求解器分配

| 问题 | 求解器 | 求最优旋钮 |
|------|--------|-----------|
| TSP   | **LKH**（固定，缺 `LKH_BIN` 直接报错，不回退） | `--lkh_runs` 调大（默认 10） |
| CVRP  | **HGS**（PyVRP） | `--timeout` 调大（MaxRuntime） |
| TSPTW | **HGS**（PyVRP，单车辆 VRPTW 建模） | `--timeout` 调大 |
| VRPTW | **HGS**（PyVRP） | `--timeout` 调大 |
| TSPDL | ✗ 暂不支持 | PyVRP 无 draft-limit 模型；如需对标 PIP，建议复用其官方 LKH-3 ground truth（github.com/jieyibi/PIP-constraint） |

> LKH/HGS 对 n≤100 把参数调大后达到真实最优级别（best-known），是该领域公认的“optimal”基线。
> 严格说二者是启发式，给出的是**最优级/best-known 而非可证明最优**（要可证明最优需 exact TSP 用 Concorde、VRP 用分支定界，代价高得多）。对算 LLM 的 gap 而言这点差异可忽略。

## 依赖（轻量，独立于训练栈）

```
numpy            # 阶段一（生成实例）仅需 numpy
pyvrp            # 阶段二（求解）需要，pip install pyvrp
LKH 二进制        # 可选，仅 TSP；export LKH_BIN=/path/to/LKH
```

不依赖 torch / vllm / deepspeed / datasets，可在任意机器单独跑。

## 用法：两阶段（生成与求解解耦）

均在 **`UniCOP/`（仓库根）目录下**运行。

### 阶段一 — 冻结测试实例（只用 numpy，很快）

```bash
# 默认：4 类 × n∈{20,50,100} × 1000 实例，seed=9999
python -m optimal.generate_testset

# 自定义
python -m optimal.generate_testset --problem_types tsp cvrp --sizes 50 --num_instances 1000 --seed 9999
```

输出冻结实例：`optimal/instances/{type}_n{n}_seed{seed}_N{num}.json`
（含 metadata + `instances` 列表；求解与评测都读这同一批。）

### 阶段二 — 求 LKH/HGS 近最优（读冻结实例）

**TSP 固定用 LKH**，必须先有 LKH 二进制（`export LKH_BIN=/path/to/LKH`，与 UniCOP-Distill 同一变量）；否则求解 TSP 会直接报错。

```bash
# 求全部冻结实例（HGS 跑 30s/实例求最优；TSP 用 LKH，RUNS=10）
export LKH_BIN=/path/to/LKH
python -m optimal.build_optimal --sizes 20 50 100 --timeout 30 --lkh_runs 10 --workers 8

# 只求前 N 个 / 指定类型（不含 TSP 时无需 LKH）
python -m optimal.build_optimal --problem_types cvrp vrptw --sizes 50 --num_test 200 --timeout 30
```

输出缓存：`optimal/cache/{type}_n{n}_seed{seed}_N{num}.json`（`costs[i]` 对应冻结实例第 i 个）

> **求最优旋钮**：HGS 调大 `--timeout`（n=100 建议 30~60s），LKH 调大 `--lkh_runs`。
> **时间成本**：HGS 的 `MaxRuntime` 会跑满 timeout，总耗时 ≈ `实例数 × timeout / workers`。
> 例：cvrp/vrptw/tsptw 各 1000 × 3 规模 × 30s ÷ 16 进程 ≈ 每类约 1.6h；用 `--workers` 拉满 CPU、必要时先小 `--num_test` 验证。

## ⚠️ 与 evaluate.py 的对齐（关键）

冻结实例用 `np.random.default_rng(seed)` 顺序调用 `generate_instance` 生成，与 `evaluate.py:evaluate_single`（其每次 `default_rng(9999)` 后顺序生成）**完全一致**。所以 seed=9999 冻结的实例与 evaluate.py 现场生成的逐一相同。

- **seed 默认 9999**，须与 `evaluate.py` 一致；改了 evaluate 的 seed 要同步 `--seed`。
- **前缀一致性**：序列只由 seed 顺序决定，故“冻结/求解 N=1000、评测用前 100 个”合法——`load_instances` / `load_costs` 会自动找 N≥请求值的文件取前缀。
- **真正固定**：`instances/` 默认 gitignore（确定性可重生成）；若要字节级永久固定，`git add -f instances/` 入库。
- **推荐下一步**：让 `evaluate.py` 也改用 `load_instances` 读冻结集，使模型与基线严格同一批（当前靠 seed 一致性保证等价）。

## 集成到 evaluate.py

`evaluate.py` 目前只输出 `avg_best_dist`，无 gap。`evaluate.py` 在 `UniCOP-Reason/`，而本模块在仓库根，需先把根目录加入 path。接入示意：

```python
import sys, os
# UniCOP 仓库根（UniCOP-Reason 的上一级），使 `import optimal` 可用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimal.loader import load_costs, optimality_gap

# evaluate_single 内，已有 best_dists（每个实例最优可行解距离；不可行实例补 None）
opt = load_costs(problem_type, problem_size, num_test, seed=9999)
gap = optimality_gap(model_dists=best_dists_aligned, optimal_costs=opt)
print(f"  optimality gap: {gap['mean_gap_pct']:.2f}% "
      f"± {gap['sem_pct']:.2f}%  (matched {gap['matched']}/{gap['n_total']})")
```

注意 `best_dists` 需按实例索引对齐（不可行实例占位 None），不能只 append 可行的那些。

## 测试集规模说明

- 本仓库默认 `num_test=1000`（`config.py`）。
- 吴耀鑫等 PIP 框架（NeurIPS 2024，TSPTW/TSPDL）测试集为 **10,000**（n=50/100），是 POMO/AM 一脉的惯例。
- 基线侧（LKH/HGS）很便宜，可一次性算到 1000 甚至 10000；瓶颈在 LLM 推理，评测时按算力抽子集即可。N=100 仅建议用于开发期 smoke test（gap 标准误约 0.1–0.4%）。
