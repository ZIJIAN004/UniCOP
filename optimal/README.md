# optimal/ — LKH / HGS 近最优基线

为 UniCOP-Reason 的 LLM 求解器提供 **optimality gap** 的分母：用传统求解器对**同一批测试实例**求近最优解，缓存后供 `evaluate.py` 计算 `gap = model_dist / optimal_cost − 1`。

现有 `ref_solver.py` 只是启发式（NN+2opt，注释明确写了 "not meant to be optimal"），仅用于训练 reward 归一化；本模块才是评测用的近最优基线。

## 求解器分配

| 问题 | 求解器 | 说明 |
|------|--------|------|
| TSP   | **LKH** | 设 `LKH_BIN` 时用 LKH；未设则回退 PyVRP/HGS |
| CVRP  | **HGS**（PyVRP） | |
| TSPTW | **HGS**（PyVRP，单车辆 VRPTW 建模） | |
| VRPTW | **HGS**（PyVRP） | |
| TSPDL | ✗ 暂不支持 | PyVRP 无 draft-limit 模型；如需对标 PIP，建议复用其官方 LKH-3 ground truth（github.com/jieyibi/PIP-constraint） |

> LKH/HGS 对 n≤100 实际就是最优（gap < 0.01%），是该领域公认的“optimal”基线（exact TSP 用 Concorde，其余用 LKH/HGS）。

## 依赖（轻量，独立于训练栈）

```
numpy
pyvrp            # pip install pyvrp
LKH 二进制        # 可选，仅 TSP；export LKH_BIN=/path/to/LKH
```

不依赖 torch / vllm / deepspeed / datasets，可在任意机器单独跑。

## 用法

在 `UniCOP-Reason/` 目录下：

```bash
# 默认：4 类问题 × n=100 × 1000 实例，单实例 5s
python -m optimal.build_optimal --sizes 100 --num_test 1000 --timeout 5

# 指定类型/规模/并行
python -m optimal.build_optimal --problem_types cvrp vrptw --sizes 50 100 --workers 8

# TSP 用 LKH
LKH_BIN=/path/to/LKH python -m optimal.build_optimal --problem_types tsp --sizes 100
```

输出缓存：`optimal/cache/{type}_n{n}_seed{seed}_N{num_test}.json`

## ⚠️ 与 evaluate.py 的对齐（关键）

`evaluate.py:evaluate_single` 用 `rng = np.random.default_rng(seed=9999)` 顺序生成 `num_test` 个实例。本模块以**同样的 seed、同样的顺序**调用 `generate_instance` 重建实例，因此第 i 个实例严格对应。

- **seed 默认 9999**，必须与 `evaluate.py` 一致。若改了 evaluate 的 seed，务必同步 `--seed`。
- **前缀一致性**：实例序列只由 seed 顺序决定，故“缓存 N=1000、评测用前 100 个”是合法的——`loader.load_costs` 会自动找到 N≥请求值的缓存并取前缀。可“基线一次性算大集合，LLM 评测抽子集”。

## 集成到 evaluate.py

`evaluate.py` 目前只输出 `avg_best_dist`，无 gap。接入示意：

```python
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
