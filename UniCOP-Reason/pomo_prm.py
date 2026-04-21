"""
POMO Process Reward Model (PRM)

用预训练的 POMO 模型为 LLM 输出的路径提供 per-step 增量过程奖励。
支持 TSP / CVRP / VRPTW 三种问题。

核心流程：
    1. 解析 LLM 输出 → 节点序列
    2. 规范化排序（多路线）
    3. 前缀校验（约束模拟）
    4. POMO 批量前缀评估
    5. 计算增量 reward（客户节点）+ 反事实 reward（depot 回访）
    6. 映射到 token 位置
"""

import os
import sys
import math
import re
import glob
import torch
import numpy as np
from dataclasses import dataclass


# ── POMO 模型超参数（与 POMO-Baseline/HYPER_PARAMS.py 一致） ─────────────────

POMO_MODEL_PARAMS = {
    "embedding_dim":     128,
    "encoder_layer_num": 6,
    "head_num":          8,
    "qkv_dim":           16,
    "ff_hidden_dim":     512,
    "logit_clipping":    10,
}


# ── 数据格式 ──────────────────────────────────────────────────────────────────


def parse_route_numbers(completion: str) -> list[tuple[int, int]]:
    """
    仅在 </think> 之后的答案区找所有 'Route N:' / '路线 N：'，返回 [(N值, N 数字的字符位置)]。
    无 </think> → 视为解析失败，返回 []。

    位置取的是数字 N 本身（regex capture group 1 起点），而非 'Route' 起点，
    便于把奖励精确归因到编号 token。
    """
    think_end = completion.rfind("</think>")
    if think_end == -1:
        return []
    offset = think_end + len("</think>")
    text = completion[offset:]

    pattern = re.compile(r'(?:route|路线)\s*(\d+)[：:]', re.IGNORECASE)
    return [(int(m.group(1)), offset + m.start(1)) for m in pattern.finditer(text)]


@dataclass
class StepRewards:
    """
    per-step PRM 奖励结果（仅"在合法前缀内每一步走得好不好"）。
    违规惩罚/格式判定/覆盖检查均交给 terminal reward 处理。

    customer_rewards/depot_rewards 长度 = 合法前缀内的客户/depot 数；
    其对应的字符位置由 trainer 映射到 token，未涵盖的位置 PRM mask = 0
    （不参与 PRM loss 的分子或分母）。
    """
    customer_rewards: list[float]       # 合法前缀内每个客户的增量 reward
    depot_rewards: list[float]          # 合法前缀内每个 depot 回访的反事实 reward
    customer_token_positions: list[int] # 对应客户在 completion 中的字符位置
    depot_token_positions: list[int]    # 对应 depot 在 completion 中的字符位置
    n: int = 0                          # 总客户数
    covered: int = 0                    # 合法前缀内覆盖的客户数


# ── POMO 模型加载 ─────────────────────────────────────────────────────────────

def _import_pomo(pomo_baseline_dir: str):
    """将 POMO-Baseline 目录加入 sys.path，导入模型和环境类。"""
    if pomo_baseline_dir not in sys.path:
        sys.path.insert(0, pomo_baseline_dir)

    from source.models.tsp_model  import TSPModel
    from source.models.cvrp_model import CVRPModel
    from source.models.vrptw_model import VRPTWModel
    from source.envs.tsp_env  import TSPEnv
    from source.envs.cvrp_env import CVRPEnv
    from source.envs.vrptw_env import VRPTWEnv

    return {
        "tsp":  (TSPModel, TSPEnv),
        "cvrp": (CVRPModel, CVRPEnv),
        "vrptw": (VRPTWModel, VRPTWEnv),
    }


class POMOPRM:
    """POMO Process Reward Model

    TSP/CVRP/VRPTW 走标准 POMO checkpoint;
    TSPTW 走 PIP-D (NeurIPS 2024) checkpoint,同时提供 foresight 判决 +
    POMO-style 批量 rollout value (详见 utils/pipd_wrapper.py)。
    """

    # POMO PRM 支持的问题类型 (TSPTW 用 PIP-D backbone)
    SUPPORTED = {"tsp", "cvrp", "vrptw", "tsptw"}

    # TSPTW 不走 POMO 接口,这里单独维护 PIP-D ckpt 路径约定
    PIPD_CKPT_SUBDIR_FMT   = "tsptw{n_total}_easy"    # n_total = 客户数 + 1
    PIPD_CKPT_MODEL_SUBDIR = "POMO_star_PIP-D"
    PIPD_CKPT_FILENAME     = "epoch-10000.pt"

    def __init__(self, pomo_ckpt_dir: str, pomo_baseline_dir: str,
                 device: str = "cuda",
                 pipd_ckpt_dir: str | None = None,
                 pipd_dir: str | None = None):
        self.ckpt_dir      = pomo_ckpt_dir
        self.device        = torch.device(device)
        self._classes      = _import_pomo(pomo_baseline_dir)
        self._cache        = {}          # (problem_type, problem_size) → model
        # TSPTW 走 PIP-D: 允许显式传入 ckpt 根目录和代码目录,默认自动推导
        self.pipd_ckpt_dir = pipd_ckpt_dir   # 期望指向 {PIP-D baseline}/POMO+PIP/pretrained/TSPTW
        self.pipd_dir      = pipd_dir         # 期望指向 {PIP-D baseline}/POMO+PIP

    # ── Checkpoint 检查 ───────────────────────────────────────────────

    def check_checkpoints(self, problem_types: list[str],
                          problem_sizes: list[int]):
        """训练开始前检查所需 POMO checkpoint 是否存在。"""
        missing = []
        for pt in problem_types:
            if pt not in self.SUPPORTED:
                continue
            for ps in problem_sizes:
                ckpt_path = self._ckpt_path(pt, ps)
                if not os.path.exists(ckpt_path):
                    missing.append(ckpt_path)
        if missing:
            raise FileNotFoundError(
                f"缺少 POMO checkpoint:\n" +
                "\n".join(f"  {p}" for p in missing)
            )
        print(f"POMO PRM: 所有 checkpoint 就绪 "
              f"(types={problem_types}, sizes={problem_sizes})")

    def _ckpt_path(self, problem_type: str, problem_size: int) -> str:
        """
        返回指定问题/规模的 checkpoint 绝对路径。
        - TSP/CVRP/VRPTW: POMO-Baseline 格式
            {pomo_ckpt_dir}/{timestamp}__POMO_{TYPE_UPPER}_n{N}/MODEL_FINAL.pt
        - TSPTW: PIP-D (NeurIPS 2024) 格式
            {pipd_ckpt_dir}/tsptw{N+1}_easy/POMO_star_PIP-D/epoch-10000.pt
            注: PIP-D 的 tsptw50 = 50 总节点 (49 customers),
                所以 UniCOP n 客户对应 PIP-D 的 tsptw{n+1}
        """
        if problem_type == "tsptw":
            if not self.pipd_ckpt_dir:
                return f"<PIPD_CKPT_DIR 未设置: TSPTW n={problem_size}>"
            n_total = problem_size + 1
            return os.path.join(
                self.pipd_ckpt_dir,
                self.PIPD_CKPT_SUBDIR_FMT.format(n_total=n_total),
                self.PIPD_CKPT_MODEL_SUBDIR,
                self.PIPD_CKPT_FILENAME,
            )

        pattern = os.path.join(
            self.ckpt_dir,
            f"*POMO_{problem_type.upper()}_n{problem_size}",
        )
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            return os.path.join(
                self.ckpt_dir,
                f"<NOT FOUND: POMO_{problem_type.upper()}_n{problem_size}>",
                "MODEL_FINAL.pt",
            )
        # 字典序最大 = 时间戳最新（POMO-Baseline 用 {YYYYMMDD}_{HHMM} 前缀）
        return os.path.join(candidates[-1], "MODEL_FINAL.pt")

    # ── 模型加载（懒加载 + 缓存） ────────────────────────────────────

    def _get_model(self, problem_type: str, problem_size: int):
        key = (problem_type, problem_size)
        if key not in self._cache:
            if problem_type == "tsptw":
                # TSPTW 走 PIP-D 包装器,不是 POMO model
                from utils.pipd_wrapper import PIPDWrapper, _resolve_pipd_dir
                pipd_dir = _resolve_pipd_dir(self.pipd_dir)
                ckpt = self._ckpt_path("tsptw", problem_size)
                wrapper = PIPDWrapper(
                    ckpt_path=ckpt,
                    total_nodes=problem_size + 1,
                    device=self.device,
                    pipd_dir=pipd_dir,
                )
                self._cache[key] = wrapper
                print(f"  POMO PRM: 加载 TSPTW (PIP-D) n={problem_size}")
            else:
                ModelClass, _ = self._classes[problem_type]
                model = ModelClass(**POMO_MODEL_PARAMS)
                ckpt = torch.load(self._ckpt_path(*key), map_location=self.device)
                model.load_state_dict(ckpt)
                model.to(self.device)
                model.eval()
                self._cache[key] = model
                print(f"  POMO PRM: 加载 {problem_type}_n{problem_size}")
        return self._cache[key]

    # ── 核心评估接口 ──────────────────────────────────────────────────

    @torch.no_grad()
    def compute_step_rewards(
        self,
        completion: str,
        instance: dict,
        problem_type: str,
    ) -> StepRewards:
        """
        计算合法前缀内的 per-step 增量 reward + depot 反事实 reward。

        新设计（与 terminal reward 解耦）：
            - parse 失败 → 返回空 StepRewards（trainer 端自动 mask=0，不参与 PRM loss）
            - 违规之后的位置：不返回任何 reward / 位置（trainer mask=0，不影响分母）
            - 覆盖不全：仍正常评估合法前缀那部分（terminal 已用 R_coverage 惩罚）
            - 完全合法：所有客户/depot 都给 PRM reward
        """
        n = instance["n"]

        # ── 1. 解析路线（失败直接返空 PRM 信号） ─────────────────────
        if problem_type in ("tsp", "tsptw"):
            route = self._parse_single_route(completion, n)
            if route is None:
                return self._empty(n)
            routes = [route]
        else:
            routes = self._parse_multi_route(completion, n)
            if routes is None:
                return self._empty(n)

        # ── 2. 提取完整步骤序列 + 客户/depot 索引 ────────────────────
        # 注：不再对 routes 排序——排序后 customer_seq 顺序与 completion 文本
        # 顺序不一致，会让 _find_token_positions 把奖励写到错的 token 上。
        # GRPO 跨 completion 同步骤组归一化按"document order 第 k 个客户"对比，
        # 没有排序也有意义（都是模型在同一 instance 上的第 k 个选择）。
        full_steps, customer_indices, depot_indices = self._extract_sequences(
            routes, problem_type
        )

        # ── 3. 前缀校验（约束模拟，违规处截断） ─────────────────────
        valid_full_length = self._validate_prefix(
            full_steps, instance, problem_type
        )

        # ── 4. POMO 批量前缀评估（只评估合法前缀） ──────────────────
        customer_seq = [full_steps[i] for i in customer_indices
                        if i < valid_full_length]
        valid_customers = len(customer_seq)

        pomo_values = self._batch_evaluate_prefixes(
            full_steps[:valid_full_length],
            customer_indices,
            instance, problem_type
        )

        # ── 5. 客户累积 reward（不再用增量差分） ────────────────────
        # 用 pomo_values[k] 而非 (pomo_values[k] - pomo_values[k-1])。
        # 理由：第 k 步 token 由其之前的所有 think + 选择决定，无法把"功劳"
        # 单独切到这一个 token 上。累积值反映"截至第 k 步的整段推理质量"，
        # GRPO 同步骤组归一化让各 completion 在每个时间点上 z-score 对比。
        customer_rewards = list(pomo_values[:valid_customers])

        # ── 6. Depot 反事实 reward ──────────────────────────────────
        depot_rewards = self._compute_depot_counterfactual(
            full_steps[:valid_full_length],
            depot_indices,
            instance, problem_type,
        )

        # ── 7. 字符位置映射 ─────────────────────────────────────────
        cust_positions  = self._find_token_positions(completion, customer_seq)
        depot_positions = self._find_depot_positions(
            completion, depot_indices, full_steps, valid_full_length
        )

        return StepRewards(
            customer_rewards=customer_rewards,
            depot_rewards=depot_rewards,
            customer_token_positions=cust_positions,
            depot_token_positions=depot_positions,
            n=n,
            covered=valid_customers,
        )

    # ── 解析 ──────────────────────────────────────────────────────────

    def _parse_single_route(self, text: str, n: int):
        from utils.parse import parse_single_route
        return parse_single_route(text, n)

    def _parse_multi_route(self, text: str, n: int):
        from utils.parse import parse_multi_route
        return parse_multi_route(text, n)

    # ── 序列提取 ──────────────────────────────────────────────────────

    def _extract_sequences(self, routes, problem_type):
        """
        从路线列表提取完整步骤序列、客户索引、depot 索引。

        Returns:
            full_steps:       完整步骤序列 (含 depot)
            customer_indices: full_steps 中客户节点的索引
            depot_indices:    full_steps 中 depot 回访的索引（不含首个 depot）
        """
        if problem_type in ("tsp", "tsptw"):
            # TSP/TSPTW: 单路线,去掉尾部 0 (隐式闭合),客户为非零节点,无 depot 回访
            route = routes[0]
            if route[-1] == 0 and len(route) > 1:
                full_steps = route[:-1]
            else:
                full_steps = route[:]
            customer_indices = [i for i, v in enumerate(full_steps) if v != 0]
            depot_indices = []
        else:
            # CVRP / VRPTW: 展平多路线
            full_steps = []
            for route in routes:
                full_steps.extend(route)
            # 去掉连续重复的 depot（路线衔接处 ...0, 0... → ...0...）
            deduped = [full_steps[0]]
            for v in full_steps[1:]:
                if v == 0 and deduped[-1] == 0:
                    continue
                deduped.append(v)
            full_steps = deduped

            customer_indices = [i for i, v in enumerate(full_steps) if v != 0]
            # depot 回访：不含首个 depot（位置 0）
            depot_indices = [i for i, v in enumerate(full_steps)
                            if v == 0 and i > 0]

        return full_steps, customer_indices, depot_indices

    # ── 前缀校验 ──────────────────────────────────────────────────────

    def _validate_prefix(self, full_steps, instance, problem_type) -> int:
        """
        模拟约束状态，返回合法步骤序列长度。在第一个违规处截断。
        """
        n = instance["n"]
        visited = set()
        valid_full_length = 0

        # CVRP 状态
        load = 1.0
        demands = instance.get("demands")
        capacity = instance.get("capacity", 1.0)

        # VRPTW 状态
        current_time = 0.0
        coords = np.array(instance["coords"]) if "coords" in instance else None
        tw = np.array(instance["time_windows"]) if "time_windows" in instance else None

        prev_node = None

        for i, node in enumerate(full_steps):
            if node == 0:
                if problem_type in ("tsp", "tsptw"):
                    # TSP/TSPTW: 只允许首位 depot,不可中途回访
                    if 0 in visited:
                        break
                    visited.add(0)
                else:
                    # CVRP/VRPTW: depot 可重复访问，重置状态
                    if problem_type == "cvrp":
                        load = 1.0
                    elif problem_type == "vrptw":
                        current_time = 0.0
                prev_node = node
                valid_full_length = i + 1
                continue

            # 客户节点检查
            if node in visited:
                break  # 重复节点，截断
            if node < 1 or node > n:
                break  # 越界，截断

            # CVRP 容量检查（容差与 POMO env 一致：1e-5）
            if problem_type == "cvrp" and demands is not None:
                d = demands[node] if node < len(demands) else 0
                if load + 1e-5 < d:
                    break
                load -= d

            # VRPTW / TSPTW 时间窗检查 (TSPTW 无 depot 重置,时间一直累积)
            if problem_type in ("vrptw", "tsptw") and coords is not None and tw is not None:
                if prev_node is not None:
                    travel = float(np.linalg.norm(
                        coords[node] - coords[prev_node]))
                    arrival = current_time + travel
                    if arrival > tw[node][1]:
                        break
                    current_time = max(arrival, tw[node][0])

            visited.add(node)
            valid_full_length = i + 1
            prev_node = node

        # TSPTW: 在硬约束验证之上加 PIP-D foresight 截断
        if problem_type == "tsptw":
            pipd = self._get_model("tsptw", n)
            fs_idx = pipd.foresight_check(instance, full_steps)
            if fs_idx is not None:
                # PIP-D 在 fs_idx 这一步判定不可行: 保留 [0:fs_idx] 为合法
                valid_full_length = min(valid_full_length, fs_idx)

        return valid_full_length

    # ── POMO 批量前缀评估 ─────────────────────────────────────────────

    def _batch_evaluate_prefixes(self, valid_steps, customer_indices,
                                 instance, problem_type):
        """
        对每个客户前缀做 POMO/PIP-D 评估，返回 value 列表。
        pomo_values[k] = value(prefix 到第 k 个客户) = -total_distance (越大越好)
        """
        n = instance["n"]
        # 筛选落在合法范围内的客户索引
        valid_cust_idx = [ci for ci in customer_indices if ci < len(valid_steps)]
        if not valid_cust_idx:
            return []

        num_prefixes = len(valid_cust_idx)
        prefix_lengths = [valid_cust_idx[k] + 1 for k in range(num_prefixes)]

        # TSPTW: 走 PIP-D wrapper 的 batch_rollout,接口不同于 POMO env
        if problem_type == "tsptw":
            pipd = self._get_model("tsptw", n)
            return pipd.batch_rollout(instance,
                                      valid_prefix=list(valid_steps),
                                      prefix_lengths=prefix_lengths)

        model = self._get_model(problem_type, n)

        # 构造 POMO 环境数据
        if problem_type == "tsp":
            problem_size = n + 1  # depot + n customers
            env = self._setup_tsp_env(instance, num_prefixes, problem_size)
        elif problem_type == "cvrp":
            problem_size = n
            env = self._setup_cvrp_env(instance, num_prefixes, problem_size)
        elif problem_type == "vrptw":
            problem_size = n
            env = self._setup_vrptw_env(instance, num_prefixes, problem_size)

        # prefix_lengths 已经在函数顶部算过,这里直接用
        max_prefix_len = valid_cust_idx[-1] + 1

        # Reset env
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, _, _ = env.pre_step()

        # 逐步执行：强制前缀 + POMO greedy 补全
        done = False
        step = 0
        while not done:
            # model greedy selection
            selected, _ = model(state)  # (batch, pomo=1)

            # 对仍在前缀范围内的 batch 元素，覆盖为强制节点
            for k in range(num_prefixes):
                if step < prefix_lengths[k]:
                    selected[k, 0] = valid_steps[step]

            state, reward, done = env.step(selected)
            step += 1

        # reward shape: (num_prefixes, 1) → 取负得到距离
        pomo_values = reward[:, 0].cpu().tolist()  # -distance → value
        return pomo_values

    # ── Depot 反事实评估 ──────────────────────────────────────────────

    def _compute_depot_counterfactual(self, valid_steps, depot_indices,
                                      instance, problem_type):
        """
        对每个 depot 回访点，计算反事实 advantage：

          val_with    = -dist(prefix 含 depot + POMO 自由补全)   # 越大越好
          val_without = -dist(prefix 不含 depot, 屏蔽 0 + POMO 补全)
          depot_reward = val_with - val_without

        语义（与标准 GRPO advantage 一致）：
          正值 = 回 depot 是对的（advantage 正 → 鼓励该选择）
          负值 = 不该回 depot

        特殊情况：屏蔽 0 后无节点可选 → 不回 depot 不可行 → depot 绝对必要
                  → 给 +DEPOT_MANDATORY_REWARD 强烈鼓励
        """
        DEPOT_MANDATORY_REWARD = 0.1

        if problem_type == "tsp" or not depot_indices:
            return []

        depot_rewards = []

        for di in depot_indices:
            if di >= len(valid_steps):
                break

            # 尾部 depot（后面没有客户了）→ reward = 0（无影响）
            has_customer_after = any(
                valid_steps[j] != 0 for j in range(di + 1, len(valid_steps))
            )
            if not has_customer_after:
                depot_rewards.append(0.0)
                continue

            # WITH depot: 前缀包含 depot 回访，POMO 自由补全
            prefix_with = valid_steps[:di + 1]
            val_with = self._single_prefix_evaluate(
                prefix_with, instance, problem_type
            )

            # WITHOUT depot: 前缀到 depot 前一个节点，屏蔽 node 0
            prefix_without = valid_steps[:di]
            val_without = self._single_prefix_evaluate(
                prefix_without, instance, problem_type,
                mask_node_at_first_free_step=0
            )

            if val_without is None:
                # 屏蔽 0 后无节点可选 → depot 回访是必要的，强鼓励
                depot_rewards.append(DEPOT_MANDATORY_REWARD)
            else:
                depot_rewards.append(val_with - val_without)

        return depot_rewards

    def _single_prefix_evaluate(self, prefix, instance, problem_type,
                                mask_node_at_first_free_step=None):
        """
        单个前缀的 POMO 评估，返回 -total_distance。

        Args:
            mask_node_at_first_free_step: 在前缀结束后的第一个自由步骤中，
                额外屏蔽该节点。如果屏蔽后无节点可选，返回 None。
        """
        n = instance["n"]
        model = self._get_model(problem_type, n)
        batch_size = 1

        if problem_type == "tsp":
            env = self._setup_tsp_env(instance, batch_size, n + 1)
        elif problem_type == "cvrp":
            env = self._setup_cvrp_env(instance, batch_size, n)
        elif problem_type == "vrptw":
            env = self._setup_vrptw_env(instance, batch_size, n)

        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, _, _ = env.pre_step()

        done = False
        step = 0
        first_free_step = len(prefix)

        while not done:
            selected, _ = model(state)

            # 强制前缀
            if step < len(prefix):
                selected[0, 0] = prefix[step]
            elif step == first_free_step and mask_node_at_first_free_step is not None:
                # 在第一个自由步骤屏蔽指定节点
                mask = state.ninf_mask.clone()
                mask[0, 0, mask_node_at_first_free_step] = float('-inf')

                # 检查是否还有可选节点
                if (mask[0, 0] == float('-inf')).all():
                    return None  # 无节点可选

                # 用屏蔽后的 mask 重新让模型选择
                state_masked = type(state)(**{
                    k: v for k, v in state.__dict__.items()
                })
                state_masked.ninf_mask = mask
                selected, _ = model(state_masked)

            state, reward, done = env.step(selected)
            step += 1

        return reward[0, 0].item()

    # ── POMO 环境构造 ─────────────────────────────────────────────────

    def _setup_tsp_env(self, instance, batch_size, problem_size):
        _, EnvClass = self._classes["tsp"]
        env = EnvClass(problem_size, pomo_size=1)
        env.batch_size = batch_size
        env.device = self.device

        coords = torch.tensor(instance["coords"], dtype=torch.float32,
                              device=self.device)
        # (n+1, 2) → (batch, n+1, 2)
        env.node_xy = coords.unsqueeze(0).expand(batch_size, -1, -1)

        env.BATCH_IDX = torch.arange(batch_size, device=self.device
                                     )[:, None].expand(batch_size, 1)
        env.POMO_IDX = torch.zeros(batch_size, 1, dtype=torch.long,
                                   device=self.device)

        from source.envs.tsp_env import Reset_State, Step_State
        env.reset_state = Reset_State(node_xy=env.node_xy)
        env.step_state = Step_State(BATCH_IDX=env.BATCH_IDX,
                                    POMO_IDX=env.POMO_IDX)
        return env

    def _setup_cvrp_env(self, instance, batch_size, problem_size):
        _, EnvClass = self._classes["cvrp"]
        env = EnvClass(problem_size, pomo_size=1)
        env.batch_size = batch_size
        env.device = self.device

        coords = np.array(instance["coords"])
        demands = np.array(instance["demands"])

        depot_xy = torch.tensor(coords[0:1], dtype=torch.float32,
                                device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        node_xy = torch.tensor(coords[1:], dtype=torch.float32,
                               device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        node_demand = torch.tensor(demands[1:], dtype=torch.float32,
                                   device=self.device).unsqueeze(0).expand(batch_size, -1)

        env.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(batch_size, 1, device=self.device)
        env.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        env.BATCH_IDX = torch.arange(batch_size, device=self.device
                                     )[:, None].expand(batch_size, 1)
        env.POMO_IDX = torch.zeros(batch_size, 1, dtype=torch.long,
                                   device=self.device)

        from source.envs.cvrp_env import Reset_State, Step_State
        env.reset_state = Reset_State(depot_xy=depot_xy, node_xy=node_xy,
                                      node_demand=node_demand)
        env.step_state = Step_State(BATCH_IDX=env.BATCH_IDX,
                                    POMO_IDX=env.POMO_IDX)
        return env

    def _setup_vrptw_env(self, instance, batch_size, problem_size):
        _, EnvClass = self._classes["vrptw"]
        env = EnvClass(problem_size, pomo_size=1)
        env.batch_size = batch_size
        env.device = self.device

        coords = np.array(instance["coords"])
        tw = np.array(instance["time_windows"])

        depot_xy = torch.tensor(coords[0:1], dtype=torch.float32,
                                device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        node_xy = torch.tensor(coords[1:], dtype=torch.float32,
                               device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        tw_start = torch.tensor(tw[:, 0], dtype=torch.float32,
                                device=self.device).unsqueeze(0).expand(batch_size, -1)
        tw_end = torch.tensor(tw[:, 1], dtype=torch.float32,
                              device=self.device).unsqueeze(0).expand(batch_size, -1)

        env.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        env.tw_start = tw_start
        env.tw_end = tw_end

        env.BATCH_IDX = torch.arange(batch_size, device=self.device
                                     )[:, None].expand(batch_size, 1)
        env.POMO_IDX = torch.zeros(batch_size, 1, dtype=torch.long,
                                   device=self.device)

        from source.envs.vrptw_env import Reset_State, Step_State
        env.reset_state = Reset_State(
            depot_xy=depot_xy, node_xy=node_xy,
            tw_start=tw_start, tw_end=tw_end,
        )
        env.step_state = Step_State(BATCH_IDX=env.BATCH_IDX,
                                    POMO_IDX=env.POMO_IDX)
        return env

    # ── 字符位置映射 ─────────────────────────────────────────────────
    # 返回的都是 completion 文本中的**字符位置**，
    # 由 trainer 用 offset_mapping 转换为 token 索引。
    #
    # 重要：不能直接 re.finditer(r'\d+') 抓所有数字，因为 'Route 1:' 里的
    # 头部数字 1 会被当成 customer 1 的访问。下面用"路线内容区间"做隔离：
    # 多路线问题先定位每条 'Route N:' 的覆盖范围，只在 header 之后到下一个
    # header 之前的内容里抓节点；单路线问题（无 header）整段都是内容区。

    _HEADER_PATTERN = re.compile(r'(?:route|路线)\s*\d+[：:]', re.IGNORECASE)
    _NUMBER_PATTERN = re.compile(r'(?<!\d)(\d+)(?!\d)')

    def _content_numbers(self, route_text):
        """
        返回路径内容区间内的所有独立数字 (value, char_pos in route_text)。
        排除 'Route N:' 头部数字。
        """
        headers = list(self._HEADER_PATTERN.finditer(route_text))
        if headers:
            ranges = [
                (headers[i].end(),
                 headers[i + 1].start() if i + 1 < len(headers) else len(route_text))
                for i in range(len(headers))
            ]
        else:
            # 单路线无头部 → 整文本作为一个内容区
            ranges = [(0, len(route_text))]

        numbers = []
        for cstart, cend in ranges:
            for m in self._NUMBER_PATTERN.finditer(route_text[cstart:cend]):
                numbers.append((int(m.group(1)), cstart + m.start()))
        return numbers

    def _find_token_positions(self, completion, nodes):
        """按顺序定位每个节点数字在 completion 中的字符位置。"""
        route_text, route_offset = self._get_route_text(completion)
        all_numbers = self._content_numbers(route_text)

        positions = []
        num_idx = 0
        for node in nodes:
            found = False
            while num_idx < len(all_numbers):
                num_val, char_pos = all_numbers[num_idx]
                num_idx += 1
                if num_val == node:
                    positions.append(route_offset + char_pos)
                    found = True
                    break
            if not found:
                positions.append(-1)
        return positions

    def _find_depot_positions(self, completion, depot_indices,
                              full_steps, valid_full_length):
        """
        定位 depot 回访（中间/尾部的 0）在 completion 文本中的字符位置。
        起始 depot 不在 depot_indices 中（由 _extract_sequences 保证）。
        """
        route_text, route_offset = self._get_route_text(completion)
        all_numbers = self._content_numbers(route_text)

        positions = []
        valid_depot_indices = set(di for di in depot_indices if di < valid_full_length)

        num_idx = 0
        for step_i, node in enumerate(full_steps[:valid_full_length]):
            while num_idx < len(all_numbers):
                num_val, char_pos = all_numbers[num_idx]
                num_idx += 1
                if num_val == node:
                    if step_i in valid_depot_indices:
                        positions.append(route_offset + char_pos)
                    break
        return positions

    def _get_route_text(self, completion):
        """获取 </think> 之后的路线文本区域，返回 (text, char_offset)。"""
        think_end = completion.rfind("</think>")
        if think_end != -1:
            offset = think_end + len("</think>")
            return completion[offset:], offset
        return completion, 0

    # ── 空 PRM（解析失败时） ──────────────────────────────────────────

    def _empty(self, n):
        """parse 失败时的空信号：trainer 端 mask=0 → 不参与 PRM loss 分子分母。"""
        return StepRewards(
            customer_rewards=[],
            depot_rewards=[],
            customer_token_positions=[],
            depot_token_positions=[],
            n=n,
            covered=0,
        )
