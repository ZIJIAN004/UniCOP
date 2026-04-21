"""
PIP-D wrapper for UniCOP-Reason TSPTW PRM.

封装 PIP-D (Bi et al., NeurIPS 2024) 的 SINGLEModel + TSPTWEnv,
为 POMOPRM 提供 TSPTW 的两个核心信号:

1. foresight_check(instance, prefix_nodes):
   逐步前进 prefix,每步读 aux decoder 的 probs_sl[next_node]。
   > 0.5 (= PIP-D 的 decision_boundary) 视为"PIP-D 判该节点不可行",
   返回首次截断的 prefix 索引 (或 None 全部通过)。

2. batch_rollout(instance, valid_prefix, prefix_lengths):
   POMO-style 批量 rollout: 每个 batch slot 对应一个 prefix 长度,
   强制前缀 + PIP-D greedy 补全,返回每个 prefix 的 final tour length。

PIP-D 的 problem_size = 总节点数 (含 depot); UniCOP 的 n = 客户数。
转换: pipd_problem_size = instance["n"] + 1。
"""
import os
import sys
import torch
import numpy as np


# ── sys.path 注入 (让 SINGLEModel / TSPTWEnv 能 import) ──────────────────────

_PIPD_DIR_CACHE: str | None = None


def _resolve_pipd_dir(explicit: str | None = None) -> str:
    """
    定位 PIP-D baseline/POMO+PIP 目录。优先级:
      1. 显式传入
      2. 环境变量 PIPD_POMO_DIR
      3. 代码仓库相对路径 (../../PIP-D baseline/POMO+PIP)
    """
    global _PIPD_DIR_CACHE
    if _PIPD_DIR_CACHE and os.path.isdir(_PIPD_DIR_CACHE):
        return _PIPD_DIR_CACHE

    candidates = []
    if explicit:
        candidates.append(explicit)
    env_var = os.environ.get("PIPD_POMO_DIR")
    if env_var:
        candidates.append(env_var)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "PIP-D baseline", "POMO+PIP")))

    for cand in candidates:
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "models", "SINGLEModel.py")):
            _PIPD_DIR_CACHE = cand
            return cand

    raise FileNotFoundError(
        "找不到 PIP-D baseline/POMO+PIP 目录。请设置环境变量 PIPD_POMO_DIR "
        "或在 PIPDWrapper.load(...) 显式传入 pipd_dir=...\n"
        f"已尝试: {candidates}"
    )


def _ensure_pipd_import(pipd_dir: str):
    if pipd_dir not in sys.path:
        sys.path.insert(0, pipd_dir)


# ── PIP-D 模型 / 环境超参 ─────────────────────────────────────────────────────
# 对应 train.py 默认值 + "POMO_star_PIP-D" 模式 (pip_decoder=True)

PIPD_MODEL_PARAMS = {
    "embedding_dim":          128,
    "sqrt_embedding_dim":     128 ** 0.5,
    "encoder_layer_num":      6,
    "decoder_layer_num":      1,
    "qkv_dim":                16,
    "head_num":               8,
    "logit_clipping":         10,
    "ff_hidden_dim":          512,
    "norm":                   "instance",
    "norm_loc":               "norm_last",
    "eval_type":              "argmax",
    "problem":                "TSPTW",
    "pip_decoder":            True,
    "tw_normalize":           True,
    "decision_boundary":      0.5,
    "detach_from_encoder":    False,
    "W_q_sl":                 True,
    "W_out_sl":               True,
    "W_kv_sl":                True,
    "use_ninf_mask_in_sl_MHA": False,
    "generate_PI_mask":       False,
}


def _pipd_env_params(problem_size: int, pomo_size: int,
                     device: torch.device, k_sparse: int = 10_000) -> dict:
    """k_sparse > problem_size 禁用稀疏化 (对 n<=100 恒成立)。"""
    return {
        "problem_size": problem_size,
        "pomo_size":    pomo_size,
        "hardness":     "easy",
        "device":       device,
        "k_sparse":     k_sparse,
    }


class PIPDWrapper:
    """PIP-D 推断封装。一个 wrapper 对应一个 (problem_size, ckpt)。"""

    def __init__(self, ckpt_path: str, total_nodes: int,
                 device: torch.device, pipd_dir: str):
        _ensure_pipd_import(pipd_dir)
        from models.SINGLEModel import SINGLEModel  # type: ignore  # noqa
        from envs.TSPTWEnv import TSPTWEnv           # type: ignore  # noqa

        self._SINGLEModel = SINGLEModel
        self._TSPTWEnv    = TSPTWEnv

        self.device       = device
        self.total_nodes  = total_nodes   # = n + 1 (含 depot)
        self.ckpt_path    = ckpt_path

        model_params = dict(PIPD_MODEL_PARAMS)
        self.model = SINGLEModel(**model_params).to(device)
        self.model.eval()

        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        # 硬性检查: PIP-D 的 aux decoder 相关权重必须存在,否则 ckpt 指错
        # (比如指到了 POMO_star 而非 POMO_star_PIP-D),probs_sl 会是纯噪声
        sl_missing = [k for k in missing if "_sl" in k or "sl_" in k]
        if sl_missing:
            raise RuntimeError(
                f"PIP-D ckpt 缺少 aux decoder 权重 ({len(sl_missing)} 个),"
                f"ckpt 路径可能指错 (应为 POMO_star_PIP-D,不是 POMO_star)。\n"
                f"ckpt: {ckpt_path}\n"
                f"前几个缺失: {sl_missing[:5]}"
            )
        if missing or unexpected:
            print(f"[PIPDWrapper] ckpt load (strict=False): "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")

    # ── 实例格式转换 ─────────────────────────────────────────────────────────

    def _instance_to_tensors(self, instance: dict, batch_size: int):
        """
        UniCOP instance (coords [0,1]²、tw in tn units) → PIP-D env inputs。
        UniCOP 已经是 normalize 后的尺度 (与 PIP-D 训练时 /100 后一致),
        所以 env.load_problems 用 normalize=False 直接喂。
        """
        n = instance["n"]
        assert n + 1 == self.total_nodes, \
            f"instance n+1={n+1} 与 wrapper total_nodes={self.total_nodes} 不匹配"

        coords = np.asarray(instance["coords"], dtype=np.float32)          # (n+1, 2)
        tw     = np.asarray(instance["time_windows"], dtype=np.float32)    # (n+1, 2)

        # ── depot tw_end 紧缩 ─────────────────────────────────────────────
        # PIP-D 训练时用 normalize=True,内置了:
        #   tw_end[:, 0] = max over customers of (dist_depot_to_customer + customer_tw_end)
        # 这也是 "归一化分母",如果 depot tw_end 不紧 (UniCOP 是 1e9),
        # pre_forward 的 tw_normalize 会把所有客户 tw 归一化到 ~0,模型看不到 TW。
        tw_end_np = tw[:, 1].copy()
        depot_xy  = coords[0]
        cust_xy   = coords[1:]
        cust_twe  = tw_end_np[1:]
        dist_to_cust = np.linalg.norm(cust_xy - depot_xy, axis=1)
        tight_depot_twe = float((dist_to_cust + cust_twe).max())
        tw_end_np[0] = tight_depot_twe

        node_xy      = torch.from_numpy(coords).to(self.device).unsqueeze(0)  # (1, n+1, 2)
        service_time = torch.zeros(1, n + 1, dtype=torch.float32, device=self.device)
        tw_start     = torch.from_numpy(tw[:, 0]).to(self.device).unsqueeze(0)
        tw_end       = torch.from_numpy(tw_end_np.astype(np.float32)).to(self.device).unsqueeze(0)

        if batch_size > 1:
            node_xy      = node_xy.expand(batch_size, -1, -1).contiguous()
            service_time = service_time.expand(batch_size, -1).contiguous()
            tw_start     = tw_start.expand(batch_size, -1).contiguous()
            tw_end       = tw_end.expand(batch_size, -1).contiguous()

        return node_xy, service_time, tw_start, tw_end

    # ── API 1: foresight check ────────────────────────────────────────────────

    @torch.no_grad()
    def foresight_check(self, instance: dict, prefix_nodes: list[int]) -> int | None:
        """
        沿 prefix 走 pomo_size=1 的单条 rollout,每步前用 aux decoder 预测
        probs_sl[next_node]。首次 > 0.5 返回该 prefix 索引;全部通过返回 None。

        prefix_nodes 应包含从 depot (0) 开始的完整步骤序列。
        """
        env = self._TSPTWEnv(**_pipd_env_params(self.total_nodes, 1, self.device))
        problems = self._instance_to_tensors(instance, batch_size=1)
        env.load_problems(batch_size=1, problems=problems,
                          aug_factor=1, normalize=False)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        state, _, _ = env.pre_step()

        for i, node in enumerate(prefix_nodes):
            # selected_count==0 时 depot 是固定的 (见 SINGLEModel.forward),
            # 没有 probs_sl 可读,直接 advance 一步。
            if state.selected_count == 0:
                selected = torch.tensor([[0]], dtype=torch.long, device=self.device)
                state, _, done, _ = env.step(selected)
                if node != 0:
                    # 第 0 步不是 depot —— 这是数据层面错误,走兜底截断
                    return i
                if done:
                    break
                continue

            # 先拿 probs_sl (no_select_prob=True 只返回 probs_sl,不跑采样)
            # no_sigmoid=True 拿 raw logit,与 PIP-D Tester 的推断路径严格对齐,
            # 决策边界相当于 sigmoid(x) > 0.5 ⟺ x > 0
            probs_sl = self.model(
                state, no_select_prob=True, no_sigmoid=True,
                tw_end=env.node_tw_end,
            )
            # shape: (1, 1, problem_size)
            if 0 <= node < self.total_nodes:
                p_logit = probs_sl[0, 0, node].item()
                if p_logit > 0.0:   # 等价于 sigmoid(p) > 0.5
                    return i
            else:
                # node 越界,当作判不可行
                return i

            selected = torch.tensor([[node]], dtype=torch.long, device=self.device)
            state, _, done, _ = env.step(selected)
            if done:
                break

        return None

    # ── API 2: batch rollout (POMO-style PRM value) ──────────────────────────

    @torch.no_grad()
    def batch_rollout(self, instance: dict, valid_prefix: list[int],
                      prefix_lengths: list[int]) -> list[float]:
        """
        对每个 prefix 长度 L_k,前 L_k 步强制走 valid_prefix[0:L_k],
        之后 PIP-D greedy 走到结束。返回每个 prefix 的 -total_distance (value)。

        Args:
            valid_prefix: 合法前缀的完整步骤序列 (含 depot 起点)
            prefix_lengths: 每个 batch slot 要强制的 prefix 长度

        Returns:
            list of floats, 长度 = len(prefix_lengths), value = -distance
        """
        num_prefixes = len(prefix_lengths)
        if num_prefixes == 0:
            return []

        env = self._TSPTWEnv(**_pipd_env_params(self.total_nodes, 1, self.device))
        problems = self._instance_to_tensors(instance, batch_size=num_prefixes)
        env.load_problems(batch_size=num_prefixes, problems=problems,
                          aug_factor=1, normalize=False)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        state, _, _ = env.pre_step()

        step = 0
        max_prefix = max(prefix_lengths)
        reward = None
        done = False
        while not done:
            if state.selected_count == 0:
                # 第 0 步所有 batch 都选 depot (0)
                selected = torch.zeros(num_prefixes, 1, dtype=torch.long, device=self.device)
            else:
                # 先让 PIP-D 自己预测,再对"仍在 prefix 区间"的 slot 覆盖
                selected, _ = self.model(
                    state, selected=None, pomo=False,
                    use_predicted_PI_mask=True, no_sigmoid=True,
                    tw_end=env.node_tw_end,
                )
                # shape: (num_prefixes, 1)
                for k in range(num_prefixes):
                    if step < prefix_lengths[k] and step < len(valid_prefix):
                        selected[k, 0] = valid_prefix[step]

            state, reward, done, _ = env.step(
                selected,
                use_predicted_PI_mask=True,
            )
            step += 1

        # reward 在 TSPTWEnv 中: done 时为 -total_distance (越大越好)
        # shape: (num_prefixes, pomo_size=1)
        if reward is None:
            return [0.0] * num_prefixes
        values = reward.detach().cpu().view(num_prefixes, -1).max(dim=1).values.tolist()
        return values
