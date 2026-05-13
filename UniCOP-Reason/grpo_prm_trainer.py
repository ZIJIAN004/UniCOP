"""
GRPOPRMTrainer: 继承 trl.GRPOTrainer，Per-Customer 增量 PRM + 段内广播。

设计（A_feasibility + A_outcome + Per-Customer PRM → per-token advantage）：
    1. A_feasibility：parse / (coverage × constraint) / format 三项加权
       - cov 与 con 用乘积合并: cov=0 时 con 无效, 防"丢覆盖换约束"hack
       → 全组 num_gen 条做 z-score（零均值），GRPO 标准 baseline 中心化
    2. A_outcome：可行子集 z-score(-distance)，子集 ≥ 2 才算，不可行 = 0
    3. A_out = A_feasibility_norm + A_outcome_norm → 所有 token 共享（整组零均值）
    4. A_proc：per-customer 增量 PRM，段内广播到 [R{r},{s}] 标记对应的推理段
       - 仅可行 completion 计算；不可行 A_proc = 0
       - per-customer 跨 completion z-score 归一化
    5. per-token advantage = A_out + α · A_proc（步骤段内）/ A_out（非步骤区域）
    6. Loss：单一 DAPO token-level loss（按 token 数加权）
"""

import json
import time
import numpy as np
import torch
from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer

from pomo_prm import POMOPRM, ThinkPRMResult
from terminal_reward import compute_terminal_components, is_fully_feasible
from foarl_reward import compute_foarl_reward
from ref_solver import solve_reference
from config import config

from problems.tsp import TSP
from problems.cvrp import CVRP
from problems.vrptw import VRPTW
from problems.tsptw import TSPTW
from problems.tspdl import TSPDL

_PROBLEM_OBJS = {
    "tsp": TSP(), "cvrp": CVRP(), "vrptw": VRPTW(),
    "tsptw": TSPTW(), "tspdl": TSPDL(),
}


class GRPOPRMTrainer(GRPOTrainer):

    def __init__(self, pomo_prm=None, problem_types: list[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.pomo_prm = pomo_prm
        self.problem_types = problem_types or []

        if self.args.num_generations < 2:
            raise ValueError(
                f"num_generations 必须 >= 2 才能做 GRPO 组内对比；"
                f"当前 = {self.args.num_generations}"
            )

        # ── 第 1 个 step 各阶段耗时打印 (诊断瓶颈用) ────────────────────
        # 三段: rollout (vLLM 生成) / reward (resample + advantages) /
        # forward+backward (含 NCCL AllReduce). 仅 rank 0 第 1 个 step 打印.
        # gradient_accumulation_steps > 1 时 training_step 会被调多次,
        # fwd_bwd 累加直到达到 grad_accum_steps 才视为完整 step.
        self._timing_log: dict[str, float] = {}
        self._timing_fwd_bwd_count: int = 0
        self._timing_done: bool = False

    def _get_train_sampler(self, dataset=None):
        return super()._get_train_sampler()

    # ── 多卡聚合工具 ──────────────────────────────────────────────────

    def _gather_mean(self, value) -> float:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), device=self.accelerator.device)
        if value.dim() == 0:
            value = value.unsqueeze(0)
        gathered = self.accelerator.gather(value)
        return gathered.float().mean().item()

    # ══════════════════════════════════════════════════════════════════
    #  生成 + 三信号解耦
    # ══════════════════════════════════════════════════════════════════

    def _generate_and_score_completions(self, inputs):
        _record = not self._timing_done
        if _record and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.time()

        batch = super()._generate_and_score_completions(inputs)

        if _record and torch.cuda.is_available():
            torch.cuda.synchronize()
        if _record:
            self._timing_log["rollout"] = time.time() - _t0

        completion_ids  = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        B, T = completion_ids.shape
        num_gen = self.args.num_generations

        # 展开 problem_data / problem_type
        if isinstance(inputs, list):
            problem_data_raw = [x["problem_data"] for x in inputs]
            problem_type_raw = [x["problem_type"] for x in inputs]
        elif isinstance(inputs, dict):
            problem_data_raw = inputs["problem_data"]
            problem_type_raw = inputs["problem_type"]
        else:
            raise TypeError(f"意外的 inputs 类型: {type(inputs)}")

        num_prompts = B // num_gen
        n_raw = len(problem_data_raw)

        if n_raw == num_prompts:
            problem_data_list = [pd for pd in problem_data_raw for _ in range(num_gen)]
            problem_type_list = [pt for pt in problem_type_raw for _ in range(num_gen)]
        elif n_raw == B:
            problem_data_list = list(problem_data_raw)
            problem_type_list = list(problem_type_raw)
        else:
            raise ValueError(
                f"problem_data 长度 {n_raw} 既不是 num_prompts={num_prompts} "
                f"也不是 B={B}，无法判断是否展开。"
            )

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        instances = [self._deserialize_instance(pd) for pd in problem_data_list]

        # ── 一次性诊断: dump 第一条 completion 看 think 段实际格式 ────
        # PRM_DIAG 显示 customer_groups_with_signal=0/0 但 hybrid SFT 训练
        # 数据 [R*,*] 标记完整, 怀疑 RL 阶段实际生成的 completion 格式跟
        # 训练数据有差异 (如 <think> 标签缺失/[R*,*] 形态变化). 抓一条看.
        if not getattr(self, "_completion_dump_logged", False):
            if self.accelerator.is_main_process and len(completions_text) > 0:
                import re as _re
                txt = completions_text[0]
                has_think_open  = "<think>" in txt
                has_think_close = "</think>" in txt
                te = txt.find("</think>")
                # 抓 [R*,*] 类标记 (宽松正则)
                br_matches = _re.findall(r'\[R?\d+\s*,\s*\d+\]', txt)
                print(
                    f"\n[COMPLETION_DUMP idx=0] len={len(txt)} "
                    f"has<think>={has_think_open} has</think>={has_think_close} "
                    f"bracket_count={len(br_matches)}\n"
                    f"--- 前 800 char ---\n{txt[:800]}\n"
                    f"--- think 末尾前 600 char (</think> 之前) ---\n"
                    f"{txt[max(0, te-600):te] if te > 0 else '(no </think>)'}\n"
                    f"--- 末尾 300 char ---\n{txt[-300:]}\n"
                    f"--- 前 10 个 [R*,*] 标记 ---\n{br_matches[:10]}",
                    flush=True,
                )
            self._completion_dump_logged = True

        if config.reward_mode == "foarl":
            advantages = self._build_foarl_advantages(
                completions_text, instances, problem_type_list,
                B, num_gen, device=completion_ids.device,
            )
            # FOARL 返回 (B,) → 扩展到 (B, T) 保持接口一致
            advantages = advantages.unsqueeze(-1).expand(-1, T).contiguous()
        else:
            # 可行性重采样：每组至少 2 条可行解
            self._resample_infeasible(
                batch, completions_text, instances, problem_type_list, num_gen
            )

            # Per-Customer PRM + A_feasibility + A_outcome → per-token (B, T)
            advantages = self._build_unified_advantages(
                completions_text, instances, problem_type_list,
                B, num_gen, T, device=completion_ids.device,
                prompt_ids=batch["prompt_ids"],
                prompt_mask=batch["prompt_mask"],
            )

        batch["advantages"] = advantages
        # 清理旧字段，避免父类或后续代码误用
        for key in ("terminal_advantages", "prm_advantages",
                     "prm_mask", "prm_denom"):
            batch.pop(key, None)

        if _record and torch.cuda.is_available():
            torch.cuda.synchronize()
        if _record:
            self._timing_log["reward"] = (
                time.time() - _t0 - self._timing_log["rollout"]
            )

        return batch

    # ── 可行性重采样 ─────────────────────────────────────────────────

    def _resample_infeasible(self, batch, completions_text, instances,
                              problem_type_list, num_gen):
        """每组可行解 < 2 时，通过 vLLM 重新生成替换不可行 completion，循环至 >= 2。

        注意: 单 rank B < num_gen 时 (BATCH_DIAG 实证场景), for g in range(0) 跳过,
        gather_object([]) 在所有 rank 同步空调用, 自然 no-op 安全退出.
        rank 内 group 不完整时本来 resample 判断也没意义, 跳过是正确行为.
        """
        _MAX_RETRIES = 5
        B, T = batch["completion_ids"].shape
        device = batch["completion_ids"].device
        pad_id = self.processing_class.pad_token_id or 0
        eos_id = self.processing_class.eos_token_id
        total_resampled = 0

        # ── 一次性诊断: 单 rank 看到的 B vs num_gen + prompt 分布 ──────
        # B // num_gen == 0 时, 下面的 for g 循环跑 0 次, resample/A_outcome/
        # A_feasibility z-score 全失效.
        # 跨 rank 比对 prompt_hash 区分:
        #   - hash 全不同  → 每 rank 看到独立 prompts (情况 A 或 B, 看 num_groups)
        #   - hash 全相同  → 同一 prompt 的 num_gen 被切到多 rank (情况 C, 需跨 rank gather)
        if not getattr(self, "_batch_diag_logged", False):
            rank = self.accelerator.process_index
            feas_per_group = []
            for g in range(B // num_gen):
                s = g * num_gen
                feas_per_group.append(sum(
                    1 for i in range(s, s + num_gen)
                    if is_fully_feasible(
                        completions_text[i], instances[i], problem_type_list[i]
                    )
                ))
            # 用 prompt_ids 前 32 token 求和当 hash, 跨 rank 比对
            p_ids = batch["prompt_ids"]
            first_prompt_hash = int(p_ids[0, :32].sum().item()) if B >= 1 else -1
            last_prompt_hash  = int(p_ids[-1, :32].sum().item()) if B >= 1 else -1
            distinct_in_rank  = len({
                int(p_ids[i, :32].sum().item()) for i in range(min(B, 16))
            })
            print(
                f"[BATCH_DIAG rank={rank}] B={B} num_gen={num_gen} "
                f"num_groups={B // num_gen}  feas_per_group={feas_per_group}  "
                f"prompt_hash_first={first_prompt_hash} prompt_hash_last={last_prompt_hash} "
                f"distinct_prompts_in_rank={distinct_in_rank}",
                flush=True,
            )
            self._batch_diag_logged = True

        # ── Step 门控: 前 resample_start_step 步跳过 resample ──────────
        # 训练初期可行率低 (<20%), resample 也大概率失败, 反复 vLLM 调用
        # 浪费时间. 等模型学到基本可行模式后再开 resample 救少数 outlier.
        # 所有 rank 同步跳过 (self.state.global_step 跨 rank 一致), 不会
        # collective deadlock.
        resample_start_step = getattr(config, "resample_start_step", 100)
        current_step = (
            self.state.global_step
            if hasattr(self, "state") and self.state is not None
            else 0
        )
        if current_step < resample_start_step:
            if not getattr(self, "_resample_skip_logged", False):
                if self.accelerator.is_main_process:
                    print(
                        f"[RESAMPLE_GATE] current_step={current_step} < "
                        f"resample_start_step={resample_start_step}, "
                        f"前期 resample 关闭 (模型可行率低时浪费 vLLM 时间)",
                        flush=True,
                    )
                self._resample_skip_logged = True
            return

        for _ in range(_MAX_RETRIES):
            local_requests = []
            for g in range(B // num_gen):
                s = g * num_gen
                infeasible = [
                    i for i in range(s, s + num_gen)
                    if not is_fully_feasible(
                        completions_text[i], instances[i], problem_type_list[i]
                    )
                ]
                if num_gen - len(infeasible) >= 2 or not infeasible:
                    continue
                p_ids = batch["prompt_ids"][s]
                p_mask = batch["prompt_mask"][s]
                prompt_text = self.processing_class.decode(
                    p_ids[p_mask.bool()], skip_special_tokens=False,
                )
                for idx in infeasible[:2]:
                    local_requests.append((idx, prompt_text))

            local_prompts = [p for _, p in local_requests]
            all_prompt_lists = gather_object(local_prompts)
            flat_prompts = [p for plist in all_prompt_lists for p in plist]

            if not flat_prompts:
                break

            if self.accelerator.is_main_process:
                new_ids_all = self.vllm_client.generate(
                    prompts=flat_prompts,
                    n=1,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding_regex=self.guided_decoding_regex,
                )
            else:
                new_ids_all = [None] * len(flat_prompts)
            broadcast_object_list(new_ids_all, from_process=0)

            counts = [len(plist) for plist in all_prompt_lists]
            rank = self.accelerator.process_index
            my_offset = sum(counts[:rank])
            my_new_ids = new_ids_all[my_offset:my_offset + counts[rank]]

            for j, (idx, _) in enumerate(local_requests):
                token_ids = my_new_ids[j]
                comp = torch.tensor(token_ids, dtype=torch.long, device=device)
                L = comp.shape[0]

                if L < T:
                    comp = torch.cat([
                        comp,
                        torch.full((T - L,), pad_id,
                                   dtype=torch.long, device=device),
                    ])
                    mask = torch.cat([
                        torch.ones(L, dtype=torch.long, device=device),
                        torch.zeros(T - L, dtype=torch.long, device=device),
                    ])
                else:
                    comp = comp[:T]
                    mask = torch.ones(T, dtype=torch.long, device=device)

                if eos_id is not None:
                    eos_pos = (comp == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        mask[eos_pos[0].item() + 1:] = 0

                batch["completion_ids"][idx] = comp
                batch["completion_mask"][idx] = mask
                completions_text[idx] = self.processing_class.decode(
                    token_ids, skip_special_tokens=True,
                )

            total_resampled += len(local_requests)

        for g in range(B // num_gen):
            s = g * num_gen
            n_feas = sum(
                1 for i in range(s, s + num_gen)
                if is_fully_feasible(
                    completions_text[i], instances[i], problem_type_list[i]
                )
            )
            if n_feas < 2:
                print(
                    f"⚠️ resample: group {g} 重试 {_MAX_RETRIES} 轮后"
                    f"仍只有 {n_feas} 条可行（需 >= 2），fallback 到惩罚模式"
                )

        if total_resampled > 0:
            self.log({
                "stats/resampled_completions":
                    self._gather_mean(total_resampled),
            })

    # ══════════════════════════════════════════════════════════════════
    #  三信号解耦 Advantage（Per-Customer 增量 PRM + 段内广播）
    # ══════════════════════════════════════════════════════════════════

    def _build_unified_advantages(self, completions_text, instances,
                                   problem_type_list,
                                   B, num_gen, T, device,
                                   prompt_ids, prompt_mask):
        """
        A_out  = A_feasibility_norm + A_outcome_norm （所有 token 共享，整组零均值）
        A_proc = per-customer 增量 PRM            （段内广播，仅可行解）
        → per-token advantage (B, T)

        prompt_ids/prompt_mask 传入是为了让 _compute_a_out 跨 rank 按 prompt 分组.
        """
        # ── 1. A_out (A_feasibility 跨 rank z-score + A_outcome 子集 z-score) ──
        a_out, is_feasible, components = self._compute_a_out(
            completions_text, instances, problem_type_list, B, num_gen, device,
            prompt_ids, prompt_mask,
        )

        # ── 2. 初始化 advantage tensor: 所有 token 先赋 A_out ────────
        advantages = a_out.unsqueeze(-1).expand(-1, T).contiguous()

        # ── 3. PRM 段内广播 (跨 rank) ────────────────────────────────
        # 跨 rank 设计 (跟 _compute_a_out 同样原因, 单 rank B < num_gen 时
        # rank-local group 不完整):
        #   step 1: rank-local 算每条可行 completion 的 ThinkPRMResult
        #           (parse_think_segments + POMO batch evaluate, rank-local)
        #   step 2: gather_object 跨 rank 收集 ThinkPRMResult (dict, 非 tensor)
        #   step 3: 按 (prompt_hash, customer_id) 跨 rank 分组 z-score 增量 reward
        #   step 4: 切回本 rank, 段内广播到 token (offset_maps 是 rank-local 的,
        #           所以广播必须 rank-local)
        prm_std_raw_values: list[float] = []   # 给 log 用, 兼容 disable_prm 路径
        n_with_zscore = 0
        n_with_fallback = 0
        if not config.disable_prm and self.pomo_prm is not None:
            from accelerate.utils import gather_object
            from collections import defaultdict
            from pomo_prm import ThinkPRMResult

            eps = 1e-8
            offset_maps = self._build_offset_maps(completions_text, B)

            # ── step 1: rank-local 算 ThinkPRMResult (仅可行 completion) ──
            rank_prm_results: list[ThinkPRMResult | None] = []
            for i in range(B):
                pt = problem_type_list[i]
                if is_feasible[i] and pt in self.pomo_prm.SUPPORTED:
                    rank_prm_results.append(
                        self.pomo_prm.compute_think_step_rewards(
                            completions_text[i], instances[i], pt,
                        )
                    )
                else:
                    rank_prm_results.append(None)

            # ── step 2: 跨 rank gather ──────────────────────────────────
            # gather_object: 各 rank list[B] → 拼接全 rank list[G]
            # 同时 gather prompt row hash 用作分组 key (与 _compute_a_out 一致)
            all_prm_results = gather_object(rank_prm_results)   # list of (Result|None) 长度 G

            masked_ids = prompt_ids.long() * prompt_mask.long()
            row_id_sum = masked_ids.sum(dim=-1)
            row_len    = prompt_mask.long().sum(dim=-1)
            row_hash   = (row_id_sum * 1000003 + row_len).contiguous()
            all_row_hash = self.accelerator.gather(row_hash).cpu().tolist()
            G = len(all_prm_results)
            assert G == len(all_row_hash), (
                f"PRM gather size mismatch: prm={G} hash={len(all_row_hash)}"
            )

            # ── step 3: 按 (prompt_hash, customer_id) 跨 rank z-score ──
            # 收集每个 (p_hash, c) 的正常 reward 列表 + anomaly global_idx 集合
            normal_rewards: dict = defaultdict(list)   # (p_hash, c) -> [(gidx, reward), ...]
            anomaly_lookup: dict = defaultdict(set)    # (p_hash, c) -> {gidx, ...}

            for gidx, prm_res in enumerate(all_prm_results):
                if prm_res is None:
                    continue
                p_hash = all_row_hash[gidx]
                for c, r in prm_res.customer_rewards.items():
                    normal_rewards[(p_hash, c)].append((gidx, r))
                for c in prm_res.anomaly_customers:
                    anomaly_lookup[(p_hash, c)].add(gidx)

            # 对每个 (p_hash, c) 算信号, 写 normalized_proc[gidx][c] = a_proc.
            # 三条路径:
            #   ≥2 normal + std>0    → z-score (anomaly 用 z(min) - σ, bounded)
            #   ≥2 normal + std=0    → normal 退化为 0 (无区分度), anomaly 用 fallback 常数
            #   <2 normal            → fallback 常数 (含单可行 + 孤儿 anomaly)
            # 不再用 std_floor: 新 anomaly 公式 z(min) - σ 与 std 解耦, 不会爆;
            # POMO 增量 reward 天然在 1e-3 量级, std_floor=1e-6 会误伤正常信号.
            # 遍历 normal_rewards ∪ anomaly_lookup 的 keys, 防止"孤儿 anomaly"
            # (anomaly_lookup 有但 normal_rewards 完全没出现的 customer) 被漏掉.
            normalized_proc: dict[int, dict[int, float]] = defaultdict(dict)
            all_keys = set(normal_rewards.keys()) | set(anomaly_lookup.keys())
            for key in all_keys:
                p_hash, c = key
                pairs = normal_rewards.get(key, [])
                anomaly_gidx_set = anomaly_lookup.get(key, set())

                if len(pairs) >= 2:
                    rewards_only = [r for _, r in pairs]
                    mean_c = float(np.mean(rewards_only))
                    std_raw = float(np.std(rewards_only))
                    prm_std_raw_values.append(std_raw)
                    if std_raw > 0.0:
                        # 标准 z-score 路径
                        std_c  = std_raw + eps
                        abnormal_val = min(rewards_only) - config.abnormal_sigma * std_c
                        for gidx, r in pairs:
                            normalized_proc[gidx][c] = (r - mean_c) / std_c
                        for gidx in anomaly_gidx_set:
                            if c not in normalized_proc.get(gidx, {}):
                                normalized_proc[gidx][c] = (abnormal_val - mean_c) / std_c
                        n_with_zscore += 1
                    else:
                        # 完全退化 (所有 normal reward 精确相等): normal 无信号, anomaly 仍要罚
                        for gidx in anomaly_gidx_set:
                            if c not in normalized_proc.get(gidx, {}):
                                normalized_proc[gidx][c] = config.fallback_anomaly_value
                        n_with_fallback += 1
                else:
                    # Fallback: 单条 normal 或孤儿 anomaly, 用绝对常数
                    for gidx, _ in pairs:
                        normalized_proc[gidx][c] = config.fallback_normal_value
                    for gidx in anomaly_gidx_set:
                        if c not in normalized_proc.get(gidx, {}):
                            normalized_proc[gidx][c] = config.fallback_anomaly_value
                    n_with_fallback += 1

            # ── step 4: 切回本 rank, 段内广播 ───────────────────────────
            rank = self.accelerator.process_index
            my_start = rank * B
            for j in range(B):
                gidx = my_start + j
                if gidx not in normalized_proc:
                    continue
                # 本 rank 的 ThinkPRMResult (用来取 char_range)
                prm_res = rank_prm_results[j]
                if prm_res is None or offset_maps[j] is None:
                    continue
                om = offset_maps[j]
                for c, a_proc in normalized_proc[gidx].items():
                    seg = prm_res.customer_ranges.get(c)
                    if seg is None:
                        continue
                    seg_cs, seg_ce = seg
                    tok_s, tok_e = self._char_to_token_range(seg_cs, seg_ce, om)
                    if tok_s is not None and tok_e is not None:
                        tok_e = min(tok_e, T)
                        advantages[j, tok_s:tok_e] += config.proc_alpha * a_proc

            # ── 诊断: 一次性 print PRM 跨 rank 统计 ─────────────────────
            if not getattr(self, "_prm_diag_logged", False):
                n_results_global = sum(1 for r in all_prm_results if r is not None)
                print(
                    f"[PRM_DIAG rank={rank}] all_prm_results: {n_results_global}/{G} non-None  "
                    f"keys_total={len(all_keys)} (zscore={n_with_zscore}, fallback={n_with_fallback})  "
                    f"normalized_proc 覆盖 {len(normalized_proc)} completions",
                    flush=True,
                )
                self._prm_diag_logged = True

        # ── 4. Log ───────────────────────────────────────────────────
        # PRM std_raw 量级判断: POMO 增量 reward 期望 ~1e-3, 若 std_raw 长期
        # 落在 1e-6 以下意味着 rollout 高度同质化, z-score 信号被 +eps 压扁,
        # 需要靠 fallback 兜住. n_zscore/n_fallback 组数反映分流分布.
        # advantage 实际信号: A_total_mean 因 z-score 强制零均值无意义,
        # 用 abs().mean() 看 token 级平均信号大小, std() 看分散度.
        n_feasible = sum(is_feasible)
        feas_rate = n_feasible / B if B > 0 else 0.0
        log_dict = {
            "reward/feasibility_rate":    self._gather_mean(feas_rate),
            "reward/R_parse_rate":        self._gather_mean(
                np.mean([c["parse"] for c in components])),
            "reward/R_coverage_rate":     self._gather_mean(
                np.mean([c["coverage"] for c in components])),
            "reward/R_constraint_mean":   self._gather_mean(
                np.mean([c["constraint"] for c in components])),
            "reward/R_format_mean":       self._gather_mean(
                np.mean([c["format"] for c in components])),
            "reward/A_abs_mean":          self._gather_mean(advantages.abs().mean()),
            "reward/A_std":               self._gather_mean(advantages.std()),
            "prm/n_zscore_groups":        float(n_with_zscore),
            "prm/n_fallback_groups":      float(n_with_fallback),
        }
        if prm_std_raw_values:
            std_arr = np.array(prm_std_raw_values)
            log_dict["prm/std_raw_mean"] = float(std_arr.mean())
            log_dict["prm/std_raw_min"]  = float(std_arr.min())
            log_dict["prm/std_raw_max"]  = float(std_arr.max())
        self.log(log_dict)

        return advantages

    # ── A_out = A_feasibility (跨 rank z-score) + A_outcome (可行子集 z-score) ──

    def _compute_a_out(self, completions_text, instances,
                       problem_type_list, B, num_gen, device,
                       prompt_ids, prompt_mask):
        """
        A_feasibility: parse + cov*con + format 加权 → 跨 rank 同 prompt 做 z-score
        A_outcome:     同 prompt 可行子集 z-score(-distance), 子集 ≥ 2 才算
        a_out = A_feasibility_norm + A_outcome_norm

        跨 rank 设计 (BATCH_DIAG 实证 B=4 < num_gen=8, 同 prompt 跨多 rank):
            单 rank B 可能不是 num_gen 的整数倍, 同 prompt 的 num_gen 条
            generation 散在多 rank. 必须 accelerator.gather 聚合后, 按
            prompt row hash 分组归一化, 再 slice 回本 rank.
            兼容 B 恰好 = k*num_gen 的情况 (gather 后同 group 仍在一起).

        Returns: a_out (B,), is_feasible list[bool], components list[dict]
        """
        from collections import defaultdict
        eps = 1e-8

        # ── 1. 本 rank 算 raw 信号 ──────────────────────────────────
        a_feas_raw = torch.zeros(B, device=device)
        is_feasible_local: list[bool] = []
        components: list[dict] = []
        distances_local: list[float] = []  # nan 占位不可行, 跨 rank 用 tensor 传

        for i in range(B):
            c = compute_terminal_components(
                completions_text[i], instances[i], problem_type_list[i]
            )
            components.append(c)

            feas = (c["parse"] == 1.0 and c["coverage"] == 1.0
                    and c["constraint"] == 1.0 and c["format"] == 1.0)
            is_feasible_local.append(feas)

            # cov × con 乘积合并: cov=0 时 con 无效, 防"丢覆盖换约束"hack
            a_feas_raw[i] = (config.w_p * c["parse"]
                             + config.w_cc * (c["coverage"] * c["constraint"])
                             + config.w_f * c["format"])

            if feas:
                prob = _PROBLEM_OBJS.get(problem_type_list[i])
                d = (prob.get_tour_distance(completions_text[i], instances[i])
                     if prob else None)
                distances_local.append(d if d is not None else float("nan"))
            else:
                distances_local.append(float("nan"))

        # ── 2. prompt row hash (抗跨 rank padding 长度差异) ────────
        # prompt_ids / prompt_mask 跨 rank 形状可能不同 (不同 pad 长度),
        # 不能直接 gather. 改 gather row-wise hash = (sum-of-valid-token-ids,
        # valid-length) 合并值. 同 prompt 不同 rank 的 hash 必相同, 不同
        # prompt 碰撞概率极低.
        masked_ids = prompt_ids.long() * prompt_mask.long()
        row_id_sum = masked_ids.sum(dim=-1)              # (B,)
        row_len    = prompt_mask.long().sum(dim=-1)      # (B,)
        row_hash   = row_id_sum * 1000003 + row_len      # (B,) int

        # ── 3. 跨 rank gather (所有 rank 必须同时调用) ─────────────
        all_a_feas_raw = self.accelerator.gather(a_feas_raw.contiguous())   # (G,)
        feas_t  = torch.tensor(is_feasible_local, device=device, dtype=torch.bool)
        all_feas_t = self.accelerator.gather(feas_t.contiguous())            # (G,) bool
        dist_t  = torch.tensor(distances_local, device=device, dtype=torch.float32)
        all_dist_t = self.accelerator.gather(dist_t.contiguous())            # (G,) nan-for-infeas
        all_row_hash = self.accelerator.gather(row_hash.contiguous())        # (G,) int

        G = all_a_feas_raw.shape[0]

        # ── 4. 按 prompt hash 分组 ──────────────────────────────────
        hash_cpu = all_row_hash.cpu().tolist()
        groups = defaultdict(list)
        for i, h in enumerate(hash_cpu):
            groups[h].append(i)

        # ── 5. 每组 z-score (A_feasibility 全组, A_outcome 仅可行子集) ──
        all_a_feas_norm    = torch.zeros_like(all_a_feas_raw)
        all_a_outcome_norm = torch.zeros_like(all_a_feas_raw)

        for h, idxs in groups.items():
            if len(idxs) < 2:
                continue   # 单条 group 无法 z-score, 全 0
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)

            # A_feasibility: 整组 z-score
            grp = all_a_feas_raw[idx_t]
            all_a_feas_norm[idx_t] = (grp - grp.mean()) / (grp.std() + eps)

            # A_outcome: 可行子集 z-score
            feas_mask = all_feas_t[idx_t] & ~torch.isnan(all_dist_t[idx_t])
            feas_local_pos = feas_mask.nonzero(as_tuple=True)[0]   # 子集在 idxs 中的位置
            if feas_local_pos.numel() >= 2:
                feas_idx_t = idx_t[feas_local_pos]
                neg_d = -all_dist_t[feas_idx_t]
                all_a_outcome_norm[feas_idx_t] = (
                    (neg_d - neg_d.mean()) / (neg_d.std() + eps)
                )

        # ── 6. 一次性诊断: gather 后的 group 分布 ────────────────────
        if not getattr(self, "_aout_diag_logged", False):
            rank = self.accelerator.process_index
            group_sizes = [len(v) for v in groups.values()]
            print(
                f"[A_OUT_DIAG rank={rank}] B={B} G={G} num_groups_after_gather={len(groups)} "
                f"group_sizes={group_sizes[:8]}{'...' if len(group_sizes)>8 else ''}  "
                f"(每组应 = num_gen={num_gen})",
                flush=True,
            )
            self._aout_diag_logged = True

        # ── 7. 切回本 rank ──────────────────────────────────────────
        rank = self.accelerator.process_index
        my_start = rank * B
        a_out = (all_a_feas_norm[my_start:my_start + B]
                 + all_a_outcome_norm[my_start:my_start + B])

        return a_out, is_feasible_local, components

    # ── Char → Token 映射 ────────────────────────────────────────────

    def _build_offset_maps(self, completions_text, B):
        offset_maps: list = [None] * B
        for i in range(B):
            try:
                enc = self.processing_class(
                    completions_text[i],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offset_maps[i] = enc.offset_mapping
            except (NotImplementedError, TypeError):
                pass
        return offset_maps

    @staticmethod
    def _char_to_token_range(char_start, char_end, offset_mapping):
        tok_start = None
        tok_end = None
        for t_idx, (cs, ce) in enumerate(offset_mapping):
            if ce <= char_start:
                continue
            if cs >= char_end:
                break
            if tok_start is None:
                tok_start = t_idx
            tok_end = t_idx + 1
        return tok_start, tok_end

    # ══════════════════════════════════════════════════════════════════
    #  FOARL Advantage
    # ══════════════════════════════════════════════════════════════════

    def _build_foarl_advantages(self, completions_text, instances,
                                 problem_type_list, B, num_gen, device):
        eps = 1e-8
        rewards = torch.zeros(B, device=device)
        all_components = []
        num_groups = B // num_gen

        for g in range(num_groups):
            s = g * num_gen
            pt = problem_type_list[s]
            inst = instances[s]
            ref_dist = solve_reference(inst, pt)

            for j in range(num_gen):
                i = s + j
                r, comp = compute_foarl_reward(
                    completions_text[i], inst, pt, ref_dist,
                    alpha=config.foarl_alpha,
                    omega_parse=config.foarl_omega_parse,
                    omega_coverage=config.foarl_omega_coverage,
                    omega_constraint=config.foarl_omega_constraint,
                    omega_format=config.foarl_omega_format,
                )
                rewards[i] = r
                all_components.append(comp)

        advantages = torch.zeros(B, device=device)
        for g in range(num_groups):
            s, e = g * num_gen, (g + 1) * num_gen
            group_r = rewards[s:e]
            advantages[s:e] = (group_r - group_r.mean()) / (group_r.std() + eps)

        r_f_vals = [c["R_f"] for c in all_components]
        r_o_vals = [c["R_o"] for c in all_components]
        gaps = [c["gap"] for c in all_components if c["gap"] is not None]
        self.log({
            "foarl/R_total_mean":    self._gather_mean(rewards.mean()),
            "foarl/R_f_mean":        self._gather_mean(np.mean(r_f_vals)),
            "foarl/R_o_mean":        self._gather_mean(np.mean(r_o_vals)),
            "foarl/gap_mean":        self._gather_mean(np.mean(gaps) if gaps else 0.0),
            "foarl/parse_rate":      self._gather_mean(np.mean([c["parse"] for c in all_components])),
            "foarl/coverage_rate":   self._gather_mean(np.mean([c["coverage"] for c in all_components])),
            "foarl/constraint_mean": self._gather_mean(np.mean([c["constraint"] for c in all_components])),
            "foarl/format_mean":     self._gather_mean(np.mean([c["format"] for c in all_components])),
            "foarl/A_mean":          self._gather_mean(advantages.mean()),
        })

        return advantages

    # ══════════════════════════════════════════════════════════════════
    #  Loss 计算（单一 DAPO token-level）
    # ══════════════════════════════════════════════════════════════════

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        """
        单一 DAPO token-level loss。
        A_total 广播到所有 token，长 completion 按 token 数占更多权重。
        """
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        advantages      = inputs["advantages"]                 # (B, T) or (B,)
        old_per_token_logps = inputs.get("old_per_token_logps")

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        prompt_length  = prompt_ids.shape[1]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        completion_logits = logits[:, prompt_length - 1:-1, :]
        per_token_logps = torch.log_softmax(completion_logits, dim=-1)
        per_token_logps = per_token_logps.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        if old_per_token_logps is None:
            old_per_token_logps = inputs.get("logprobs")
        if old_per_token_logps is None:
            old_per_token_logps = inputs.get("old_logprobs")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        eps_low  = config.clip_epsilon_low
        eps_high = config.clip_epsilon_high
        clipped_ratio = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)

        # advantage: (B, T) per-token 或 (B,) 标量
        adv = advantages.unsqueeze(-1) if advantages.dim() == 1 else advantages

        per_token_loss = -torch.min(ratio * adv, clipped_ratio * adv)

        # vLLM IS 校正
        is_ratio = inputs.get("importance_sampling_ratio")
        if is_ratio is not None:
            per_token_loss = per_token_loss * is_ratio
        elif not getattr(self, '_is_ratio_warning_emitted', False):
            print(
                f"⚠️ WARNING: inputs 里没找到 importance_sampling_ratio。\n"
                f"   当前 inputs keys: {sorted(inputs.keys())}"
            )
            self._is_ratio_warning_emitted = True

        # KL 正则
        beta_kl = getattr(self, 'beta', 0.0)
        if beta_kl != 0.0:
            ref_logps = inputs.get("ref_per_token_logps")
            if ref_logps is None:
                ref_logps = inputs.get("ref_logprobs")
            if ref_logps is not None:
                kl = torch.exp(ref_logps - per_token_logps) \
                     - (ref_logps - per_token_logps) - 1
                per_token_loss = per_token_loss + beta_kl * kl
            elif not getattr(self, '_kl_warning_emitted', False):
                print(
                    f"⚠️ WARNING: kl_coef={beta_kl} > 0 但 inputs 里找不到 "
                    f"ref_per_token_logps / ref_logprobs，KL anchor 实际未生效！"
                )
                self._kl_warning_emitted = True

        # DAPO token-level 聚合：总 token 数为分母，长 completion 权重更大
        loss = (per_token_loss * completion_mask).sum() \
               / completion_mask.sum().clamp(min=1.0)

        truncation_rate = (completion_mask.sum(-1) == 0).float().mean()
        self.log({
            "loss/total":                self._gather_mean(loss),
            "stats/truncation_rate":     self._gather_mean(truncation_rate),
        })

        return loss

    # ══════════════════════════════════════════════════════════════════
    #  第 1 个 step 耗时打印 (诊断瓶颈)
    # ══════════════════════════════════════════════════════════════════

    def training_step(self, *args, **kwargs):
        """包住单次 micro-batch 的 forward + backward.
        gradient_accumulation_steps 次 training_step 累加后视为完整 step.
        """
        _record = not self._timing_done
        if _record and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.time()

        loss = super().training_step(*args, **kwargs)

        if _record and torch.cuda.is_available():
            torch.cuda.synchronize()
        if _record:
            self._timing_log["fwd_bwd"] = (
                self._timing_log.get("fwd_bwd", 0.0) + time.time() - _t0
            )
            self._timing_fwd_bwd_count += 1
            grad_accum = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
            if self._timing_fwd_bwd_count >= grad_accum:
                self._timing_done = True
                self._print_first_step_timing()

        return loss

    def _print_first_step_timing(self):
        if not self.accelerator.is_main_process:
            return
        r = self._timing_log
        total = r.get("rollout", 0) + r.get("reward", 0) + r.get("fwd_bwd", 0)
        if total <= 0:
            return
        pct = lambda x: 100.0 * x / total
        n_micro = self._timing_fwd_bwd_count
        print(
            f"\n{'='*64}\n"
            f"  第 1 个 step 各阶段耗时 (rank 0)\n"
            f"{'='*64}\n"
            f"  rollout (vLLM 生成):              "
            f"{r.get('rollout', 0):7.2f}s  ({pct(r.get('rollout', 0)):5.1f}%)\n"
            f"  reward  (resample + advantages):  "
            f"{r.get('reward', 0):7.2f}s  ({pct(r.get('reward', 0)):5.1f}%)\n"
            f"  fwd+bwd (含 NCCL AllReduce, "
            f"{n_micro}×accum): "
            f"{r.get('fwd_bwd', 0):7.2f}s  ({pct(r.get('fwd_bwd', 0)):5.1f}%)\n"
            f"  {'-'*58}\n"
            f"  单 step 总计:                     {total:7.2f}s\n"
            f"{'='*64}\n",
            flush=True,
        )

    # ══════════════════════════════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════════════════════════════

    def _deserialize_instance(self, pd):
        if isinstance(pd, str):
            return json.loads(pd)
        if isinstance(pd, dict):
            return pd
        raise TypeError(
            f"problem_data 类型异常 (期望 str/dict): {type(pd).__name__}={pd!r}"
        )
