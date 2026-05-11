"""
GRPOPRMTrainer: 继承 trl.GRPOTrainer，Per-Customer 增量 PRM + 段内广播。

设计（A_feasibility + A_outcome + Per-Customer PRM → per-token advantage）：
    1. A_feasibility：4D 信号加权（可行拿满，不可行按维度给部分分）
    2. A_outcome：可行解之间 z-score(-distance)，不可行 = 0
    3. A_out = A_feasibility + A_outcome → 所有 token 共享
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
        """每组可行解 < 2 时，通过 vLLM 重新生成替换不可行 completion，循环至 >= 2。"""
        _MAX_RETRIES = 5
        B, T = batch["completion_ids"].shape
        device = batch["completion_ids"].device
        pad_id = self.processing_class.pad_token_id or 0
        eos_id = self.processing_class.eos_token_id
        total_resampled = 0

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
                                   B, num_gen, T, device):
        """
        A_out = A_outcome + A_feasibility  （所有 token 共享）
        A_proc = per-customer 增量 PRM     （段内广播，仅可行解）
        → per-token advantage (B, T)
        """
        # ── 1. A_out（A_outcome + A_feasibility） ────────────────────
        a_out, is_feasible, components = self._compute_a_out(
            completions_text, instances, problem_type_list, B, num_gen, device
        )

        # ── 2. 初始化 advantage tensor: 所有 token 先赋 A_out ────────
        advantages = a_out.unsqueeze(-1).expand(-1, T).contiguous()

        # ── 3. Per-group PRM 段内广播 ────────────────────────────────
        if not config.disable_prm and self.pomo_prm is not None:
            offset_maps = self._build_offset_maps(completions_text, B)

            num_groups = B // num_gen
            for g in range(num_groups):
                s = g * num_gen
                group_texts = completions_text[s:s + num_gen]
                group_feasible = is_feasible[s:s + num_gen]
                inst = instances[s]
                pt = problem_type_list[s]

                prm_segments = self._compute_per_customer_prm(
                    group_texts, inst, pt, group_feasible
                )

                for j in range(num_gen):
                    k = s + j
                    segs = prm_segments.get(j)
                    if not segs or offset_maps[k] is None:
                        continue
                    om = offset_maps[k]
                    for (seg_cs, seg_ce, a_proc) in segs:
                        tok_s, tok_e = self._char_to_token_range(
                            seg_cs, seg_ce, om
                        )
                        if tok_s is not None and tok_e is not None:
                            tok_e = min(tok_e, T)
                            advantages[k, tok_s:tok_e] += (
                                config.proc_alpha * a_proc
                            )

        # ── 4. Log ───────────────────────────────────────────────────
        n_feasible = sum(is_feasible)
        feas_rate = n_feasible / B if B > 0 else 0.0
        self.log({
            "reward/feasibility_rate":    self._gather_mean(feas_rate),
            "reward/R_parse_rate":        self._gather_mean(
                np.mean([c["parse"] for c in components])),
            "reward/R_coverage_rate":     self._gather_mean(
                np.mean([c["coverage"] for c in components])),
            "reward/R_constraint_mean":   self._gather_mean(
                np.mean([c["constraint"] for c in components])),
            "reward/R_format_mean":       self._gather_mean(
                np.mean([c["format"] for c in components])),
            "reward/A_out_mean":          self._gather_mean(a_out.mean()),
            "reward/A_total_mean":        self._gather_mean(advantages.mean()),
        })

        return advantages

    # ── A_out = A_outcome + A_feasibility ────────────────────────────

    def _compute_a_out(self, completions_text, instances,
                       problem_type_list, B, num_gen, device):
        """
        A_feasibility: 4D 加权（可行拿满, 不可行部分分）
        A_outcome:     可行解之间 z-score(-distance)（组内 ≥2 才有）

        Returns: a_out (B,), is_feasible list[bool], components list[dict]
        """
        eps = 1e-8
        a_out = torch.zeros(B, device=device)
        is_feasible: list[bool] = []
        components: list[dict] = []
        distances: list[float | None] = []

        for i in range(B):
            c = compute_terminal_components(
                completions_text[i], instances[i], problem_type_list[i]
            )
            components.append(c)

            feas = (c["parse"] == 1.0 and c["coverage"] == 1.0
                    and c["constraint"] == 1.0 and c["format"] == 1.0)
            is_feasible.append(feas)

            a_feas = (config.w_p * c["parse"] + config.w_c * c["coverage"]
                      + config.w_k * c["constraint"] + config.w_f * c["format"])
            a_out[i] = a_feas

            if feas:
                prob = _PROBLEM_OBJS.get(problem_type_list[i])
                dist = (prob.get_tour_distance(completions_text[i], instances[i])
                        if prob else None)
                distances.append(dist)
            else:
                distances.append(None)

        num_groups = B // num_gen
        for g in range(num_groups):
            s = g * num_gen
            feas_in_grp = [
                (j, distances[s + j])
                for j in range(num_gen)
                if is_feasible[s + j] and distances[s + j] is not None
            ]
            if len(feas_in_grp) >= 2:
                neg_dists = torch.tensor(
                    [-d for _, d in feas_in_grp], device=device,
                    dtype=torch.float32,
                )
                mean_d = neg_dists.mean()
                std_d = neg_dists.std() + eps
                for idx, (j, _) in enumerate(feas_in_grp):
                    a_out[s + j] += (neg_dists[idx] - mean_d) / std_d

        return a_out, is_feasible, components

    # ── Per-Customer PRM（仅可行 completion） ────────────────────────

    def _compute_per_customer_prm(self, group_texts, instance,
                                   problem_type, group_feasible):
        """
        Per-Customer 增量 PRM + z-score 归一化。

        Returns:
            dict: local_index → [(char_start, char_end, a_proc), ...]
        """
        K = len(group_texts)
        n = instance["n"]

        feasible_idx = [k for k in range(K) if group_feasible[k]]
        if len(feasible_idx) < 2:
            return {}

        prm_results: dict[int, ThinkPRMResult] = {}
        for k in feasible_idx:
            pt = problem_type
            if pt not in POMOPRM.SUPPORTED:
                continue
            prm_results[k] = self.pomo_prm.compute_think_step_rewards(
                group_texts[k], instance, pt,
            )

        if len(prm_results) < 2:
            return {}

        prm_segments: dict[int, list] = {k: [] for k in range(K)}

        for c in range(1, n + 1):
            normal_rewards = []
            for k in feasible_idx:
                if k not in prm_results:
                    continue
                r = prm_results[k]
                if c in r.customer_rewards and c not in r.anomaly_customers:
                    normal_rewards.append((k, r.customer_rewards[c]))

            if len(normal_rewards) < 2:
                continue

            rewards_only = [rw for _, rw in normal_rewards]
            mean_c = float(np.mean(rewards_only))
            std_c = float(np.std(rewards_only)) + 1e-8
            abnormal_val = min(rewards_only) - config.abnormal_margin

            for k in feasible_idx:
                if k not in prm_results:
                    continue
                r = prm_results[k]
                is_abn = c in r.anomaly_customers or c not in r.customer_rewards

                if is_abn:
                    if c in r.customer_ranges:
                        a_proc = (abnormal_val - mean_c) / std_c
                        prm_segments[k].append(
                            (*r.customer_ranges[c], a_proc)
                        )
                else:
                    a_proc = (r.customer_rewards[c] - mean_c) / std_c
                    prm_segments[k].append(
                        (*r.customer_ranges[c], a_proc)
                    )

        return prm_segments

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
