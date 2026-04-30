"""
GRPOPRMTrainer: 继承 trl.GRPOTrainer，三信号解耦 GRPO 训练。

设计（三信号解耦 + DAPO token-level）：
    1. Feasibility gate：parse + coverage + constraint + format 全满足才算可行
       - 不可行 → A_total = 动态惩罚（历史最小可行 advantage - margin）
       - 可行   → 进入 outcome + process 计算
    2. A_out (Outcome)：负距离，只在可行解之间独立归一化
    3. A_proc (Process)：客户节点 PRM 分数每步减组内均值 → 聚合标量 → 独立归一化
       - depot 不参与（可行性由 gate 管）
    4. A_total = A_out + α · A_proc → 广播到所有 token
    5. Loss：单一 DAPO token-level loss（按 token 数加权）
"""

import json
import numpy as np
import torch
from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer

from pomo_prm import POMOPRM, StepRewards
from terminal_reward import compute_terminal_components, is_fully_feasible
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

    def __init__(self, pomo_prm: POMOPRM, problem_types: list[str], **kwargs):
        super().__init__(**kwargs)
        self.pomo_prm = pomo_prm
        self.problem_types = problem_types
        self._hist_min = float('inf')

        if self.args.num_generations < 2:
            raise ValueError(
                f"num_generations 必须 >= 2 才能做 GRPO 组内对比；"
                f"当前 = {self.args.num_generations}"
            )

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

    # ── 动态惩罚 ─────────────────────────────────────────────────────

    def _get_penalty(self) -> float:
        if self._hist_min == float('inf'):
            return config.infeasible_default_penalty
        return self._hist_min - config.infeasible_margin

    # ══════════════════════════════════════════════════════════════════
    #  生成 + 三信号解耦
    # ══════════════════════════════════════════════════════════════════

    def _generate_and_score_completions(self, inputs):
        batch = super()._generate_and_score_completions(inputs)

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

        # 可行性重采样：每组至少 2 条可行解
        self._resample_infeasible(
            batch, completions_text, instances, problem_type_list, num_gen
        )

        # PRM step rewards（ablation 模式下跳过）
        if config.disable_prm:
            all_step_rewards = [None] * B
        else:
            all_step_rewards = []
            for i in range(B):
                pt = problem_type_list[i]
                if pt not in POMOPRM.SUPPORTED:
                    raise ValueError(
                        f"Problem type '{pt}' 不在 POMO PRM 支持列表 "
                        f"{sorted(POMOPRM.SUPPORTED)}。"
                    )
                sr = self.pomo_prm.compute_step_rewards(
                    completions_text[i], instances[i], pt
                )
                all_step_rewards.append(sr)

        # 三信号解耦 → 统一 advantage (B,)
        advantages = self._build_unified_advantages(
            completions_text, instances, problem_type_list,
            all_step_rewards, B, num_gen, device=completion_ids.device,
        )

        batch["advantages"] = advantages
        # 清理旧字段，避免父类或后续代码误用
        for key in ("terminal_advantages", "prm_advantages",
                     "prm_mask", "prm_denom"):
            batch.pop(key, None)

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
    #  三信号解耦 Advantage
    # ══════════════════════════════════════════════════════════════════

    def _build_unified_advantages(self, completions_text, instances,
                                   problem_type_list, all_step_rewards,
                                   B, num_gen, device):
        """
        Feasibility gate + Outcome (A_out) + Process (A_proc) → 统一 advantage (B,)

        不可行解: A_total = hist_min - margin（动态惩罚，最后赋值）
        可行解:   A_total = A_out + α · A_proc（独立归一化后相加）
        """
        eps = 1e-8
        A_total = torch.zeros(B, device=device)

        # ── 1. Feasibility + Outcome raw ─────────────────────────────
        feasible = torch.zeros(B, dtype=torch.bool, device=device)
        outcome_raw = torch.zeros(B, device=device)

        all_parse, all_cov, all_con, all_fmt = [], [], [], []
        n_feasible_total = 0

        for i in range(B):
            c = compute_terminal_components(
                completions_text[i], instances[i], problem_type_list[i]
            )
            all_parse.append(c["parse"])
            all_cov.append(c["coverage"])
            all_con.append(c["constraint"])
            all_fmt.append(c["format"])

            is_feas = (c["parse"] == 1.0 and c["coverage"] == 1.0
                       and c["constraint"] == 1.0 and c["format"] == 1.0)
            feasible[i] = is_feas

            if is_feas:
                n_feasible_total += 1
                prob = _PROBLEM_OBJS.get(problem_type_list[i])
                dist = prob.get_tour_distance(
                    completions_text[i], instances[i]
                ) if prob else None
                outcome_raw[i] = -dist if dist is not None else 0.0

        # ── 2. 逐组计算 A_out + A_proc ──────────────────────────────
        num_groups = B // num_gen
        fallback_mask = torch.zeros(B, dtype=torch.bool, device=device)

        for g in range(num_groups):
            s, e = g * num_gen, (g + 1) * num_gen
            f_mask = feasible[s:e]
            n_f = f_mask.sum().item()

            if n_f < 2:
                if n_f == 1:
                    A_total[s:e][f_mask] = 0.0
                    self._hist_min = min(self._hist_min, 0.0)
                fallback_mask[s:e] = ~f_mask
                continue

            # ── A_out: 可行解之间 z-score ────────────────────────────
            f_outcomes = outcome_raw[s:e][f_mask]
            A_out = (f_outcomes - f_outcomes.mean()) / (f_outcomes.std() + eps)

            # ── A_proc: 每步减均值 → 聚合 → z-score ────────────────
            f_indices = [s + j for j in range(num_gen) if f_mask[j]]
            K = instances[s]["n"]
            A_proc = torch.zeros(n_f, device=device)

            if not config.disable_prm:
                prm_matrix = torch.zeros(n_f, K, device=device)
                valid_proc = torch.ones(n_f, dtype=torch.bool, device=device)

                for fi, idx in enumerate(f_indices):
                    sr = all_step_rewards[idx]
                    if sr is None or len(sr.customer_rewards) < K:
                        valid_proc[fi] = False
                        continue
                    for k in range(K):
                        prm_matrix[fi, k] = sr.customer_rewards[k]

                n_valid = valid_proc.sum().item()

                if n_valid >= 2:
                    valid_prm = prm_matrix[valid_proc]
                    delta = valid_prm - valid_prm.mean(dim=0, keepdim=True)
                    raw = delta.mean(dim=1)
                    normalized = raw / (raw.std() + eps)
                    vi = 0
                    for fi in range(n_f):
                        if valid_proc[fi]:
                            A_proc[fi] = normalized[vi]
                            vi += 1

            # ── 组合 ────────────────────────────────────────────────
            A_feasible = A_out + config.proc_alpha * A_proc

            # 写回 A_total
            fi = 0
            for j in range(num_gen):
                if f_mask[j]:
                    A_total[s + j] = A_feasible[fi]
                    fi += 1

            # 更新历史最小值
            batch_min = A_feasible.min().item()
            self._hist_min = min(self._hist_min, batch_min)

        # ── 3. 不可行解赋惩罚 ────────────────────────────────────────
        penalty = self._get_penalty()
        # 正常组：有 >= 2 可行解做对比，即便 penalty > 0 也能通过相对关系学习
        A_total[(~feasible) & (~fallback_mask)] = penalty
        # fallback 组：没有足够可行解做对比，正惩罚会误导模型，强制负值
        fb_penalty = penalty if penalty <= 0 else -1.0
        A_total[fallback_mask] = fb_penalty

        # ── Log ──────────────────────────────────────────────────────
        feas_rate = n_feasible_total / B if B > 0 else 0.0
        self.log({
            "reward/feasibility_rate":    self._gather_mean(feas_rate),
            "reward/R_parse_rate":        self._gather_mean(np.mean(all_parse)),
            "reward/R_coverage_rate":     self._gather_mean(np.mean(all_cov)),
            "reward/R_constraint_mean":   self._gather_mean(np.mean(all_con)),
            "reward/R_format_mean":       self._gather_mean(np.mean(all_fmt)),
            "reward/penalty":             self._gather_mean(penalty),
            "reward/hist_min":            self._gather_mean(self._hist_min
                                                            if self._hist_min != float('inf')
                                                            else config.infeasible_default_penalty),
            "reward/A_total_mean":        self._gather_mean(A_total.mean()),
        })

        return A_total

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
        advantages      = inputs["advantages"]                 # (B,)
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

        eps_low  = getattr(self, 'epsilon_low', 0.2)
        eps_high = getattr(self, 'epsilon_high', 0.2)
        clipped_ratio = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)

        # 广播 advantage 到 (B, T)
        adv = advantages.unsqueeze(-1)

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
