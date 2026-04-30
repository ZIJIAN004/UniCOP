"""
GRPOPRMTrainer: 继承 trl.GRPOTrainer，双信号 GRPO 训练。

设计：
    Signal A (Terminal)：completion 级标量 reward (parse + coverage + constraint + format)
        → 组归一化为 z-score → 广播到该 completion 所有 valid token
    Signal B (PRM)     ：per-token POMO reward (客户累积值 + depot 反事实)
        → 客户 reward 同步骤跨 completion 组归一化；depot 直赋 (本身已是相对量)
        → 仅在合法前缀内的客户/depot token 位置非零

合成 (DAPO token-level 思想)：
    L = α · L_terminal + β · L_prm
    L_terminal = Σ_{i,t} loss_term · cmask  /  Σ_{i,t} cmask
                 (标准 DAPO token-level，长 completion 按 token 数占更多权重)
    L_prm      = mean_i [ Σ_t loss_prm · pmask  /  n_i ] over completions with prm
                 (固定分母 n_i，摆烂 k 客户被自动稀释到 k/n 强度，抑制短摆烂 hack)
    pmask[i,t] = 1 仅当 t 落在合法前缀内的客户/depot 位置；其他=0
    （违规之后/think/格式/parse 失败的 completion 整体 → pmask=0，不进 PRM 平均）

Token 位置映射：tokenizer offset_mapping 优先；不一致时退回增量解码。
"""

import json
import numpy as np
import torch
from trl import GRPOTrainer

from pomo_prm import POMOPRM, StepRewards
from terminal_reward import compute_terminal_components
from config import config


class GRPOPRMTrainer(GRPOTrainer):

    def __init__(self, pomo_prm: POMOPRM, problem_types: list[str], **kwargs):
        super().__init__(**kwargs)
        self.pomo_prm = pomo_prm
        self.problem_types = problem_types

        # GRPO 必须组内对比；num_generations < 2 时 std() 返 NaN，
        # 组归一化产生 NaN advantage → loss NaN → 训练崩溃。
        if self.args.num_generations < 2:
            raise ValueError(
                f"num_generations 必须 >= 2 才能做 GRPO 组内对比；"
                f"当前 = {self.args.num_generations}"
            )

    def _get_train_sampler(self, dataset=None):
        return super()._get_train_sampler()

    # ── 多卡聚合工具 ──────────────────────────────────────────────────

    def _gather_mean(self, value) -> float:
        """
        将标量（python float / 0-dim tensor）跨所有 rank gather，返回全局均值。
        单卡时 accelerator.gather 是 no-op，等价于本地值；多卡时正确聚合。
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), device=self.accelerator.device)
        if value.dim() == 0:
            value = value.unsqueeze(0)
        gathered = self.accelerator.gather(value)
        return gathered.float().mean().item()

    # ══════════════════════════════════════════════════════════════════
    #  生成 + 双信号构建
    # ══════════════════════════════════════════════════════════════════

    def _generate_and_score_completions(self, inputs):
        """生成 completions，构建 terminal 标量 + PRM per-token + PRM mask。"""

        # 1. 调用父类生成（标准 reward 仅作占位日志）
        batch = super()._generate_and_score_completions(inputs)

        completion_ids  = batch["completion_ids"]    # (B, T)
        completion_mask = batch["completion_mask"]    # (B, T)
        B, T = completion_ids.shape
        num_gen = self.args.num_generations

        # 2. 展开 problem_data / problem_type 到 B 维（每 prompt 对应 num_gen 条）
        # 数据由 data/generate.py 保证 problem_data/problem_type 字段齐全;
        # 任何缺失或长度不合理都视为数据 pipeline 异常,直接抛错。
        #
        # 格式兼容:
        #   TRL ≤ 0.14: inputs 是 dict-of-lists, len = num_prompts
        #   TRL 1.1+:  inputs 是 list-of-dicts, len 可能 = num_prompts (未展开)
        #              或 = B (num_prompts × num_gen, 已展开)。两种都要处理。
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
            # 老行为或 TRL 1.1 未扩展: 每条 prompt 复制 num_gen 次
            problem_data_list = [pd for pd in problem_data_raw for _ in range(num_gen)]
            problem_type_list = [pt for pt in problem_type_raw for _ in range(num_gen)]
        elif n_raw == B:
            # TRL 1.1 已扩展: 直接用
            problem_data_list = list(problem_data_raw)
            problem_type_list = list(problem_type_raw)
        else:
            raise ValueError(
                f"problem_data 长度 {n_raw} 既不是 num_prompts={num_prompts} "
                f"也不是 B={B},无法判断是否展开。"
                f"B={B}, num_gen={num_gen}, problem_data_raw[0]={problem_data_raw[0] if problem_data_raw else None!r}"
            )

        # 3. 解码 completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # 4. 反序列化 instance（若 pd 类型不对会直接抛 TypeError/JSONDecodeError）
        instances = [self._deserialize_instance(pd) for pd in problem_data_list]

        # 5. 计算 PRM step rewards（POMO 不支持的问题类型直接报错，
        #    train.py 启动时已做早期检查，运行时不应触发）
        all_step_rewards = []
        for i in range(B):
            pt = problem_type_list[i]
            if pt not in POMOPRM.SUPPORTED:
                raise ValueError(
                    f"Problem type '{pt}' 不在 POMO PRM 支持列表 "
                    f"{sorted(POMOPRM.SUPPORTED)}。vanilla reward 模式已删除，"
                    f"请仅训练 POMO 支持的问题类型。"
                )
            sr = self.pomo_prm.compute_step_rewards(
                completions_text[i], instances[i], pt
            )
            all_step_rewards.append(sr)

        # 6. Terminal 标量 reward (B,)
        terminal_advantages = self._build_terminal_advantages(
            completions_text, instances, problem_type_list, B, num_gen,
            device=completion_ids.device,
        )

        # 7. PRM per-token reward + mask (B, T) × 2
        prm_advantages, prm_mask = self._build_prm_advantages(
            all_step_rewards, completion_ids, completions_text,
            B, T, num_gen, device=completion_ids.device,
        )

        # 8. PRM 归一化分母（每个 completion 用其 instance 的 n=客户数）
        # 用 n 而非 pmask_sum 的好处：摆烂 completion 的 PRM 贡献被自动稀释
        # 到 (实际 PRM token 数 / n)。完整 rollout PRM token 数 ≈ n（TSP 正好 n，
        # CVRP/VRPTW ≈ n + depot 回访数，略 >n）；摆烂 5/n 自动只有 1/4 强度。
        prm_denom = torch.tensor(
            [inst["n"] for inst in instances],
            dtype=torch.float32, device=completion_ids.device,
        )

        # 9. 写入 batch（loss 阶段读取）
        batch["terminal_advantages"] = terminal_advantages   # (B,)
        batch["prm_advantages"]      = prm_advantages         # (B, T)
        batch["prm_mask"]            = prm_mask               # (B, T)
        batch["prm_denom"]           = prm_denom              # (B,)
        # 移除标量 advantages 字段，避免父类 compute_loss 误用
        batch.pop("advantages", None)

        return batch

    # ══════════════════════════════════════════════════════════════════
    #  Terminal Advantage：标量 → 组归一化
    # ══════════════════════════════════════════════════════════════════

    def _build_terminal_advantages(self, completions_text, instances,
                                    problem_type_list, B, num_gen, device):
        """
        每条 completion 算一个 R_terminal 标量，组内 (R - μ) / σ 归一化为 (B,) 张量。
        同时 log 各维度命中率（parse/coverage/constraint/format）和 R_terminal 均值。
        """
        # 各维度权重从 config 取
        w_p = config.terminal_w_parse
        w_c = config.terminal_w_coverage
        w_n = config.terminal_w_constraint
        w_f = config.terminal_w_format

        # 收集各维度分数用于 log
        all_parse, all_cov, all_con, all_fmt = [], [], [], []
        raw = torch.zeros(B, device=device)
        for i in range(B):
            c = compute_terminal_components(
                completions_text[i], instances[i], problem_type_list[i]
            )
            all_parse.append(c["parse"])
            all_cov.append(c["coverage"])
            all_con.append(c["constraint"])
            all_fmt.append(c["format"])
            raw[i] = (
                w_p * c["parse"] +
                w_c * c["coverage"] +
                w_n * c["constraint"] +
                w_f * c["format"]
            )

            # ── 诊断: 前 2 条 completion 打印解析详情,排查 coverage=0 根因 ──
            if i < 2 and not getattr(self, '_coverage_diag_done', False):
                from utils.parse import parse_single_route, parse_multi_route
                n_inst = instances[i]["n"]
                pt = problem_type_list[i]
                is_multi = pt in ("cvrp", "vrptw")
                if is_multi:
                    parsed = parse_multi_route(completions_text[i], n_inst)
                    if parsed:
                        all_vis = [v for r in parsed for v in r if v != 0]
                    else:
                        all_vis = []
                else:
                    parsed = parse_single_route(completions_text[i], n_inst)
                    all_vis = [v for v in parsed if v != 0] if parsed else []
                # 只打答案段末尾 500 字符 + 解析结果
                think_end = completions_text[i].rfind("</think>")
                answer_tail = completions_text[i][think_end:think_end+500] if think_end != -1 else completions_text[i][-500:]
                print(f"\n{'='*60}")
                print(f"[DIAG coverage] completion[{i}] type={pt} n={n_inst}")
                print(f"  parse={c['parse']} coverage={c['coverage']} constraint={c['constraint']:.3f}")
                print(f"  parsed_route={parsed}")
                print(f"  customer_visits({len(all_vis)}): {sorted(set(all_vis))}")
                print(f"  missing: {sorted(set(range(1, n_inst+1)) - set(all_vis))}")
                print(f"  duplicates: {len(all_vis) - len(set(all_vis))}")
                print(f"  answer_tail: {answer_tail[:500]}")
                print(f"{'='*60}\n")
        if not getattr(self, '_coverage_diag_done', False):
            self._coverage_diag_done = True

        # 组内归一化：每 num_gen 一组
        adv = torch.zeros(B, device=device)
        num_groups = B // num_gen
        for g in range(num_groups):
            s, e = g * num_gen, (g + 1) * num_gen
            r = raw[s:e]
            mean_r = r.mean()
            std_r  = r.std() + 1e-8
            adv[s:e] = (r - mean_r) / std_r

        # ── 自定义 log（terminal 维度命中率 + R_terminal 均值） ──────
        # 多卡：_gather_mean 跨所有 rank 聚合后返回全局均值
        self.log({
            "terminal/R_parse_rate":      self._gather_mean(np.mean(all_parse)),
            "terminal/R_coverage_rate":   self._gather_mean(np.mean(all_cov)),
            "terminal/R_constraint_mean": self._gather_mean(np.mean(all_con)),
            "terminal/R_format_mean":     self._gather_mean(np.mean(all_fmt)),
            "terminal/R_terminal_mean":   self._gather_mean(raw.mean()),
        })
        return adv

    # ══════════════════════════════════════════════════════════════════
    #  PRM Advantage：per-token → 同步骤组归一化 + mask
    # ══════════════════════════════════════════════════════════════════

    def _build_prm_advantages(self, all_step_rewards, completion_ids,
                               completions_text, B, T, num_gen, device):
        """
        客户 reward 同步骤跨 completion 组归一化；depot 反事实直接赋值。
        prm_mask: 仅在合法前缀内的客户/depot token 位置 = 1；其他全 0。
                  违规之后 / parse 失败 / 不在 PRM SUPPORTED → 该 completion mask 全 0。
        """
        advantages = torch.zeros(B, T, device=device)
        prm_mask   = torch.zeros(B, T, device=device)

        # customer_rewards_by_step[i] = {step_k: (token_pos, reward)}
        customer_rewards_by_step = [{} for _ in range(B)]

        # ── Step 1: 写入 raw 客户/depot reward 并标记 mask ───────────
        for i in range(B):
            sr = all_step_rewards[i]
            if sr is None:
                continue   # mask 全 0，该 completion 不参与 PRM loss

            node_token_map = self._map_nodes_to_tokens(
                completion_ids[i], completions_text[i]
            )

            # 客户累积 reward：暂存，等 Step 2 组归一化
            for k, reward in enumerate(sr.customer_rewards):
                if k >= len(sr.customer_token_positions):
                    break
                char_pos = sr.customer_token_positions[k]
                if char_pos < 0:
                    continue
                tok_idx = self._char_to_token(node_token_map, char_pos)
                if tok_idx is None or tok_idx >= T:
                    continue
                customer_rewards_by_step[i][k] = (tok_idx, reward)
                prm_mask[i, tok_idx] = 1.0

            # Depot 反事实 reward：直接赋值（已是相对量 val_with - val_without，
            # 正值=该回 depot，负值=不该回，与 GRPO advantage 标准语义一致）
            for k, reward in enumerate(sr.depot_rewards):
                if k >= len(sr.depot_token_positions):
                    break
                char_pos = sr.depot_token_positions[k]
                if char_pos < 0:
                    continue
                tok_idx = self._char_to_token(node_token_map, char_pos)
                if tok_idx is None or tok_idx >= T:
                    continue
                advantages[i, tok_idx] = reward
                prm_mask[i, tok_idx] = 1.0

        # ── Step 2: 客户 reward 同步骤跨 completion 组归一化 ─────────
        num_groups = B // num_gen
        for g in range(num_groups):
            s, e = g * num_gen, (g + 1) * num_gen

            max_steps = 0
            for i in range(s, e):
                if customer_rewards_by_step[i]:
                    max_steps = max(max_steps,
                                    max(customer_rewards_by_step[i].keys()) + 1)

            for k in range(max_steps):
                entries = []
                for i in range(s, e):
                    if k in customer_rewards_by_step[i]:
                        tok_idx, reward = customer_rewards_by_step[i][k]
                        entries.append((i, tok_idx, reward))

                if len(entries) < 2:
                    # 只有 ≤1 个 completion 有该步 → 直接用 raw（无对比意义）
                    for i, tok_idx, reward in entries:
                        advantages[i, tok_idx] = reward
                    continue

                rewards = torch.tensor([e[2] for e in entries], device=device)
                mean_r = rewards.mean()
                std_r  = rewards.std() + 1e-8
                for j, (i, tok_idx, _) in enumerate(entries):
                    advantages[i, tok_idx] = (rewards[j] - mean_r) / std_r

        return advantages, prm_mask

    # ══════════════════════════════════════════════════════════════════
    #  Loss 计算（Method A：terminal/PRM 双 loss 各自归一）
    # ══════════════════════════════════════════════════════════════════

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        """
        始终走双信号 loss 路径。
        _generate_and_score_completions 总是注入 terminal/prm 字段，所以无需 fallback。
        """
        return self._compute_loss_dual(model, inputs)

    def _compute_loss_dual(self, model, inputs):
        """
        L = α · L_terminal + β · L_prm   （两个信号都用 DAPO token-level 思想）

        L_terminal: 全 batch 所有 token 累加 / 总 token 数（标准 DAPO token-level）
                    长 completion 按 token 数占更多权重——对 long-form reasoning 友好
        L_prm     : 仅客户/depot token 受影响，按"该 instance 的 n"归一
                    （DAPO token-level 的"固定期望分母"变种：摆烂 k 客户得到 k/n 强度，
                     防止短摆烂靠"少几步好选择"和完整 rollout 等价竞争）
        两 loss 各按"自己的有效 token 数"分母平均，量纲对齐 (z-score 量级)。
        """
        prompt_ids      = inputs["prompt_ids"]                # (B, P)
        prompt_mask     = inputs["prompt_mask"]                # (B, P)
        completion_ids  = inputs["completion_ids"]             # (B, T)
        completion_mask = inputs["completion_mask"]            # (B, T)
        terminal_adv    = inputs["terminal_advantages"]         # (B,)
        prm_adv         = inputs["prm_advantages"]              # (B, T)
        prm_mask        = inputs["prm_mask"]                    # (B, T)
        prm_denom       = inputs["prm_denom"]                   # (B,) = 各 instance 的 n
        old_per_token_logps = inputs.get("old_per_token_logps")

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        prompt_length  = prompt_ids.shape[1]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        # 取 completion 段 logits（autoregressive shift by 1）
        completion_logits = logits[:, prompt_length - 1:-1, :]  # (B, T, V)

        per_token_logps = torch.log_softmax(completion_logits, dim=-1)
        per_token_logps = per_token_logps.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)

        # importance sampling ratio
        # trl 1.x 可能改 batch key 名（如 logprobs），加 fallback 兼容
        # 注意: Tensor 在 `or` 中会触发 "bool value ambiguous" 错,必须显式 is None 检查
        #
        # num_iterations=1 (默认) 时 TRL 不在 batch 里存 old_per_token_logps,
        # 三个 key 全返 None → 必须用 per_token_logps.detach() 兜底。
        # 写 torch.ones_like 会新建 requires_grad=False 的常数 tensor,
        # 彻底切断 per_token_logps 的计算图,导致 grad_norm 恒为 0 (实测 bug)。
        # 正确姿势: exp(logp - logp.detach()) 数值恒=1 但梯度 1.0 保留,
        # 退化为标准 policy gradient (与 TRL 官方 grpo_trainer 一致)。
        # 参考: trl Issue #2769
        if old_per_token_logps is None:
            old_per_token_logps = inputs.get("logprobs")
        if old_per_token_logps is None:
            old_per_token_logps = inputs.get("old_logprobs")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        # GRPO clipped per-token loss 的两个分量
        eps_low  = getattr(self, 'epsilon_low', 0.2)
        eps_high = getattr(self, 'epsilon_high', 0.2)
        clipped_ratio = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)

        # 广播 terminal advantage 到 (B, T)
        terminal_adv_b = terminal_adv.unsqueeze(-1).expand_as(prm_adv)

        # Terminal per-token loss（每 token 都用同一 A_terminal[i]）
        loss_term_t = -torch.min(ratio * terminal_adv_b,
                                  clipped_ratio * terminal_adv_b)
        # PRM per-token loss
        loss_prm_t  = -torch.min(ratio * prm_adv,
                                  clipped_ratio * prm_adv)

        # ── vLLM Importance Sampling 校正 (TRL 1.2+ 新特性, 对齐官方 compute_loss) ──
        # TRL 1.2 默认 vllm_importance_sampling_correction=True, 在
        # _generate_and_score_completions 里把 importance_sampling_ratio =
        # exp(old_logps - vllm_logps) 写入 batch (已按 mode 做过 truncate/mask)。
        # 官方 compute_loss: per_token_loss *= importance_sampling_ratio (在 min-clip
        # 之后, KL 之前). 我们的双 loss 同样处理,否则 vLLM/训练模型的 logp 数值差
        # 无法校正 → off-policy 偏置。
        # 参考: trl/trainer/grpo_trainer.py main line 2541-2542
        # shape: token_* mode → (B, T);  sequence_* mode → (B, 1),都能广播。
        is_ratio = inputs.get("importance_sampling_ratio")
        if is_ratio is not None:
            loss_term_t = loss_term_t * is_ratio
            loss_prm_t  = loss_prm_t  * is_ratio
        else:
            # TRL 版本不同可能换 key 名或关闭此特性; 一次性 WARN
            if not getattr(self, '_is_ratio_warning_emitted', False):
                print(
                    f"⚠️ WARNING: inputs 里没找到 importance_sampling_ratio。\n"
                    f"   若 use_vllm=True 且 vllm_importance_sampling_correction=True (TRL 1.2+ 默认),\n"
                    f"   vLLM/训练 logp 数值差的 IS 校正未应用 → off-policy 偏置.\n"
                    f"   当前 inputs keys: {sorted(inputs.keys())}"
                )
                self._is_ratio_warning_emitted = True

        # KL 正则（仍用 completion_mask 范围，按 terminal 那项归一）
        beta_kl = getattr(self, 'beta', 0.0)
        if beta_kl != 0.0:
            # trl 1.x 可能改 key 名，加 fallback
            # 注意: Tensor 在 `or` 中会触发 "bool value ambiguous" 错,必须显式 is None 检查
            ref_logps = inputs.get("ref_per_token_logps")
            if ref_logps is None:
                ref_logps = inputs.get("ref_logprobs")
            if ref_logps is not None:
                kl = torch.exp(ref_logps - per_token_logps) \
                     - (ref_logps - per_token_logps) - 1
                loss_term_t = loss_term_t + beta_kl * kl
            else:
                # 静默失败防御：beta_kl > 0 却拿不到 ref logps，
                # 说明 trl 改了 batch key 名 → 一次性大声警告
                if not getattr(self, '_kl_warning_emitted', False):
                    print(
                        f"⚠️ WARNING: kl_coef={beta_kl} > 0 但 inputs 里找不到 "
                        f"ref_per_token_logps / ref_logprobs，KL anchor 实际未生效！"
                        f"\n  可能是 trl 版本改了 batch key 名。"
                        f"\n  当前 inputs keys: {sorted(inputs.keys())}"
                    )
                    self._kl_warning_emitted = True

        # ── 双信号聚合（DAPO token-level） ────────────────────────
        # Terminal: 全 batch 所有 valid token 累加 / 总 token 数
        # 长 completion 按 token 比例占更多权重——长正确链 think 段被更强奖励，
        # 长错误链 think 段被更强惩罚（DAPO 核心动机，对 long-form reasoning 友好）
        L_terminal = (loss_term_t * completion_mask).sum() \
                     / completion_mask.sum().clamp(min=1.0)

        # PRM: completion 长度加权 (DAPO token-level 精神)
        # 原始设计: L_prm = mean_i[ sum_t(loss * pmask) / n_i ]
        # 问题: ratio=1 时 loss_prm_t = -adv (线性), z-score 保证
        #   Σ_i A[i,k] = 0 @ 每个 step k, 所以 Σ_i L_prm_per[i] ≡ 0。
        # 修复: 乘以 completion 长度 T_i, 让
        #   Σ_i L_prm_per[i] * T_i = Σ_k Σ_i (-A[i,k] * T_i) ≠ 0
        #   (和 L_terminal 的 DAPO 加权同理: 长 completion 的 PRM 信号更强)
        pmask_sum = prm_mask.sum(-1)                                       # (B,)
        has_prm   = (pmask_sum > 0).float()                                # (B,)
        comp_len  = completion_mask.sum(-1).clamp(min=1.0)                 # (B,)
        L_prm_per = (loss_prm_t * prm_mask).sum(-1) / prm_denom            # (B,)
        L_prm_per = L_prm_per * has_prm * comp_len                        # 长度加权

        # batch-level：按有效 completion 的 token 总数归一（和 L_terminal 量纲对齐）
        L_prm = L_prm_per.sum() / (has_prm * comp_len).sum().clamp(min=1.0)

        alpha = config.terminal_alpha
        beta_w = config.prm_beta
        loss = alpha * L_terminal + beta_w * L_prm

        # ── 自定义 log（loss 拆分 + PRM 覆盖率 + 截断率），多卡 gather ──
        # truncation: trl 的 mask_truncated_completions=True 会把整条截断
        # completion 的 cmask 全部设为 0，所以 cmask_sum==0 即截断
        truncation_rate = (completion_mask.sum(-1) == 0).float().mean()
        prm_coverage = has_prm.mean()
        self.log({
            "loss/L_terminal":       self._gather_mean(L_terminal),
            "loss/L_prm":            self._gather_mean(L_prm),
            "stats/prm_coverage":    self._gather_mean(prm_coverage),
            "stats/truncation_rate": self._gather_mean(truncation_rate),
        })

        return loss

    # ══════════════════════════════════════════════════════════════════
    #  Token 映射（保留：offset_mapping 方案 + 增量解码 fallback）
    # ══════════════════════════════════════════════════════════════════

    def _map_nodes_to_tokens(self, token_ids, completion_text):
        num_tokens = len(token_ids) if isinstance(token_ids, list) else token_ids.shape[0]
        try:
            encoding = self.processing_class(
                completion_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offset_mapping = encoding.get("offset_mapping", [])
            if offset_mapping:
                return offset_mapping
        except Exception:
            pass

        tokens = token_ids.cpu().tolist() if not isinstance(token_ids, list) else token_ids
        offsets = []
        prev_text = ""
        for i in range(len(tokens)):
            curr_text = self.processing_class.decode(
                tokens[:i + 1], skip_special_tokens=True
            )
            offsets.append((len(prev_text), len(curr_text)))
            prev_text = curr_text
        return offsets

    def _char_to_token(self, offset_mapping, char_pos):
        for t, (start, end) in enumerate(offset_mapping):
            if start <= char_pos < end:
                return t
        for t, (start, end) in enumerate(offset_mapping):
            if start >= char_pos:
                return t
        return None

    # ══════════════════════════════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════════════════════════════

    def _deserialize_instance(self, pd):
        """problem_data 必须是 JSON 字符串或 dict；其他类型视为数据异常直接抛。"""
        if isinstance(pd, str):
            return json.loads(pd)
        if isinstance(pd, dict):
            return pd
        raise TypeError(
            f"problem_data 类型异常 (期望 str/dict): {type(pd).__name__}={pd!r}"
        )
