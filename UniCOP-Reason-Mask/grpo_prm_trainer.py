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

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from accelerate.utils import broadcast_object_list, gather_object

# ── SDPA 守卫: 训练前向禁 math 后端, 强制 flash/efficient(O(S))。───────────────
# 若 SDPA 因某种 mask/dtype 真要退回 naive math(O(S²)), 直接报错而非悄悄变慢。
# torch 无 torch.nn.attention API 时降级为 no-op(可移植到其它主机/旧 torch)。
from contextlib import nullcontext as _nullcontext, contextmanager


@contextmanager
def _cuda_timer(store: dict, key: str, enabled: bool):
    """精细化耗时统计: cuda.synchronize() 包夹一段代码, 累加到 store[key]。
    enabled=False 时零开销 no-op (正常训练不受影响)。GPU 异步, 必须 sync 才准。"""
    if not enabled:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t = time.time()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store[key] = store.get(key, 0.0) + (time.time() - _t)
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend as _SDPB
    def _sdpa_no_math():
        return _sdpa_kernel([_SDPB.FLASH_ATTENTION, _SDPB.EFFICIENT_ATTENTION])
except Exception:
    def _sdpa_no_math():
        return _nullcontext()
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


# 用 skip_special_tokens=False decode 后, chat template 的控制 token (Qwen3 的
# <|im_end|>, R1-Distill 的 <｜end▁of▁sentence｜> 等) 会以字面字符串残留, 影响
# parse/regex. 这里手工抹掉, 但保留 <think>/</think> (Qwen3 special, R1 是普通
# BPE 都按字面保留, 下游 parse / mask 段切分依赖此边界).
_CHAT_SPECIAL_TOKENS = (
    "<|im_end|>", "<|im_start|>", "<|endoftext|>",
    "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>",
    "<|begin_of_text|>", "<|eot_id|>",
)


def _strip_chat_specials(text: str) -> str:
    for tok in _CHAT_SPECIAL_TOKENS:
        if tok in text:
            text = text.replace(tok, "")
    return text


# 逐 token log-prob 的 chunk 大小 (token 维分块, 与 completion 总长解耦)。
# 越小峰值显存越低; 512 对 7B + 词表~151k 约 1.8GiB/块 (fp32), 安全且开销可忽略。
_LOGP_CHUNK_SIZE = 512


def _selective_logp_chunked(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """显存友好的逐 token log-prob, 不物化整块 [B, T, V] 的 log_softmax。

    数学上等价于 ``log_softmax(logits, -1).gather(-1, index)``:
        log p(token) = logit[token] - logsumexp(logits over vocab)
    但在 token 维分块、每块临时 upcast fp32 计算, 因此:
      * 不再额外造一个和 logits 等大的 [B, T, V] 张量 (原 log_softmax 的 OOM 元凶);
      * fp32 reduction 比原 bf16 log_softmax 数值更稳 (解决"logsumexp bf16 不稳"顾虑)。

    建议配合 torch.utils.checkpoint(use_reentrant=False) 调用: backward 会按块重算,
    不把任何 [B, T, V] 中间量留到反传, forward / backward 峰值都只是单块大小。

    Args:
        logits: [B, T, V], 通常是 outputs.logits[:, :-1, :] (policy 对 completion 的预测)。
        index:  [B, T], 各位置实际 token id (completion_ids)。
    Returns:
        [B, T] 的 per-token log-prob (fp32)。
    """
    parts = []
    T = logits.shape[1]
    for s in range(0, T, _LOGP_CHUNK_SIZE):
        e = min(s + _LOGP_CHUNK_SIZE, T)
        lg = logits[:, s:e, :].float()                                      # [B, c, V] fp32 (临时)
        sel = lg.gather(dim=-1, index=index[:, s:e].unsqueeze(-1)).squeeze(-1)  # [B, c]
        parts.append(sel - torch.logsumexp(lg, dim=-1))                     # [B, c]
    return torch.cat(parts, dim=1)                                          # [B, T]


def _logp_from_hidden_chunked(hidden: torch.Tensor, index: torch.Tensor,
                              weight: torch.Tensor) -> torch.Tensor:
    """直接从 hidden state 分块过 LM head 算 per-token log-prob, 连 [B, T, V] logits 都不物化。

    这是 _selective_logp_chunked 的"更省一档"版本 (change A / 手写 FLCE 思路):
      * _selective_logp_chunked 仍需要外部先有一个完整 [B, T, V] 的 logits (LM head 输出);
      * 本函数把 LM head 也搬进分块循环 —— 每块只算 [B, c, V], 全程不存在完整 logits。
    单步前向峰值 ~= 一个 chunk 的 [B, c, V] fp32, 与 completion 总长、词表无关 (只跟 chunk 走)。

    数学等价 log_softmax(hidden @ weight.T, -1).gather(-1, index):
        log p = (hidden·w_token) - logsumexp(hidden·W over vocab)
    weight 必须是 frozen 的完整 [V, H] 稠密权重 (ZeRO-3 下需先 gather, 见 _gather_head_weight);
    因 lm_head 不在 LoRA target 内, 对 weight 无需梯度, 梯度只回流到 hidden→backbone(LoRA)。

    必须配合 torch.utils.checkpoint(use_reentrant=False): backward 按块重算, 用同一份
    captured weight (不再 re-gather), 不把任何 [B, T, V] 量留到反传。

    Args:
        hidden: [B, T, H], backbone 最后一层 (已过 final norm) 的 completion 段 hidden。
        index:  [B, T], completion_ids。
        weight: [V, H], lm_head.weight (frozen, 完整未分片)。
    Returns:
        [B, T] per-token log-prob (fp32)。
    """
    parts = []
    T = hidden.shape[1]
    for s in range(0, T, _LOGP_CHUNK_SIZE):
        e = min(s + _LOGP_CHUNK_SIZE, T)
        lg = F.linear(hidden[:, s:e, :], weight).float()                    # [B, c, V] fp32 (单块临时)
        sel = lg.gather(dim=-1, index=index[:, s:e].unsqueeze(-1)).squeeze(-1)  # [B, c]
        parts.append(sel - torch.logsumexp(lg, dim=-1))                     # [B, c]
    return torch.cat(parts, dim=1)                                          # [B, T]


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

        # ── 精细化阶段耗时 profiler (诊断速度瓶颈 / A-B 各加速改动) ──────────
        # 把一个 optimizer step 拆成: gen(vLLM) / score(PRM+advantage) /
        # fwd.backbone / fwd.head / bwd(含 GC 重算 + NCCL + ZeRO-3 gather) / optim。
        # 每个桶对应一个可调加速旋钮 (见 _print_step_timing 的映射)。
        #   PROFILE_STEPS=N  累计 N 个 step 求平均 (默认 1)
        #   PROFILE_WARMUP=W 先跳过 W 个 step 再测 (默认 1, 避开 FusedAdam/flash JIT 噪声)
        # 仅 rank 0 打印。PROFILE_STEPS=0 时退化为旧版"只测第 1 step"。
        self._timing_log: dict[str, float] = {}
        self._timing_fwd_bwd_count: int = 0          # 当前 opt step 内已累计的 micro 数
        self._timing_done: bool = False
        self._prof_total = int(os.environ.get("PROFILE_STEPS", "1"))   # 测几个 step
        self._prof_warmup = int(os.environ.get("PROFILE_WARMUP", "1")) # 先跳几个 step
        self._prof_opt_seen = 0                       # 已完成的 optimizer step 数
        self._prof_measured = 0                       # 已计入统计的 step 数
        self._optim_wrapped = False
        self._optim_timer_ok = False

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
        _record = self._prof_on
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

        # ── vLLM logprobs + mask_hits broadcast → batch["vllm_per_token_logps"] / batch["mask_hits"] ──
        # train.py 已 patch VLLMClient.generate 始终 return_logprobs=True + return_mask_hits=True
        # 并把 logprobs + mask_hits 缓存到 vllm_client (只 main process 有值, 因为 TRL 父类
        # 只在 main process 调 generate). 立刻在 super() 之后取出并 broadcast 给所有 rank.
        try:
            if self.accelerator.is_main_process:
                lp_all = getattr(self.vllm_client, "_last_logprobs", None)
                mh_all = getattr(self.vllm_client, "_last_mask_hits", None)
            else:
                lp_all = None
                mh_all = None
            # 一起 broadcast 节省通信
            payload = [lp_all, mh_all]
            broadcast_object_list(payload, from_process=0)
            lp_all_global, mh_all_global = payload  # 各自 list[list[...]] 或 None

            vllm_lp_tensor = torch.zeros(B, T, device=completion_ids.device,
                                          dtype=torch.float32)
            mask_hits_tensor = torch.zeros(B, T, device=completion_ids.device,
                                           dtype=torch.float32)  # 0/1
            if lp_all_global is not None:
                rank = self.accelerator.process_index
                my_start = rank * B
                for j in range(B):
                    global_idx = my_start + j
                    if global_idx >= len(lp_all_global):
                        continue
                    lp_j = lp_all_global[global_idx]
                    if lp_j is None or len(lp_j) == 0:
                        continue
                    actual_len = min(len(lp_j), T)
                    vllm_lp_tensor[j, :actual_len] = torch.tensor(
                        lp_j[:actual_len], dtype=torch.float32,
                        device=completion_ids.device,
                    )
                    # mask_hits 同长度 (跟 lp 配对), 可能为 None (server 未启用 mask)
                    if mh_all_global is not None and global_idx < len(mh_all_global):
                        mh_j = mh_all_global[global_idx]
                        if mh_j is not None and len(mh_j) > 0:
                            mh_len = min(len(mh_j), T)
                            mask_hits_tensor[j, :mh_len] = torch.tensor(
                                [1.0 if x else 0.0 for x in mh_j[:mh_len]],
                                dtype=torch.float32,
                                device=completion_ids.device,
                            )
                batch["vllm_per_token_logps"] = vllm_lp_tensor
                batch["mask_hits"] = mask_hits_tensor

                # ── Mask sanity check (use_mask=True 但 server 端没启 mask) ──
                # vLLM server 没传 --mask_enabled 时 mh_all_global=None → mask_hits 全 0.
                # Trainer config.use_mask=True 意味着用户预期 mask 生效, 不一致就警告.
                # 只在主进程 print 避免多 rank 重复输出.
                if (getattr(config, "use_mask", False)
                        and self.accelerator.is_main_process
                        and not getattr(self, "_mask_sanity_done", False)):
                    self._mask_sanity_done = True
                    server_mask_active = (mh_all_global is not None)
                    if not server_mask_active:
                        print("⚠️ config.use_mask=True 但 vLLM server 未返回 mask_hits "
                              "(server 启动可能缺 --mask_enabled --mask_n N); "
                              "训练仍会继续, 但实际没有 mask 强制. "
                              "请检查 run script 的 vLLM 启动参数.",
                              flush=True)
                    else:
                        print(f"✓ vLLM mask processor 已就绪 (use_mask={config.use_mask}, "
                              f"mask_n={getattr(config, 'mask_n', 0)})",
                              flush=True)
        except Exception as _e:
            if not getattr(self, "_vllm_logprob_warn", False):
                print(f"⚠️ vLLM logprobs/mask_hits 处理失败 (IS 校正退化为不校正): "
                      f"{type(_e).__name__}: {_e}", flush=True)
                self._vllm_logprob_warn = True

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

        # skip_special_tokens=False: Qwen3-Thinking 的 </think> (id 151668) 是
        # special token, skip=True 会抹掉, mask sanity check / [R*,*] 段匹配
        # 无从看 think 边界. R1-Distill 的 </think> 是普通 BPE 不受影响.
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=False
        )
        completions_text = [_strip_chat_specials(t) for t in completions_text]
        instances = [self._deserialize_instance(pd) for pd in problem_data_list]

        # ── 周期诊断: 每 N step dump 2 条 completion 看 think 段形态演变 ────
        # 用途: 跟踪 reward hacking pattern. 7362 run 中 step 80-95 期间 p95
        # completion 长度从 3300 跳到 4071 触上限, parse rate 才随后崩溃 - 怀疑
        # 模型先学会写超长 think 段 (feasible list 冗长 / [R*,*] 段内文本膨胀)
        # 挤掉答案区. 需要看崩溃中期模型实际写什么定锚机制.
        #
        # dump 2 条 (idx=0 + idx=num_gen-1): 同 prompt 不同 sample, 方便对比
        # reward hacking 是否在所有 generation 上都出现 (全组同质化 vs 单条 outlier).
        _dump_interval = 10
        _cur_step = (
            self.state.global_step
            if hasattr(self, "state") and self.state is not None
            else 0
        )
        # gradient_accumulation_steps>1 时同 global_step 内此函数被调用多次,
        # 用 _last_dump_step 守卫避免重复 dump (否则每 10 step 实际 dump grad_accum 次).
        _last_dump_step = getattr(self, "_last_dump_step", -1)
        if (self.accelerator.is_main_process
                and len(completions_text) > 0
                and _cur_step % _dump_interval == 0
                and _cur_step != _last_dump_step):
            self._last_dump_step = _cur_step
            import re as _re
            _dump_indices = [0]
            if B > 1:
                # 取本 rank 内的第 num_gen-1 条 (若 B < num_gen 则取最后一条)
                _dump_indices.append(min(num_gen - 1, B - 1))
            for _di in _dump_indices:
                txt = completions_text[_di]
                has_think_open  = "<think>" in txt
                has_think_close = "</think>" in txt
                te = txt.find("</think>")
                br_matches = _re.findall(r'\[R?\d+\s*,\s*\d+\]', txt)
                # parse 检查 (terminal_reward 视角): 答案区能否解出多路线
                from utils.parse import parse_multi_route, parse_single_route
                _pt = problem_type_list[_di]
                _n = instances[_di]["n"]
                if _pt in ("cvrp", "vrptw"):
                    _parse_ok = parse_multi_route(txt, _n) is not None
                else:
                    _parse_ok = parse_single_route(txt, _n) is not None
                # 答案区长度 (</think> 之后)
                _answer_len = len(txt) - (te + len("</think>")) if te >= 0 else 0
                print(
                    f"\n[COMPLETION_DUMP step={_cur_step} idx={_di}] "
                    f"len={len(txt)} answer_len={_answer_len} "
                    f"has<think>={has_think_open} has</think>={has_think_close} "
                    f"bracket_count={len(br_matches)} parse_ok={_parse_ok}\n"
                    f"--- think 末尾前 800 char (</think> 之前) ---\n"
                    f"{txt[max(0, te-800):te] if te > 0 else '(no </think>)'}\n"
                    f"--- 答案区前 400 char (</think> 之后) ---\n"
                    f"{txt[te+len('</think>'):te+len('</think>')+400] if te >= 0 else '(no answer)'}\n"
                    f"--- 末尾 200 char (看是否触 max_len 截断) ---\n{txt[-200:]}\n"
                    f"--- 前 10 个 [R*,*] 标记 ---\n{br_matches[:10]}",
                    flush=True,
                )

        if config.reward_mode == "foarl":
            advantages = self._build_foarl_advantages(
                completions_text, instances, problem_type_list,
                B, num_gen, device=completion_ids.device,
            )
            # FOARL 返回 (B,) → 扩展到 (B, T) 保持接口一致
            advantages = advantages.unsqueeze(-1).expand(-1, T).contiguous()
        else:
            # 可行性重采样：每组至少 2 条可行解 (v3/v4 共用)
            self._resample_infeasible(
                batch, completions_text, instances, problem_type_list, num_gen
            )

            # v3 (默认, 原 hardgate+cascade) | v4 (simplified+absolute PRM) |
            # v5 (v4 + hardgate distance + cov/cons 加权)
            _scheme = getattr(config, "reward_scheme", "v3")
            if _scheme == "v5":
                advantages = self._build_unified_advantages_v5(
                    completions_text, instances, problem_type_list,
                    B, num_gen, T, device=completion_ids.device,
                    prompt_ids=batch["prompt_ids"],
                    prompt_mask=batch["prompt_mask"],
                    mask_hits=batch.get("mask_hits"),
                    completion_mask=completion_mask,
                )
            elif _scheme == "v4":
                advantages = self._build_unified_advantages_v4(
                    completions_text, instances, problem_type_list,
                    B, num_gen, T, device=completion_ids.device,
                    prompt_ids=batch["prompt_ids"],
                    prompt_mask=batch["prompt_mask"],
                )
            else:
                # v3: 原逻辑一字不改
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

        # completion 长度分布 (跨 rank gather, 用于判断 max_completion_length 余量)
        # p95 是关键: 若长期 >0.9 × max_completion_length, 截断率上升, 需扩 max_len
        # 或砍 think 长度. p50 反映典型长度, max 反映极值.
        try:
            comp_len_local = completion_mask.sum(-1).float().contiguous()
            all_lens = self.accelerator.gather(comp_len_local).cpu().numpy()
            self.log({
                "completion/p50": float(np.percentile(all_lens, 50)),
                "completion/p95": float(np.percentile(all_lens, 95)),
                "completion/max": float(all_lens.max()),
            })
        except Exception as _e:
            # log 失败不应阻塞训练
            if not getattr(self, "_completion_log_warned", False):
                print(f"⚠️ completion length log 失败: {_e}", flush=True)
                self._completion_log_warned = True

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
                        completions_text[i], instances[i], problem_type_list[i],
                        cov_gate=config.cov_gate,
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
                        completions_text[i], instances[i], problem_type_list[i],
                        cov_gate=config.cov_gate,
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
                # skip_special_tokens=False: 同 batch_decode, 保 Qwen3 </think>
                completions_text[idx] = _strip_chat_specials(
                    self.processing_class.decode(
                        token_ids, skip_special_tokens=False,
                    )
                )

            total_resampled += len(local_requests)

        for g in range(B // num_gen):
            s = g * num_gen
            n_feas = sum(
                1 for i in range(s, s + num_gen)
                if is_fully_feasible(
                    completions_text[i], instances[i], problem_type_list[i],
                    cov_gate=config.cov_gate,
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

            # ── step 1: rank-local 算 ThinkPRMResult (所有 completion) ──
            # 不再用 is_feasible gating: PRM 的同组条件是"同 step_idx 上跨 trajectory
            # 有 ≥2 条 normal", 不是"trajectory 整体可行". 不可行 completion 的 think
            # 链路前 N 步可能仍是 valid prefix (compute_think_step_rewards 内部用
            # _validate_prefix 切出 effective_anomaly), 这部分 step 应该参与跨
            # trajectory 比较, 不能丢掉.
            # 即使整条 completion 完全无 valid prefix (step 0 就违规), PRM 也会返回
            # normal_steps=[]+anomaly_step_indices=全部 的 result, 自动走 fallback
            # anomaly 路径, 不会污染信号.
            rank_prm_results: list[ThinkPRMResult | None] = []
            for i in range(B):
                pt = problem_type_list[i]
                if pt in self.pomo_prm.SUPPORTED:
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

            # ── step 3: 按 (prompt_hash, step_index) 跨 rank z-score ───
            # 收集每个 (p_hash, step_idx) 的正常 reward 列表 + anomaly global_idx 集合.
            # key 用 step_index (think 段内第几个 customer step), 不是 customer_id:
            #   - 跨 trajectory 同 step_index 表示"经过相同步数后的决策点", prefix
            #     长度一致, POMO state 量级可比.
            #   - 用 customer_id 会让 "≥2 可行但选的 customer 不重叠" 时整组 pairs=1,
            #     导致 PRM 信号完全失活 (n_zscore=0 的根因).
            normal_rewards: dict = defaultdict(list)   # (p_hash, step_idx) -> [(gidx, reward), ...]
            anomaly_lookup: dict = defaultdict(set)    # (p_hash, step_idx) -> {gidx, ...}

            for gidx, prm_res in enumerate(all_prm_results):
                if prm_res is None:
                    continue
                p_hash = all_row_hash[gidx]
                for step_idx, r in prm_res.step_rewards.items():
                    normal_rewards[(p_hash, step_idx)].append((gidx, r))
                for step_idx in prm_res.anomaly_step_indices:
                    anomaly_lookup[(p_hash, step_idx)].add(gidx)

            # 对每个 (p_hash, step_idx) 算信号, 写 normalized_proc[gidx][step_idx] = a_proc.
            # 三条路径:
            #   ≥2 normal + std>0    → z-score (anomaly 用 z(min) - σ, bounded)
            #   ≥2 normal + std=0    → normal 退化为 0 (无区分度), anomaly 用 fallback 常数
            #   <2 normal            → fallback 常数 (含单可行 / trajectory 长度差异 / 孤儿 anomaly)
            # 不再用 std_floor: 新 anomaly 公式 z(min) - σ 与 std 解耦, 不会爆;
            # POMO 增量 reward 天然在 0.1-1 量级 (实测 std_raw_mean ≈ 0.3),
            # std_floor=1e-6 在这个量级下会误伤正常信号.
            # 遍历 normal_rewards ∪ anomaly_lookup 的 keys, 防止"孤儿 anomaly"被漏掉.
            normalized_proc: dict[int, dict[int, float]] = defaultdict(dict)
            all_keys = set(normal_rewards.keys()) | set(anomaly_lookup.keys())
            for key in all_keys:
                p_hash, step_idx = key
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
                            normalized_proc[gidx][step_idx] = (r - mean_c) / std_c
                        for gidx in anomaly_gidx_set:
                            if step_idx not in normalized_proc.get(gidx, {}):
                                normalized_proc[gidx][step_idx] = (abnormal_val - mean_c) / std_c
                        n_with_zscore += 1
                    else:
                        # 完全退化 (所有 normal reward 精确相等): normal 无信号, anomaly 仍要罚
                        for gidx in anomaly_gidx_set:
                            if step_idx not in normalized_proc.get(gidx, {}):
                                normalized_proc[gidx][step_idx] = config.fallback_anomaly_value
                        n_with_fallback += 1
                else:
                    # Fallback: 单条 normal / trajectory 长度差异 / 孤儿 anomaly, 用绝对常数
                    for gidx, _ in pairs:
                        normalized_proc[gidx][step_idx] = config.fallback_normal_value
                    for gidx in anomaly_gidx_set:
                        if step_idx not in normalized_proc.get(gidx, {}):
                            normalized_proc[gidx][step_idx] = config.fallback_anomaly_value
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
                for step_idx, a_proc in normalized_proc[gidx].items():
                    seg = prm_res.step_ranges.get(step_idx)
                    if seg is None:
                        continue
                    seg_cs, seg_ce = seg
                    tok_s, tok_e = self._char_to_token_range(seg_cs, seg_ce, om)
                    if tok_s is not None and tok_e is not None:
                        tok_e = min(tok_e, T)
                        # 段广播 mean (而非 sum): a_proc 平均分摊到段内 token,
                        # 段对 loss 总贡献 = α × a_proc (与段长解耦), 防 length-runaway:
                        # 原 sum 模式下 段贡献 = α × a_proc × seg_len, 模型有边际激励
                        # 写更冗长的 feasible list 以放大每段的 advantage 总量,
                        # 导致 think 段顶死 max_completion_length 截断答案区.
                        # 注意: 量级整体压到原来的 1/seg_len (~1/100), 若需保持 PRM
                        # 在 loss 中权重, 同步上调 config.proc_alpha (建议 50 起步).
                        seg_len = max(tok_e - tok_s, 1)
                        advantages[j, tok_s:tok_e] += (
                            config.proc_alpha * a_proc / seg_len
                        )

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
        # R_coverage_rate 含义随 reward v3 变化:
        #   v2 (hinge cov)  → 全覆盖比例 (mean of {0,1})
        #   v3 (连续 cov)   → 平均覆盖率 (mean of [0,1])
        # 为了直接跟 v2 对比 + 监控硬墙开门频率, 加两个独立 metric:
        #   fullcov_rate  = mean(cov == 1.0)         ← 对应 v2 含义
        #   gate_open_rate = mean(cov >= cov_gate)   ← 硬墙开门率, cons 信号生效比例
        coverage_arr = np.array([c["coverage"] for c in components])
        fullcov_rate    = float((coverage_arr >= 1.0 - 1e-9).mean())
        gate_open_rate  = float((coverage_arr >= config.cov_gate).mean())
        log_dict = {
            "reward/feasibility_rate":    self._gather_mean(feas_rate),
            "reward/R_parse_rate":        self._gather_mean(
                np.mean([c["parse"] for c in components])),
            "reward/R_coverage_rate":     self._gather_mean(
                np.mean([c["coverage"] for c in components])),
            "reward/R_fullcov_rate":      self._gather_mean(fullcov_rate),
            "reward/gate_open_rate":      self._gather_mean(gate_open_rate),
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

    # ══════════════════════════════════════════════════════════════════
    #  v4: simplified advantage (absolute PRM + repaired distance)
    #  完全独立分支, 不调 v3 函数, 不污染 v3 行为. config.reward_scheme="v4" 触发.
    # ══════════════════════════════════════════════════════════════════

    def _build_unified_advantages_v4(self, completions_text, instances,
                                      problem_type_list,
                                      B, num_gen, T, device,
                                      prompt_ids, prompt_mask):
        """v4 advantage 构造.

        差异 vs v3 (_build_unified_advantages):
            1. A_out 用 _compute_a_out_v4: A_feas 只剩 parse+format, A_outcome 用
               repaired_distance, feas 子集 = parse=1 (大子集, z-score 噪声小).
            2. PRM 段广播用 absolute: a_proc = prm_base + tanh(R_step) (始终 > 0),
               违例/重复及之后 step 不在 step_rewards 里, 自动游离 (机会成本).
            3. 不做跨 trajectory z-score, 不需要 fallback / anomaly_value 等.
            4. mean 模式段广播 (sum 改 mean 防 length runaway): 段贡献 = α × a_proc.
        """
        # ── 1. A_out (v4 简化版) ─────────────────────────────────────
        a_out_pkg = self._compute_a_out_v4(
            completions_text, instances, problem_type_list, B, num_gen, device,
            prompt_ids, prompt_mask,
        )
        a_out = a_out_pkg["a_out"]
        is_feasible = a_out_pkg["is_feasible"]
        components = a_out_pkg["components"]
        a_feas_norm = a_out_pkg["a_feas_norm"]
        a_outcome_norm = a_out_pkg["a_outcome_norm"]
        distances_local = a_out_pkg["distance_local"]
        dist_std_per_group = a_out_pkg["dist_std_per_group"]
        dist_cv_per_group = a_out_pkg["dist_cv_per_group"]

        # ── 2. 初始化 advantage tensor: 所有 token 先赋 A_out ────────
        advantages = a_out.unsqueeze(-1).expand(-1, T).contiguous()

        # ── 3. PRM 段广播 (absolute, rank-local) ─────────────────────
        prm_base = getattr(config, "prm_base_v4", 1.5)
        proc_alpha = getattr(config, "proc_alpha_v4", 10.0)

        n_segments_total = 0
        a_proc_sum = 0.0
        a_proc_min = float("inf")
        # R_step 原始 (tanh 之前) 统计, 看 tanh 饱和率
        R_step_raw_abs_sum = 0.0
        R_step_raw_abs_max = 0.0
        R_step_saturated = 0   # |R_step_raw| > 2 算饱和 (tanh(2)≈0.96)
        R_step_total = 0
        # 段内 vs 段外 token advantage 对比, 看 PRM 段广播实际影响
        seg_token_count = 0      # 受 PRM 段广播影响的 token 数 (跨 batch)

        if not config.disable_prm and self.pomo_prm is not None:
            offset_maps = self._build_offset_maps(completions_text, B)
            for i in range(B):
                pt = problem_type_list[i]
                if pt not in self.pomo_prm.SUPPORTED:
                    continue
                prm_res = self.pomo_prm.compute_think_step_rewards_v4(
                    completions_text[i], instances[i], pt,
                )
                if prm_res is None or offset_maps[i] is None:
                    continue
                om = offset_maps[i]

                # 跟踪 trajectory 是否完整 (违例/重复触发的 trajectory step_rewards 截断)
                n_segments_total += len(prm_res.step_rewards)
                # R_step raw 统计 (诊断 tanh 饱和)
                for step_idx, R_raw in prm_res.raw_step_rewards.items():
                    R_step_raw_abs_sum += abs(R_raw)
                    if abs(R_raw) > R_step_raw_abs_max:
                        R_step_raw_abs_max = abs(R_raw)
                    if abs(R_raw) > 2.0:
                        R_step_saturated += 1
                    R_step_total += 1
                for step_idx, R_step_tanh in prm_res.step_rewards.items():
                    a_proc = prm_base + R_step_tanh   # ∈ (prm_base - 1, prm_base + 1)
                    a_proc_sum += a_proc
                    if a_proc < a_proc_min:
                        a_proc_min = a_proc
                    seg = prm_res.step_ranges.get(step_idx)
                    if seg is None:
                        continue
                    tok_s, tok_e = self._char_to_token_range(seg[0], seg[1], om)
                    if tok_s is None or tok_e is None:
                        continue
                    tok_e = min(tok_e, T)
                    seg_len = max(tok_e - tok_s, 1)
                    seg_token_count += seg_len
                    # mean 模式: 段贡献 = α × a_proc (不依赖 seg_len, 防 length runaway)
                    advantages[i, tok_s:tok_e] += proc_alpha * a_proc / seg_len

        # ── 4. 错误类型统计 (核心避险诊断) ──────────────────────────
        # 解析 routes 算 miss/violate/dup 频次, 直接看模型策略选择.
        # 期望 v4 跑出来 miss/violate 比值反转 (v3 是 miss >> violate, v4 应近似或 miss < violate).
        from terminal_reward import route_stats
        from utils.parse import parse_multi_route, parse_single_route
        miss_list, violate_list, dup_list = [], [], []
        for i in range(B):
            pt = problem_type_list[i]
            inst = instances[i]
            n_inst = inst["n"]
            if pt == "cvrp":
                routes = parse_multi_route(completions_text[i], n_inst)
                if routes is None:
                    continue
                demands = inst.get("demands", [0.0] * (n_inst + 1))
                cap = inst.get("capacity", 1.0)
                stats = route_stats(routes, n_inst, demands, cap)
                miss_list.append(stats["n_missing"])
                violate_list.append(stats["n_violate_routes"])
                dup_list.append(stats["n_duplicates"])

        # ── 5. Log (v4 metric, 跟 v3 解耦, prm 统计跨 rank gather) ────
        # fully_feas_rate: 严格 fully feasible (parse+全访+全合规+format=1.0),
        # 跟 v3 的 feasibility_rate 定义一致, 方便跟 7362 run 对照看 fullcov 是否反弹.
        fully_feas_per_traj = [
            float(c["parse"] == 1.0
                  and c["coverage"] >= 1.0 - 1e-9
                  and c["constraint"] == 1.0
                  and c["format"] == 1.0)
            for c in components
        ]
        coverage_arr = np.array([c["coverage"] for c in components])
        a_proc_mean = a_proc_sum / max(n_segments_total, 1) if n_segments_total > 0 else 0.0
        # R_step raw 统计 (诊断 tanh 是否过度压缩极端信号)
        R_step_raw_abs_mean = (R_step_raw_abs_sum / R_step_total) if R_step_total > 0 else 0.0
        R_step_saturation_rate = (R_step_saturated / R_step_total) if R_step_total > 0 else 0.0
        # a_out 分解: 看 a_feas vs a_outcome 谁主导 (绝对值 mean)
        a_feas_abs = float(a_feas_norm.abs().mean().item())
        a_outcome_abs = float(a_outcome_norm.abs().mean().item())
        feas_dominance = a_feas_abs / max(a_feas_abs + a_outcome_abs, 1e-8)
        # 有效 distance (非 nan) 跨 trajectory 平均
        valid_dist = [d for d in distances_local if not (d != d)]  # filter nan
        distance_mean_local = float(np.mean(valid_dist)) if valid_dist else 0.0

        log_dict = {
            "reward_v4/parse_rate":         self._gather_mean(
                np.mean([c["parse"] for c in components])),
            "reward_v4/coverage_rate":      self._gather_mean(float(coverage_arr.mean())),
            "reward_v4/fullcov_rate":       self._gather_mean(
                float((coverage_arr >= 1.0 - 1e-9).mean())),
            "reward_v4/fully_feas_rate":    self._gather_mean(np.mean(fully_feas_per_traj)),
            "reward_v4/R_constraint_mean":  self._gather_mean(
                np.mean([c["constraint"] for c in components])),
            "reward_v4/R_format_mean":      self._gather_mean(
                np.mean([c["format"] for c in components])),
            "reward_v4/A_abs_mean":         self._gather_mean(advantages.abs().mean()),
            "reward_v4/A_std":              self._gather_mean(advantages.std()),
            # 错误类型分布 (避险诊断的核心: v3 miss>>violate, v4 期望反转)
            "stats_v4/miss_per_traj":       self._gather_mean(
                float(np.mean(miss_list)) if miss_list else 0.0),
            "stats_v4/violate_per_traj":    self._gather_mean(
                float(np.mean(violate_list)) if violate_list else 0.0),
            "stats_v4/dup_per_traj":        self._gather_mean(
                float(np.mean(dup_list)) if dup_list else 0.0),
            # outcome distance 方差 (cv > 0.3 提示 distance 不稳定, 考虑归一化)
            "outcome_v4/distance_mean":     self._gather_mean(distance_mean_local),
            "outcome_v4/distance_std_per_group": self._gather_mean(
                float(np.mean(dist_std_per_group)) if dist_std_per_group else 0.0),
            "outcome_v4/distance_cv":       self._gather_mean(
                float(np.mean(dist_cv_per_group)) if dist_cv_per_group else 0.0),
            # A_out 分解 (feas_dominance > 0.7 说明 A_feas 主导, < 0.3 说明 A_outcome 主导)
            "a_out_v4/a_feas_abs_mean":     self._gather_mean(a_feas_abs),
            "a_out_v4/a_outcome_abs_mean":  self._gather_mean(a_outcome_abs),
            "a_out_v4/feas_dominance":      self._gather_mean(feas_dominance),
            # PRM 段广播统计
            "prm_v4/n_segments_per_traj":   self._gather_mean(
                float(n_segments_total) / max(B, 1)),
            "prm_v4/seg_token_ratio":       self._gather_mean(
                float(seg_token_count) / max(B * T, 1)),  # 段内 token 占 completion 比例
            "prm_v4/a_proc_mean":           self._gather_mean(float(a_proc_mean)),
            # a_proc_min 用 sentinel +inf 表示本 rank 无 PRM 数据 (跨 rank mean 会被拉高,
            # 但全部 rank 都 inf 时取 0 退化). 实际监控只要看是否 > 0 即可.
            "prm_v4/a_proc_min":            self._gather_mean(
                float(a_proc_min) if a_proc_min < float("inf") else 0.0),
            # R_step raw 统计 (tanh 之前): saturation_rate > 0.1 说明 tanh 频繁饱和,
            # 极端信号被压扁; raw_max >> 2 说明 R_step 偶尔有极端值, 关注是否影响训练
            "prm_v4/R_step_raw_abs_mean":   self._gather_mean(R_step_raw_abs_mean),
            "prm_v4/R_step_raw_abs_max":    self._gather_mean(R_step_raw_abs_max),
            "prm_v4/R_step_saturation_rate": self._gather_mean(R_step_saturation_rate),
            # 实验区分: use_mask 是否启用 (常量, 方便 WandB 跨 run 对照)
            "train/use_mask":               1.0 if getattr(config, "use_mask", False) else 0.0,
        }
        self.log(log_dict)
        return advantages

    def _compute_a_out_v4(self, completions_text, instances,
                           problem_type_list, B, num_gen, device,
                           prompt_ids, prompt_mask):
        """v4 A_out: A_feas (parse+format only) + A_outcome (repaired distance).

        差异 vs v3 _compute_a_out:
            - A_feas_raw 只剩 parse + format 加权 (cov/cons 没了, 因为 outcome 已覆盖)
            - feas 子集判定: parse=1 即可进 (子集变大, z-score 信号更稳)
            - distance 用 repaired_distance (漏访补全 + 违例拆分 + 重复 ε)
        """
        from collections import defaultdict
        from terminal_reward import (
            compute_terminal_components, repaired_distance,
        )
        from utils.parse import parse_multi_route, parse_single_route

        eps = 1e-8
        w_p = getattr(config, "w_p_v4", 1.0)
        w_f = getattr(config, "w_f_v4", 0.5)
        dup_eps = getattr(config, "dup_distance_eps", 0.2)

        # ── 1. 本 rank 算 raw 信号 ──────────────────────────────────
        a_feas_raw = torch.zeros(B, device=device)
        is_feasible_local: list[bool] = []
        components: list[dict] = []
        distances_local: list[float] = []

        for i in range(B):
            c = compute_terminal_components(
                completions_text[i], instances[i], problem_type_list[i]
            )
            components.append(c)

            # v4 feas 判定: 只要 parse 成功就进 outcome 子集
            feas = c["parse"] == 1.0
            is_feasible_local.append(feas)

            # A_feas 只剩 parse + format (cov/cons 完全去除)
            a_feas_raw[i] = w_p * c["parse"] + w_f * c["format"]

            if feas:
                inst = instances[i]
                pt = problem_type_list[i]
                n = inst["n"]
                # v4 repaired_distance 只对 CVRP 设计 (拆分/补全/重复都跟容量+多路线相关).
                # 其他问题类型 (TSP/TSPTW/TSPDL/VRPTW) 退化到 v3 的 prob.get_tour_distance.
                if pt == "cvrp":
                    routes = parse_multi_route(completions_text[i], n)
                    if routes is None:
                        distances_local.append(float("nan"))
                        continue
                    demands = inst.get("demands", [0.0] * (n + 1))
                    cap = inst.get("capacity", 1.0)
                    coords = inst["coords"]
                    try:
                        d = repaired_distance(routes, coords, n, demands, cap, dup_eps)
                        distances_local.append(d)
                    except Exception:
                        distances_local.append(float("nan"))
                else:
                    prob = _PROBLEM_OBJS.get(pt)
                    d = (prob.get_tour_distance(completions_text[i], inst)
                         if prob else None)
                    distances_local.append(d if d is not None else float("nan"))
            else:
                distances_local.append(float("nan"))

        # ── 2. prompt hash (同 v3) ───────────────────────────────────
        masked_ids = prompt_ids.long() * prompt_mask.long()
        row_id_sum = masked_ids.sum(dim=-1)
        row_len    = prompt_mask.long().sum(dim=-1)
        row_hash   = row_id_sum * 1000003 + row_len

        # ── 3. 跨 rank gather (同 v3) ────────────────────────────────
        all_a_feas_raw = self.accelerator.gather(a_feas_raw.contiguous())
        feas_t  = torch.tensor(is_feasible_local, device=device, dtype=torch.bool)
        all_feas_t = self.accelerator.gather(feas_t.contiguous())
        dist_t  = torch.tensor(distances_local, device=device, dtype=torch.float32)
        all_dist_t = self.accelerator.gather(dist_t.contiguous())
        all_row_hash = self.accelerator.gather(row_hash.contiguous())
        G = all_a_feas_raw.shape[0]

        # ── 4. 分组 z-score ──────────────────────────────────────────
        hash_cpu = all_row_hash.cpu().tolist()
        groups = defaultdict(list)
        for i, h in enumerate(hash_cpu):
            groups[h].append(i)

        all_a_feas_norm    = torch.zeros_like(all_a_feas_raw)
        all_a_outcome_norm = torch.zeros_like(all_a_feas_raw)

        for h, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)

            # A_feas: 整组 z-score
            grp = all_a_feas_raw[idx_t]
            all_a_feas_norm[idx_t] = (grp - grp.mean()) / (grp.std() + eps)

            # A_outcome: parse=1 子集 z-score(-distance), distance 短 → 正信号
            feas_mask = all_feas_t[idx_t] & ~torch.isnan(all_dist_t[idx_t])
            feas_local_pos = feas_mask.nonzero(as_tuple=True)[0]
            if feas_local_pos.numel() >= 2:
                feas_idx_t = idx_t[feas_local_pos]
                neg_d = -all_dist_t[feas_idx_t]
                all_a_outcome_norm[feas_idx_t] = (
                    (neg_d - neg_d.mean()) / (neg_d.std() + eps)
                )

        # ── 5. 切回本 rank ──────────────────────────────────────────
        rank = self.accelerator.process_index
        my_start = rank * B
        a_feas_norm_local = all_a_feas_norm[my_start:my_start + B]
        a_outcome_norm_local = all_a_outcome_norm[my_start:my_start + B]
        a_out = a_feas_norm_local + a_outcome_norm_local

        # ── 6. 跨组 distance 统计 (用于监控 outcome 方差) ────────────
        # 每组 distance std + cv (= std/mean), 反映"per-trajectory distance"
        # 的离散程度. cv > 0.3 说明 repaired distance 方差大, 可能需要归一化.
        dist_std_per_group = []
        dist_cv_per_group = []
        for h, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
            valid_d = all_dist_t[idx_t][~torch.isnan(all_dist_t[idx_t])]
            if valid_d.numel() >= 2:
                std = float(valid_d.std().item())
                mean = float(valid_d.mean().item())
                dist_std_per_group.append(std)
                if mean > 1e-6:
                    dist_cv_per_group.append(std / mean)

        return {
            "a_out":                  a_out,
            "is_feasible":            is_feasible_local,
            "components":             components,
            "a_feas_norm":            a_feas_norm_local,
            "a_outcome_norm":         a_outcome_norm_local,
            "distance_local":         distances_local,
            "dist_std_per_group":     dist_std_per_group,
            "dist_cv_per_group":      dist_cv_per_group,
        }

    # ══════════════════════════════════════════════════════════════════
    #  v5: v4 + hardgate distance + cov/cons 加权 A_feas
    #  完全独立分支, 跟 v3/v4 解耦. config.reward_scheme="v5" 触发.
    #  设计意图 (用户决定): v4 7414 run 信号弱, A_feas=parse+format 同质化
    #  z-score=0; 改回 A_feas=parse+cov+cons(hardgate)+format 提供持续 contrastive,
    #  outcome 改用 raw distance + strict fully_feasible 子集 (子集 <2 时全 0,
    #  完全靠 A_feas + PRM 推可行性).
    # ══════════════════════════════════════════════════════════════════

    def _build_unified_advantages_v5(self, completions_text, instances,
                                      problem_type_list,
                                      B, num_gen, T, device,
                                      prompt_ids, prompt_mask,
                                      mask_hits=None, completion_mask=None):
        """v5 advantage 构造: A_feas (hardgate) + A_outcome (strict subset) + PRM (v4 absolute).

        mask_hits / completion_mask: 用于 mask_health 计算 vLLM 端真实触发率,
        作为 cov_eq_1 的独立维度判断 mask 是否在 server 端跑起来.
        """
        a_out_pkg = self._compute_a_out_v5(
            completions_text, instances, problem_type_list, B, num_gen, device,
            prompt_ids, prompt_mask,
        )
        a_out = a_out_pkg["a_out"]
        components = a_out_pkg["components"]
        a_feas_norm = a_out_pkg["a_feas_norm"]
        a_outcome_norm = a_out_pkg["a_outcome_norm"]
        distances_local = a_out_pkg["distance_local"]
        dist_std_per_group = a_out_pkg["dist_std_per_group"]
        dist_cv_per_group = a_out_pkg["dist_cv_per_group"]
        feas_subset_size_per_group = a_out_pkg["feas_subset_size_per_group"]

        # 初始化 advantage tensor: 所有 token 先赋 A_out
        advantages = a_out.unsqueeze(-1).expand(-1, T).contiguous()

        # PRM 段广播 (跟 v4 完全相同, absolute, rank-local)
        prm_base = getattr(config, "prm_base_v4", 1.5)
        proc_alpha = getattr(config, "proc_alpha_v4", 50.0)

        n_segments_total = 0
        a_proc_sum = 0.0
        a_proc_min = float("inf")
        R_step_raw_abs_sum = 0.0
        R_step_raw_abs_max = 0.0
        R_step_saturated = 0
        R_step_total = 0
        seg_token_count = 0
        # 真正注入 advantage 的 per-token 量级累计 (诊断 mean 模式 seg_len 反相关问题):
        # inject_per_tok = proc_alpha * a_proc / seg_len, 段越短单 token 注入越大.
        prm_inject_tok_sum = 0.0     # Σ_token 注入值 (= Σ_seg proc_alpha*a_proc), token 加权
        inject_per_tok_max = 0.0     # 最强单 token 注入 (seg_len 小尖峰)

        # PRM 只对 fully_feasible trajectory 算 (v5 用户决定 2026-05-18):
        # 减少 PRM 偏置 (0.37 → 0.115) + 增强 fully_feas 推力 (差距 ×3.5)
        prm_only_fully_feas = getattr(config, "prm_only_fully_feas_v5", True)
        prm_skipped_non_feas = 0  # 诊断: 多少 trajectory 因为不是 fully_feas 被跳过

        if not config.disable_prm and self.pomo_prm is not None:
            offset_maps = self._build_offset_maps(completions_text, B)
            for i in range(B):
                pt = problem_type_list[i]
                if pt not in self.pomo_prm.SUPPORTED:
                    continue
                # ── 新: 只对 fully_feas trajectory 算 PRM ──
                if prm_only_fully_feas and not a_out_pkg["is_feasible"][i]:
                    prm_skipped_non_feas += 1
                    continue
                prm_res = self.pomo_prm.compute_think_step_rewards_v4(
                    completions_text[i], instances[i], pt,
                )
                if prm_res is None or offset_maps[i] is None:
                    continue
                om = offset_maps[i]
                n_segments_total += len(prm_res.step_rewards)
                for step_idx, R_raw in prm_res.raw_step_rewards.items():
                    R_step_raw_abs_sum += abs(R_raw)
                    if abs(R_raw) > R_step_raw_abs_max:
                        R_step_raw_abs_max = abs(R_raw)
                    if abs(R_raw) > 2.0:
                        R_step_saturated += 1
                    R_step_total += 1
                for step_idx, R_step_tanh in prm_res.step_rewards.items():
                    a_proc = prm_base + R_step_tanh
                    a_proc_sum += a_proc
                    if a_proc < a_proc_min:
                        a_proc_min = a_proc
                    seg = prm_res.step_ranges.get(step_idx)
                    if seg is None:
                        continue
                    tok_s, tok_e = self._char_to_token_range(seg[0], seg[1], om)
                    if tok_s is None or tok_e is None:
                        continue
                    tok_e = min(tok_e, T)
                    seg_len = max(tok_e - tok_s, 1)
                    seg_token_count += seg_len
                    inject_per_tok = proc_alpha * a_proc / seg_len
                    advantages[i, tok_s:tok_e] += inject_per_tok
                    prm_inject_tok_sum += inject_per_tok * seg_len   # = proc_alpha*a_proc
                    if inject_per_tok > inject_per_tok_max:
                        inject_per_tok_max = inject_per_tok

        # 错误类型统计
        from terminal_reward import route_stats
        from utils.parse import parse_multi_route
        miss_list, violate_list, dup_list = [], [], []
        for i in range(B):
            pt = problem_type_list[i]
            inst = instances[i]
            n_inst = inst["n"]
            if pt == "cvrp":
                routes = parse_multi_route(completions_text[i], n_inst)
                if routes is None:
                    continue
                demands = inst.get("demands", [0.0] * (n_inst + 1))
                cap = inst.get("capacity", 1.0)
                stats = route_stats(routes, n_inst, demands, cap)
                miss_list.append(stats["n_missing"])
                violate_list.append(stats["n_violate_routes"])
                dup_list.append(stats["n_duplicates"])

        # Log (v5 metric, 跟 v3/v4 解耦)
        fully_feas_per_traj = [
            float(c["parse"] == 1.0
                  and c["coverage"] >= 1.0 - 1e-9
                  and c["constraint"] == 1.0
                  and c["format"] == 1.0)
            for c in components
        ]
        coverage_arr = np.array([c["coverage"] for c in components])
        a_proc_mean = a_proc_sum / max(n_segments_total, 1) if n_segments_total > 0 else 0.0
        R_step_raw_abs_mean = (R_step_raw_abs_sum / R_step_total) if R_step_total > 0 else 0.0
        R_step_saturation_rate = (R_step_saturated / R_step_total) if R_step_total > 0 else 0.0
        a_feas_abs = float(a_feas_norm.abs().mean().item())
        a_outcome_abs = float(a_outcome_norm.abs().mean().item())
        feas_dominance = a_feas_abs / max(a_feas_abs + a_outcome_abs, 1e-8)
        valid_dist = [d for d in distances_local if not (d != d)]
        distance_mean_local = float(np.mean(valid_dist)) if valid_dist else 0.0

        # 真正注入 advantage 的 per-token PRM 量级 + 与 A_out 的相对大小 (量级诊断)
        prm_inject_per_tok_mean = prm_inject_tok_sum / max(seg_token_count, 1)
        aout_per_tok_abs = float(a_out.abs().mean().item())
        inject_vs_aout = prm_inject_per_tok_mean / (aout_per_tok_abs + 1e-8)

        log_dict = {
            "reward_v5/parse_rate":         self._gather_mean(
                np.mean([c["parse"] for c in components])),
            "reward_v5/coverage_rate":      self._gather_mean(float(coverage_arr.mean())),
            "reward_v5/fullcov_rate":       self._gather_mean(
                float((coverage_arr >= 1.0 - 1e-9).mean())),
            "reward_v5/fully_feas_rate":    self._gather_mean(np.mean(fully_feas_per_traj)),
            "reward_v5/R_constraint_mean":  self._gather_mean(
                np.mean([c["constraint"] for c in components])),
            "reward_v5/R_format_mean":      self._gather_mean(
                np.mean([c["format"] for c in components])),
            "reward_v5/A_abs_mean":         self._gather_mean(advantages.abs().mean()),
            "reward_v5/A_std":              self._gather_mean(advantages.std()),
            "stats_v5/miss_per_traj":       self._gather_mean(
                float(np.mean(miss_list)) if miss_list else 0.0),
            "stats_v5/violate_per_traj":    self._gather_mean(
                float(np.mean(violate_list)) if violate_list else 0.0),
            "stats_v5/dup_per_traj":        self._gather_mean(
                float(np.mean(dup_list)) if dup_list else 0.0),
            # outcome 子集大小 (前期可能 0~1, 后期应该 >=2 启用 A_outcome)
            "outcome_v5/distance_mean":     self._gather_mean(distance_mean_local),
            "outcome_v5/distance_std_per_group": self._gather_mean(
                float(np.mean(dist_std_per_group)) if dist_std_per_group else 0.0),
            "outcome_v5/distance_cv":       self._gather_mean(
                float(np.mean(dist_cv_per_group)) if dist_cv_per_group else 0.0),
            "outcome_v5/feas_subset_size":  self._gather_mean(
                float(np.mean(feas_subset_size_per_group)) if feas_subset_size_per_group else 0.0),
            "outcome_v5/feas_subset_active_rate": self._gather_mean(
                float(np.mean([1.0 if s >= 2 else 0.0 for s in feas_subset_size_per_group]))
                if feas_subset_size_per_group else 0.0),
            "a_out_v5/a_feas_abs_mean":     self._gather_mean(a_feas_abs),
            "a_out_v5/a_outcome_abs_mean":  self._gather_mean(a_outcome_abs),
            "a_out_v5/feas_dominance":      self._gather_mean(feas_dominance),
            "prm_v5/n_segments_per_traj":   self._gather_mean(
                float(n_segments_total) / max(B, 1)),
            "prm_v5/seg_token_ratio":       self._gather_mean(
                float(seg_token_count) / max(B * T, 1)),
            "prm_v5/a_proc_mean":           self._gather_mean(float(a_proc_mean)),
            "prm_v5/a_proc_min":            self._gather_mean(
                float(a_proc_min) if a_proc_min < float("inf") else 0.0),
            # ── 量级诊断: PRM 段广播真正注入 advantage 的 per-token 大小 ──
            # inject_per_tok = proc_alpha * a_proc / seg_len, 强依赖 seg_len.
            # inject_vs_aout > 1 → 段内 PRM 注入压倒 A_out (z-score ~1), 量级失衡;
            # inject_per_tok_max 远大于 mean → 短段尖峰, 信号不均.
            "prm_v5/inject_per_tok_mean":   self._gather_mean(float(prm_inject_per_tok_mean)),
            "prm_v5/inject_per_tok_max":    self._gather_mean(float(inject_per_tok_max)),
            "prm_v5/aout_per_tok_abs":      self._gather_mean(float(aout_per_tok_abs)),
            "prm_v5/inject_vs_aout":        self._gather_mean(float(inject_vs_aout)),
            "prm_v5/R_step_raw_abs_mean":   self._gather_mean(R_step_raw_abs_mean),
            "prm_v5/R_step_raw_abs_max":    self._gather_mean(R_step_raw_abs_max),
            "prm_v5/R_step_saturation_rate": self._gather_mean(R_step_saturation_rate),
            # 新: PRM 跳过 (非 fully_feas) trajectory 比例 (验证 prm_only_fully_feas 行为)
            "prm_v5/skipped_non_feas_rate": self._gather_mean(
                float(prm_skipped_non_feas) / max(B, 1)),
            "train/use_mask":               1.0 if getattr(config, "use_mask", False) else 0.0,
        }

        # ── mask 生效指标 (mask 启用时报送, 跟 stats_v5 解耦) ─────────
        # 跟 stats_v5/miss_per_traj 区别: stats_v5 是 parse 成功 cvrp 子集平均;
        # mask_health 用全 B 条对齐 (parse 失败 trajectory 算 miss=n, dup=0).
        # 这样 cov_eq_1 跟 zero_miss/zero_dup 分母一致, 可对比.
        # 期望 mask 完美工作时全部 → 1.0; < 0.95 说明 mask 没真生效.
        if getattr(config, "use_mask", False):
            # 全 B 条对齐的 miss/dup (parse 失败算全漏)
            mask_miss_full: list[float] = []
            mask_dup_full: list[float] = []
            for i in range(B):
                pt = problem_type_list[i]
                inst = instances[i]
                n_inst = inst["n"]
                if pt != "cvrp":
                    # 非 cvrp: mask 不实现, 跳过 (不影响 cvrp 单 problem run)
                    mask_miss_full.append(0.0)
                    mask_dup_full.append(0.0)
                    continue
                routes = parse_multi_route(completions_text[i], n_inst)
                if routes is None:
                    # parse 失败: 视作 mask 完全没生效该 trajectory → 全漏
                    mask_miss_full.append(float(n_inst))
                    mask_dup_full.append(0.0)
                    continue
                demands = inst.get("demands", [0.0] * (n_inst + 1))
                cap = inst.get("capacity", 1.0)
                stats = route_stats(routes, n_inst, demands, cap)
                mask_miss_full.append(float(stats["n_missing"]))
                mask_dup_full.append(float(stats["n_duplicates"]))

            miss_arr = np.array(mask_miss_full)
            dup_arr = np.array(mask_dup_full)
            parse_arr = np.array([c["parse"] for c in components])

            # 本 rank 算
            cov1_local      = float((coverage_arr >= 1.0 - 1e-9).mean())
            zero_miss_local = float((miss_arr == 0).mean())
            zero_dup_local  = float((dup_arr == 0).mean())
            parse_local     = float(parse_arr.mean())
            # perfect: cov=1 AND parse=1 (cov=1 数学上隐含 no_miss AND no_dup)
            perfect_local   = float(
                ((coverage_arr >= 1.0 - 1e-9) & (parse_arr == 1.0)).mean()
            )
            avg_miss_local  = float(miss_arr.mean())
            avg_dup_local   = float(dup_arr.mean())

            # ── 独立维度: vLLM 端 mask 真实触发率 ────────────────────
            # 关键: 这是跟 cov_eq_1 完全独立的诊断维度.
            # - cov_eq_1 看的是"模型最终输出是否完美"（受 max_length/multi-token/etc 影响）
            # - mask_hit_rate 看的是"vLLM server 端 mask processor 是否真的在 step 拦截"
            # 两者组合诊断:
            #   hit > 0 + cov=1 → ✓ Mask 完美生效
            #   hit > 0 + cov<1 → ⚠️ Mask 在跑但有 fallthrough
            #   hit = 0          → ❌ vLLM server 端 mask 完全没启动 (不管 cov)
            if mask_hits is not None and completion_mask is not None:
                cm_bool = completion_mask.bool()
                # 1. token-level: mask 触发的 token 占有效 completion token 的比例
                if cm_bool.any():
                    mh_valid = mask_hits[cm_bool]
                    mask_hit_rate_local = float(mh_valid.float().mean().item())
                else:
                    mask_hit_rate_local = 0.0
                # 2. trajectory-level: 至少触发过一次 mask 的 trajectory 比例
                #    mask 启用且正常时几乎每条 trajectory 都该触发 (每个 "→ select" 都被规则 1 拦)
                per_traj_hit = (mask_hits * completion_mask.float()).sum(dim=-1)  # (B,)
                traj_with_hit = (per_traj_hit > 0).float()
                traj_hit_rate_local = float(traj_with_hit.mean().item())
            else:
                mask_hit_rate_local = 0.0
                traj_hit_rate_local = 0.0

            # 跨 rank gather (所有 rank 同步调用 NCCL, log+print 共用 gathered 值)
            cov1_g           = self._gather_mean(cov1_local)
            zero_miss_g      = self._gather_mean(zero_miss_local)
            zero_dup_g       = self._gather_mean(zero_dup_local)
            parse_g          = self._gather_mean(parse_local)
            perfect_g        = self._gather_mean(perfect_local)
            avg_miss_g       = self._gather_mean(avg_miss_local)
            avg_dup_g        = self._gather_mean(avg_dup_local)
            mask_hit_rate_g  = self._gather_mean(mask_hit_rate_local)
            traj_hit_rate_g  = self._gather_mean(traj_hit_rate_local)

            log_dict.update({
                # 独立维度 A: vLLM 端 mask 是否真在跑 (跟 cov 输出无关)
                "mask_health/mask_hit_rate_token": mask_hit_rate_g,  # token 级触发率 > 0 即 mask 在 server 端工作
                "mask_health/mask_hit_rate_traj":  traj_hit_rate_g,  # trajectory 级触发率, 期望 ≈ 1.0
                # 独立维度 B: 模型输出是否被 mask 完全约束 (cov_eq_1_rate 跟 fullcov_rate 等价, 不重复 log)
                # 下面 4 个是 cov=1 的细分 (数学上 cov=1 ⟺ no_miss AND no_dup AND parse=1),
                # 保留是为了 fallthrough 时区分哪种失败模式: miss 多 = 规则 4/5 漏, dup 多 = 规则 1 漏
                "mask_health/zero_miss_rate":    zero_miss_g,
                "mask_health/zero_dup_rate":     zero_dup_g,
                "mask_health/parse_rate":        parse_g,
                "mask_health/perfect_rate":      perfect_g,
                "mask_health/avg_miss_per_traj": avg_miss_g,
                "mask_health/avg_dup_per_traj":  avg_dup_g,
            })

            # 第 1 batch 主进程 print 详细 sanity (一次性, 启动验证用)
            # 用 gathered 值, 跨 rank 一致 (避免主 rank 局部 B=6 误判)
            # 判断按"两个独立维度"分类: A) vLLM 端 mask 是否真在跑, B) 输出 cov 是否被完全约束
            if (not getattr(self, "_mask_health_printed", False)
                    and self.accelerator.is_main_process):
                self._mask_health_printed = True
                print(f"\n[MASK_HEALTH] ===== Mask 生效检查 (第 1 batch, cross-rank) =====")
                print(f"  [维度 A: vLLM 端 mask 是否真触发]  (跟 cov 输出无关)")
                print(f"    mask_hit_rate_token        = {mask_hit_rate_g:.4f}   (期望 > 0)")
                print(f"    mask_hit_rate_traj         = {traj_hit_rate_g:.3f}   (期望 ≈ 1.0)")
                print(f"  [维度 B: 模型输出是否被完全约束]")
                print(f"    fullcov_rate (cov=1)         = {cov1_g:.3f}   (期望 ≈ 1.0)")
                print(f"    zero_miss_rate (全 B 对齐)   = {zero_miss_g:.3f}   (期望 ≈ 1.0)")
                print(f"    zero_dup_rate                = {zero_dup_g:.3f}   (期望 ≈ 1.0)")
                print(f"    parse_rate                   = {parse_g:.3f}   (期望 ≈ 1.0)")
                print(f"    perfect_rate (cov=1 ∧ parse) = {perfect_g:.3f}   (期望 ≈ 1.0)")
                print(f"    avg miss/traj                = {avg_miss_g:.3f}   (期望 ≈ 0; parse 失败算 n)")
                print(f"    avg dup/traj                 = {avg_dup_g:.3f}   (期望 ≈ 0)")
                # 两个独立维度组合诊断
                dim_a_ok = mask_hit_rate_g > 0.001  # vLLM 端真触发
                dim_b_ok = cov1_g >= 0.95            # 输出完全约束
                if dim_a_ok and dim_b_ok:
                    print(f"  ✓ Mask 完美生效 (维度 A: server 端在跑; 维度 B: 输出完全约束)")
                elif dim_a_ok and not dim_b_ok:
                    print(f"  ⚠️ Mask 在 server 跑但有 fallthrough (cov_eq_1={cov1_g:.2f} < 0.95)")
                    print(f"     可能原因: max_completion_length 截断 / multi-token customer prefix 共享")
                    print(f"              resample 后 mask_hits 未更新 (已知 issue, 影响 marginal)")
                    if zero_dup_g < 0.95 and zero_miss_g >= 0.95:
                        print(f"     主因疑似: 规则 1 (select 强 mask) 没完全防 dup")
                    elif zero_miss_g < 0.95 and zero_dup_g >= 0.95:
                        print(f"     主因疑似: 规则 4/5 (visited<n 禁 all/Verification) 没完全防 miss")
                elif not dim_a_ok:
                    print(f"  ❌ Mask 未生效: vLLM server 端 mask 完全没触发 (mask_hit_rate=0)")
                    print(f"     检查 1: vLLM server 启动命令是否带 --mask_enabled --mask_n N")
                    print(f"     检查 2: utils/vllm_serve_logprobs.py 启动 log 应有 '[mask] CVRPMaskProcessor 启用'")
                    print(f"     检查 3: train.py 是否打了 '✓ vLLM mask processor 已就绪' 的 sanity 行")
                print(f"  =======================================================\n",
                      flush=True)

        self.log(log_dict)
        return advantages

    def _compute_a_out_v5(self, completions_text, instances,
                           problem_type_list, B, num_gen, device,
                           prompt_ids, prompt_mask):
        """v5 A_out: A_feas (hardgate cov+cons+parse+format) + A_outcome (raw distance on strict subset).

        差异 vs v4 _compute_a_out_v4:
            - A_feas_raw 加回 cov/cons (hardgate: cov >= cov_gate_v5 才给 cons)
            - feas 子集 strict fully_feasible (parse + cov>=gate + cons=1 + format=1)
            - distance 用 raw prob.get_tour_distance (不 repair)
            - 子集 <2 时 A_outcome 全 0 (前期信号靠 A_feas + PRM 机会成本)
        """
        from collections import defaultdict
        eps = 1e-8
        w_p = getattr(config, "w_p_v5", 0.5)
        w_cov = getattr(config, "w_cov_v5", 2.5)
        w_cons = getattr(config, "w_cons_v5", 2.0)
        w_f = getattr(config, "w_f_v5", 0.5)
        cov_gate = getattr(config, "cov_gate_v5", 1.0)

        a_feas_raw = torch.zeros(B, device=device)
        is_feasible_local: list[bool] = []
        components: list[dict] = []
        distances_local: list[float] = []

        for i in range(B):
            c = compute_terminal_components(
                completions_text[i], instances[i], problem_type_list[i]
            )
            components.append(c)

            # strict fully_feasible 子集判定 (修 bug 2026-05-19):
            # 必须强制 cov=1, 跟 cov_gate_v5 解耦. 否则 cov_gate=0 (加法模式) 下,
            # cov 条件恒成立, 漏访客户但 cons=1+parse=1+format=1 的 trajectory 也被
            # 错判为 feasible, 进入 outcome distance 子集. 漏访 trajectory distance
            # 几何上一定更短, z-score(-distance) 后拿正 outcome advantage,
            # 跟 A_feas push cov ↑ 的方向部分抵消 (约 60% 信号衰减).
            # 同时影响 PRM cutoff (跟此 is_feasible 共用), 让 cov<1 也拿 PRM.
            # cov_gate_v5 现在只控制 A_feas 内 cons_signal 是否依赖 cov.
            FEAS_COV_THRESH = 1.0 - 1e-9
            feas = (c["parse"] == 1.0
                    and c["coverage"] >= FEAS_COV_THRESH
                    and c["constraint"] == 1.0
                    and c["format"] == 1.0)
            is_feasible_local.append(feas)

            # A_feas hardgate: cov < cov_gate 时 cons_signal=0 (强迫先冲全覆盖)
            cons_signal = c["constraint"] if c["coverage"] >= cov_gate else 0.0
            a_feas_raw[i] = (w_p * c["parse"]
                             + w_cov * c["coverage"]
                             + w_cons * cons_signal
                             + w_f * c["format"])

            if feas:
                prob = _PROBLEM_OBJS.get(problem_type_list[i])
                d = (prob.get_tour_distance(completions_text[i], instances[i])
                     if prob else None)
                distances_local.append(d if d is not None else float("nan"))
            else:
                distances_local.append(float("nan"))

        # prompt hash + 跨 rank gather (同 v4)
        masked_ids = prompt_ids.long() * prompt_mask.long()
        row_id_sum = masked_ids.sum(dim=-1)
        row_len    = prompt_mask.long().sum(dim=-1)
        row_hash   = row_id_sum * 1000003 + row_len

        all_a_feas_raw = self.accelerator.gather(a_feas_raw.contiguous())
        feas_t  = torch.tensor(is_feasible_local, device=device, dtype=torch.bool)
        all_feas_t = self.accelerator.gather(feas_t.contiguous())
        dist_t  = torch.tensor(distances_local, device=device, dtype=torch.float32)
        all_dist_t = self.accelerator.gather(dist_t.contiguous())
        all_row_hash = self.accelerator.gather(row_hash.contiguous())

        # 分组 z-score
        hash_cpu = all_row_hash.cpu().tolist()
        groups = defaultdict(list)
        for i, h in enumerate(hash_cpu):
            groups[h].append(i)

        all_a_feas_norm    = torch.zeros_like(all_a_feas_raw)
        all_a_outcome_norm = torch.zeros_like(all_a_feas_raw)
        feas_subset_size_per_group: list[int] = []

        for h, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)

            # A_feas: 整组 z-score (始终有效, 提供持续 contrastive 信号)
            grp = all_a_feas_raw[idx_t]
            all_a_feas_norm[idx_t] = (grp - grp.mean()) / (grp.std() + eps)

            # A_outcome: strict feasible 子集 >=2 才算; <2 时全 0 (前期纯靠 A_feas+PRM)
            feas_mask = all_feas_t[idx_t] & ~torch.isnan(all_dist_t[idx_t])
            feas_local_pos = feas_mask.nonzero(as_tuple=True)[0]
            feas_subset_size_per_group.append(int(feas_local_pos.numel()))
            if feas_local_pos.numel() >= 2:
                feas_idx_t = idx_t[feas_local_pos]
                neg_d = -all_dist_t[feas_idx_t]
                all_a_outcome_norm[feas_idx_t] = (
                    (neg_d - neg_d.mean()) / (neg_d.std() + eps)
                )

        # 切回本 rank
        rank = self.accelerator.process_index
        my_start = rank * B
        a_feas_norm_local = all_a_feas_norm[my_start:my_start + B]
        a_outcome_norm_local = all_a_outcome_norm[my_start:my_start + B]
        a_out = a_feas_norm_local + a_outcome_norm_local

        # distance 统计 (跨组)
        dist_std_per_group = []
        dist_cv_per_group = []
        for h, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
            valid_d = all_dist_t[idx_t][~torch.isnan(all_dist_t[idx_t])]
            if valid_d.numel() >= 2:
                std = float(valid_d.std().item())
                mean = float(valid_d.mean().item())
                dist_std_per_group.append(std)
                if mean > 1e-6:
                    dist_cv_per_group.append(std / mean)

        return {
            "a_out":                  a_out,
            "is_feasible":            is_feasible_local,
            "components":             components,
            "a_feas_norm":            a_feas_norm_local,
            "a_outcome_norm":         a_outcome_norm_local,
            "distance_local":         distances_local,
            "dist_std_per_group":     dist_std_per_group,
            "dist_cv_per_group":      dist_cv_per_group,
            "feas_subset_size_per_group": feas_subset_size_per_group,
        }

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

            feas = (c["parse"] == 1.0 and c["coverage"] >= config.cov_gate
                    and c["constraint"] == 1.0 and c["format"] == 1.0)
            is_feasible_local.append(feas)

            # cov_gate 硬墙: cov < gate 时 cons 信号置 0, 强迫模型先冲 cov
            cons_signal = c["constraint"] if c["coverage"] >= config.cov_gate else 0.0
            a_feas_raw[i] = (config.w_p * c["parse"]
                             + config.w_cov * c["coverage"]
                             + config.w_cons * cons_signal
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
    #  per-token log-prob: 分块 LM head (change A) + 回退
    # ══════════════════════════════════════════════════════════════════

    def _resolve_backbone_head(self, model):
        """定位 (backbone Qwen3Model, lm_head Linear)，结果缓存。

        结构: DeepSpeed/DDP 引擎 -unwrap-> PeftModelForCausalLM
              -get_base_model-> Qwen3ForCausalLM (.model=backbone, .lm_head=head)。
        定位纯属性遍历, 不触发任何 forward, 失败抛 AttributeError 交由上层回退。
        """
        if getattr(self, "_backbone_ref", None) is not None:
            return self._backbone_ref, self._lm_head_ref
        m = self.accelerator.unwrap_model(model)
        causal = m.get_base_model() if hasattr(m, "get_base_model") else m
        backbone = getattr(causal, "model", None)
        lm_head = getattr(causal, "lm_head", None)
        if backbone is None or lm_head is None or not callable(backbone):
            raise AttributeError(
                f"无法定位 backbone/lm_head (causal={type(causal).__name__})")
        self._backbone_ref, self._lm_head_ref = backbone, lm_head
        return backbone, lm_head

    def _gather_head_weight(self, lm_head) -> torch.Tensor:
        """取完整 [V, H] lm_head 权重。ZeRO-3 下权重被分片(有 ds_id)需先 gather;
        gather 出的完整 buffer 在 context 退出即释放, 故必须 clone 成独立持久张量,
        供 checkpoint backward 重算时复用(不再 re-gather)。lm_head frozen → detach 即可。"""
        w = lm_head.weight
        if hasattr(w, "ds_id"):                     # DeepSpeed ZeRO-3 partitioned param
            import deepspeed
            with deepspeed.zero.GatheredParameters([w], enabled=True):
                return w.detach().clone()           # 完整 [V, H] 稠密, 持久
        return w.detach()                           # 非分片(本地/ZeRO-2/0): 直接用

    def _completion_per_token_logps(self, model, input_ids, attention_mask,
                                    prompt_length, completion_ids):
        """优先 change A(分块 LM head)算 completion 段 per-token log-prob;
        任一步失败 → 打印一次警告并永久回退到原 OOM-safe 路径(model()+_selective_logp_chunked)。
        回退发生在 backward 之前, 不会污染已提交的梯度/优化器状态。"""
        completion_len = completion_ids.shape[1]
        if (os.environ.get("CHUNKED_LM_HEAD", "1") == "1"
                and not getattr(self, "_chunked_head_disabled", False)):
            try:
                backbone, lm_head = self._resolve_backbone_head(model)
                with _cuda_timer(self._timing_log, "fwd.backbone", self._prof_on):
                    with _sdpa_no_math():
                        bb_out = backbone(input_ids=input_ids,
                                          attention_mask=attention_mask, use_cache=False)
                    hidden = (bb_out.last_hidden_state
                              if hasattr(bb_out, "last_hidden_state") else bb_out[0])
                # 预测 completion[j] 的 logit 在位置 (P+j-1); 取 [P-1 : S-1] 共 completion_len 个,
                # 与原路径 logits_to_keep=C+1 再 [:, :-1, :] 的切片逐元素等价。
                hidden_comp = hidden[:, prompt_length - 1:-1, :]            # [B, C, H]
                with _cuda_timer(self._timing_log, "fwd.head", self._prof_on):
                    weight = self._gather_head_weight(lm_head)              # [V, H] frozen dense
                    return torch_checkpoint.checkpoint(
                        _logp_from_hidden_chunked, hidden_comp, completion_ids, weight,
                        use_reentrant=False,
                    )
            except Exception as _e:
                if not getattr(self, "_chunked_head_warned", False):
                    print(f"⚠️ 分块 LM head (CHUNKED_LM_HEAD) 失败, 永久回退到 "
                          f"model()+log_softmax 分块路径: {type(_e).__name__}: {_e}")
                    self._chunked_head_warned = True
                self._chunked_head_disabled = True
        # ── 回退: 原 OOM-safe 路径 (仍不物化 log_softmax, 但需完整 logits) ──
        with _sdpa_no_math():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            logits_to_keep=completion_len + 1)
        completion_logits = outputs.logits[:, :-1, :]
        return torch_checkpoint.checkpoint(
            _selective_logp_chunked, completion_logits, completion_ids,
            use_reentrant=False,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Loss 计算（单一 DAPO token-level）
    # ══════════════════════════════════════════════════════════════════

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        with _cuda_timer(self._timing_log, "fwd", self._prof_on):
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

        # per-token log-prob (见 _completion_per_token_logps):
        #   主路径 = change A 分块 LM head: backbone 出 hidden, 再分块过 head 算 logp,
        #     连完整 [B, T, V] logits 都不物化, 峰值仅 1 个 chunk [B, c, V]。
        #   回退路径 = model(logits_to_keep=C+1) 拿 completion 段 logits + _selective_logp_chunked
        #     (只少算 prompt 段 head, 仍会物化 [B, C, V]); unwrap/gather 失败时自动永久回退。
        #   两路对 completion 各 token 的 logp 逐元素等价 (位置切片 [prompt_len-1:-1])。
        per_token_logps = self._completion_per_token_logps(
            model, input_ids, attention_mask, prompt_length, completion_ids,
        )

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

        # vLLM IS 校正 (Truncated IS)
        # 修正 vLLM PagedAttention 跟训练 transformers attention kernel 在同 weights 下
        # 算 logprob 的数值差异. GRPO/PPO 的 importance sampling 假设要求 "采样分布 =
        # 训练时 old_logps 分布", 但 vLLM 实际采样用的是 vllm_logps (kernel 不同).
        # IS ratio = exp(old_logps - vllm_logps) 把 vLLM 实际采样分布换算回 old_logps
        # 分布, 让公式严格成立.
        #
        # vllm_per_token_logps 优先 (我们 patched VLLMClient 拿到的);
        # importance_sampling_ratio 后备 (TRL 0.23+ 自带, 此项目用 0.16 不会有).
        vllm_logps = inputs.get("vllm_per_token_logps")
        mask_hits = inputs.get("mask_hits")     # 0/1 tensor, mask 触发位置=1
        is_correction_log = None   # for self.log 用
        if vllm_logps is not None:
            # padding 位置 vllm_logps=0, old_logps 可能非零 → diff 大 → exp 爆炸.
            # 用 completion_mask 把 padding 位置的 diff 置 0 → ratio=1 (无影响).
            diff = (old_per_token_logps - vllm_logps) * completion_mask
            # Mask 位置跳过 IS 校正 (方案 C): mask 强约束让 vllm_logp 是 post-mask 的,
            # 跟 train_logp (raw) 量纲不一致, 算出来 ratio 会被 Z 因子污染. 设 ratio=1
            # 等价于 "信任 mask 强制选择是 on-policy", 误差 < 10%, 比直接用错误 ratio
            # (Z 因子误差 50-95%) 好得多. 见 CVRP-LLM-Mask-完整规则.md 第 9 节.
            if mask_hits is not None:
                diff = diff * (1.0 - mask_hits)
            is_correction = torch.exp(diff).clamp(max=3.0)   # Truncated IS, cap=3.0
            per_token_loss = per_token_loss * is_correction
            # 监控: 健康范围 mean ≈ 1.0, max < 3.0
            valid = completion_mask.bool()
            if valid.any():
                vals = is_correction[valid]
                is_correction_log = {
                    "is/ratio_mean": vals.mean(),
                    "is/ratio_std":  vals.std(),
                    "is/ratio_max":  vals.max(),
                    "is/ratio_cap_hit_rate": (vals >= 3.0).float().mean(),
                }
                if mask_hits is not None:
                    valid_mh = mask_hits[valid]
                    is_correction_log["is/mask_hit_rate"] = valid_mh.mean()
                    # 区分 mask 位置 vs 非 mask 位置的 IS (诊断用)
                    non_mask = valid & (mask_hits == 0)
                    if non_mask.any():
                        nm_vals = is_correction[non_mask]
                        is_correction_log["is/non_mask_ratio_mean"] = nm_vals.mean()
        else:
            is_ratio = inputs.get("importance_sampling_ratio")
            if is_ratio is not None:
                per_token_loss = per_token_loss * is_ratio
            elif not getattr(self, '_is_ratio_warning_emitted', False):
                print(
                    f"⚠️ WARNING: inputs 里既没找到 vllm_per_token_logps 也没找到 "
                    f"importance_sampling_ratio, IS 校正未生效。\n"
                    f"   当前 inputs keys: {sorted(inputs.keys())}"
                )
                self._is_ratio_warning_emitted = True

        # KL 正则
        beta_kl = getattr(self, 'beta', 0.0)
        kl_term_for_diag = None     # (B,T) raw kl, 供下面 diag 用 (即使 beta=0 也算便于观察 drift)
        if beta_kl != 0.0:
            ref_logps = inputs.get("ref_per_token_logps")
            if ref_logps is None:
                ref_logps = inputs.get("ref_logprobs")
            if ref_logps is not None:
                kl = torch.exp(ref_logps - per_token_logps) \
                     - (ref_logps - per_token_logps) - 1
                per_token_loss = per_token_loss + beta_kl * kl
                kl_term_for_diag = kl
            elif not getattr(self, '_kl_warning_emitted', False):
                print(
                    f"⚠️ WARNING: kl_coef={beta_kl} > 0 但 inputs 里找不到 "
                    f"ref_per_token_logps / ref_logprobs，KL anchor 实际未生效！"
                )
                self._kl_warning_emitted = True

        # DAPO token-level 聚合：总 token 数为分母，长 completion 权重更大
        loss = (per_token_loss * completion_mask).sum() \
               / completion_mask.sum().clamp(min=1.0)

        # 真正的 truncation rate: trajectory completion 长度 >= max_completion_length - 1
        # (减 1 因为 vLLM 不一定写满最后 1 个 token, 接近上限就算截断)
        # 旧 metric "stats/truncation_rate" 实际是 empty_traj_rate, 重命名修正.
        comp_lens = completion_mask.sum(-1)
        empty_traj_rate = (comp_lens == 0).float().mean()
        max_len = self.args.max_completion_length
        truncation_rate = (comp_lens >= max_len - 1).float().mean()
        loss_log = {
            "loss/total":                self._gather_mean(loss),
            "stats/empty_traj_rate":     self._gather_mean(empty_traj_rate),
            "stats/truncation_rate":     self._gather_mean(truncation_rate),
        }
        if is_correction_log is not None:
            for k, v in is_correction_log.items():
                loss_log[k] = self._gather_mean(v)

        # ── 诊断字段 (永久基础设施, 任何调参都能看到 KL / drift / clip / adv 量级) ──
        with torch.no_grad():
            valid_mask = completion_mask.bool()
            n_valid = completion_mask.sum().clamp(min=1.0)

            # 1. KL 诊断 (核心: KL 占 loss 比, > 30% 即 dominate)
            if kl_term_for_diag is not None:
                kl_mean = (kl_term_for_diag * completion_mask).sum() / n_valid
                if valid_mask.any():
                    kl_max = kl_term_for_diag[valid_mask].max()
                else:
                    kl_max = torch.tensor(0.0, device=loss.device)
                kl_loss_contrib = (beta_kl * kl_term_for_diag * completion_mask).sum() / n_valid
                kl_share = (kl_loss_contrib.abs() / (loss.abs() + 1e-8))
                loss_log["diag/kl_mean"]       = self._gather_mean(kl_mean)
                loss_log["diag/kl_max"]        = self._gather_mean(kl_max)
                loss_log["diag/kl_loss_share"] = self._gather_mean(kl_share)

            # 2. Policy drift (模型实际偏离 old policy 多远, 11 step 后 ≈ 0 = policy 没动)
            if valid_mask.any():
                ratio_valid = ratio[valid_mask]
                loss_log["diag/ratio_drift_mean"] = self._gather_mean((ratio_valid - 1.0).abs().mean())

            # 3. Advantage 量级 (信号强度 + 非零率)
            adv_flat = adv.expand_as(per_token_loss) if adv.dim() == 2 else adv
            adv_abs = adv_flat.abs() if torch.is_tensor(adv_flat) else torch.tensor(0.0, device=loss.device)
            loss_log["diag/adv_abs_mean"]     = self._gather_mean(adv_abs.mean())
            loss_log["diag/adv_nonzero_rate"] = self._gather_mean((adv_abs > 0.01).float().mean())

            # 4. Clip 触发率 (Clip-Higher 是否真的让 ε_high 区间被用)
            clip_low_hit  = ((ratio < (1 - eps_low))  & valid_mask).float().sum() / n_valid
            clip_high_hit = ((ratio > (1 + eps_high)) & valid_mask).float().sum() / n_valid
            loss_log["diag/clip_low_hit_rate"]  = self._gather_mean(clip_low_hit)
            loss_log["diag/clip_high_hit_rate"] = self._gather_mean(clip_high_hit)

        self.log(loss_log)

        return loss

    # ══════════════════════════════════════════════════════════════════
    #  精细化阶段耗时 profiler (诊断瓶颈 / A-B 各加速旋钮)
    # ══════════════════════════════════════════════════════════════════

    @property
    def _prof_on(self) -> bool:
        """当前 optimizer step 是否在测量窗口内 (跳过 warmup, 只测 PROFILE_STEPS 个)。"""
        return (not self._timing_done
                and self._prof_warmup <= self._prof_opt_seen
                < self._prof_warmup + self._prof_total)

    def _advance_step(self):
        """一个 optimizer step 完成后推进边界: 计数 + 到达 PROFILE_STEPS 则打印。"""
        was_measured = self._prof_on            # 必须在 opt_seen++ 之前取
        self._prof_opt_seen += 1
        self._timing_fwd_bwd_count = 0          # 重置 micro 计数, 进入下一 step
        if was_measured:
            self._prof_measured += 1
        if not self._timing_done and self._prof_measured >= self._prof_total:
            self._timing_done = True
            self._print_step_timing()

    def _install_optim_timer(self):
        """懒包 self.optimizer.step: 计 optim 耗时 + 推进 step 边界。
        optim 每个 optimizer step 调一次 (grad_accum 个 micro 之后), 是 CPUAdam
        vs FusedAdam 的关键诊断点 (DS_OFFLOAD 的主要成本)。"""
        self._optim_wrapped = True
        try:
            _orig = self.optimizer.step
        except Exception:
            self._optim_timer_ok = False        # 包不上 → training_step 兜底推进 (optim 计 0)
            return
        self._optim_timer_ok = True
        _self = self

        def _timed_step(*a, **k):
            _rec = _self._prof_on
            if _rec and torch.cuda.is_available():
                torch.cuda.synchronize()
            _t = time.time()
            out = _orig(*a, **k)
            if _rec and torch.cuda.is_available():
                torch.cuda.synchronize()
            if _rec:
                _self._timing_log["optim"] = (
                    _self._timing_log.get("optim", 0.0) + time.time() - _t)
            _self._advance_step()               # optim 已计入, 此时推进边界才对
            return out

        self.optimizer.step = _timed_step

    def training_step(self, *args, **kwargs):
        """单次 micro-batch 的 forward + backward。grad_accum 次累加后才触发 optimizer.step。"""
        if not self._optim_wrapped:
            self._install_optim_timer()
        _record = self._prof_on
        if _record and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.time()

        loss = super().training_step(*args, **kwargs)

        if _record and torch.cuda.is_available():
            torch.cuda.synchronize()
        if _record:
            self._timing_log["fwd_bwd"] = (
                self._timing_log.get("fwd_bwd", 0.0) + time.time() - _t0)
        self._timing_fwd_bwd_count += 1
        # 兜底: 若 optimizer.step 没能被包 (DeepSpeed 内部 step), 用 micro 计数推进边界 (optim 计 0)
        ga = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
        if not getattr(self, "_optim_timer_ok", False) and self._timing_fwd_bwd_count >= ga:
            self._advance_step()
        return loss

    def _print_step_timing(self):
        if not self.accelerator.is_main_process:
            return
        n = max(1, self._prof_measured)          # 实测 step 数, 用于求每-step 平均
        r = {k: v / n for k, v in self._timing_log.items()}   # 每 step 平均
        gen   = r.get("rollout", 0.0)            # vLLM 生成 + 权重同步
        score = r.get("reward", 0.0)             # PRM + terminal + advantage + resample
        fwd   = r.get("fwd", 0.0)                # _compute_loss 前向 (累 grad_accum micro)
        fbb   = r.get("fwd_bwd", 0.0)            # 前向+反向 (累 grad_accum micro)
        bwd   = max(0.0, fbb - fwd)              # 反向 = fwd_bwd - fwd
        bb    = r.get("fwd.backbone", 0.0)
        head  = r.get("fwd.head", 0.0)
        optim = r.get("optim", 0.0)
        total = gen + score + fbb + optim
        if total <= 0:
            return
        pc = lambda x: 100.0 * x / total
        ga = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
        print(
            f"\n{'='*72}\n"
            f"  精细化阶段耗时 (rank0, 每 step 平均; 测 {n} step, 跳过前 {self._prof_warmup})\n"
            f"  grad_accum={ga}  →  fwd/bwd 桶是 {ga} 个 micro 之和\n"
            f"{'='*72}\n"
            f"  gen   vLLM生成+权重同步     {gen:8.2f}s ({pc(gen):5.1f}%)  ← vLLM/采样长度\n"
            f"  score PRM+terminal+adv      {score:8.2f}s ({pc(score):5.1f}%)  ← POMO PRM/CPU\n"
            f"  fwd   前向(backbone+head)   {fwd:8.2f}s ({pc(fwd):5.1f}%)  ← Liger/change A\n"
            f"      ├ backbone(36层)        {bb:8.2f}s            ← Liger/DS_OFFLOAD\n"
            f"      └ head(分块logp)        {head:8.2f}s            ← change A(CHUNKED_LM_HEAD)\n"
            f"  bwd   反向=fwd_bwd-fwd      {bwd:8.2f}s ({pc(bwd):5.1f}%)  ← 梯度重计算(GC)/NCCL/gather\n"
            f"  optim 优化器step            {optim:8.2f}s ({pc(optim):5.1f}%)  ← DS_OFFLOAD(CPUAdam vs FusedAdam)\n"
            f"  {'-'*66}\n"
            f"  单 step 总计                {total:8.2f}s\n"
            f"  (注: bwd 含被 GC/ZeRO-3 隐藏的重算与通信; optim≈0 说明 DeepSpeed 内部 step 未走 self.optimizer.step)\n"
            f"{'='*72}\n",
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
