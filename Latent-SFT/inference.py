"""
Latent-SFT 推理脚本：entropy-based 动态切换 latent/explicit 模式。

推理流程:
  1. 模型读取问题 prompt，默认以显式模式开始
  2. 显式模式: 正常自回归，监控熵
  3. 熵连续 K 步上升 → 进入 latent 模式
  4. latent 模式: 每步输入训练好的 latent embedding，通过 KV cache 中的
     注意力积累推理信息，同时过 LM head 算熵（仅监控，不采样）
  5. 熵连续 K 步下降 → 退出 latent，用当前 logits 采样 token
  6. latent 步数达到上限 → 强制退出转显式
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import LatentEmbeddings


class LatentInferenceEngine:

    def __init__(self, model_path: str, latent_emb_path: str = None,
                 entropy_window: int = 3, max_latent_steps: int = 48):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.latent_emb = None
        if latent_emb_path:
            hidden_size = self.model.config.hidden_size
            self.latent_emb = LatentEmbeddings(hidden_size)
            state = torch.load(latent_emb_path, map_location="cpu")
            self.latent_emb.load_state_dict(state)
            self.latent_emb = self.latent_emb.to(self.model.device)

        self.entropy_window = entropy_window
        self.max_latent_steps = max_latent_steps

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        log_p = F.log_softmax(logits.float(), dim=-1)
        entropy = -(log_p.exp() * log_p).sum(dim=-1)
        return entropy.item()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, start_latent: bool = False):
        """
        生成解，支持 latent/显式动态切换。

        Args:
            prompt: 问题 prompt（已格式化）
            max_new_tokens: 最大新生成 token 数
            temperature: 采样温度 (0 = greedy)
            start_latent: 是否从 latent 模式开始
        """
        if start_latent and self.latent_emb is None:
            raise ValueError("Cannot start in latent mode without latent embeddings")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        prefill_out = self.model(
            input_ids=input_ids, use_cache=True,
        )
        past_key_values = prefill_out.past_key_values

        generated_tokens = []
        entropy_history = []

        in_latent = start_latent
        latent_step_count = 0

        if not start_latent:
            first_logits = prefill_out.logits[:, -1, :]
            first_id = self._sample(first_logits, temperature)
            generated_tokens.append(first_id)
            current_input = torch.tensor([[first_id]], device=self.model.device)
            if first_id == self.tokenizer.eos_token_id:
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True), {
                    "num_tokens": 1, "latent_steps": 0, "entropy_history": [],
                }
        else:
            current_input = None

        remaining = max_new_tokens - len(generated_tokens)
        total_latent_steps = 0
        full_entropy_history = []

        for step in range(remaining):
            if in_latent:
                latent_input = self.latent_emb().unsqueeze(0).unsqueeze(0)
                outputs = self.model(
                    inputs_embeds=latent_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            entropy = self._compute_entropy(logits)
            entropy_history.append(entropy)
            full_entropy_history.append(entropy)

            if in_latent:
                latent_step_count += 1
                total_latent_steps += 1

                # 退出条件 1: 熵连续 K 步下降
                should_exit = False
                if len(entropy_history) >= self.entropy_window + 1:
                    recent = entropy_history[-(self.entropy_window + 1):]
                    should_exit = all(recent[i] > recent[i + 1] for i in range(self.entropy_window))

                # 退出条件 2: 达到上限
                if latent_step_count >= self.max_latent_steps:
                    should_exit = True

                if should_exit:
                    in_latent = False
                    latent_step_count = 0
                    entropy_history = []  # 重置，避免跨模式误触发
                    # 退出时用当前 logits 采样第一个显式 token
                    token_id = self._sample(logits, temperature)
                    generated_tokens.append(token_id)
                    current_input = torch.tensor([[token_id]], device=self.model.device)

                    if token_id == self.tokenizer.eos_token_id:
                        break
                # 否则继续 latent（不采样，不输出 token）

            else:
                # 显式模式: 正常自回归
                token_id = self._sample(logits, temperature)
                generated_tokens.append(token_id)
                current_input = torch.tensor([[token_id]], device=self.model.device)

                if token_id == self.tokenizer.eos_token_id:
                    break

                if self.latent_emb is not None and len(entropy_history) >= self.entropy_window + 1:
                    recent = entropy_history[-(self.entropy_window + 1):]
                    if all(recent[i] < recent[i + 1] for i in range(self.entropy_window)):
                        in_latent = True
                        latent_step_count = 0
                        entropy_history = []  # 重置，避免跨模式误触发

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text, {
            "num_tokens": len(generated_tokens),
            "latent_steps": total_latent_steps,
            "entropy_history": full_entropy_history,
        }

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return logits.argmax(dim=-1).item()
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


# ====================================================================
# HLR Inference Engine — Hierarchical Latent Reasoner 推理
# ====================================================================
#
# 与 LatentInferenceEngine 的差异:
#   - latent 模式不输入共享 V, 而是用 LatentReasoner incremental forward
#   - LR 有 KV cache, 每 latent step 只算新增 1 个位置 (O(1) per step, O(k) total)
#   - latent 段每步: LR.forward(h_in, k=1, past_kv) → 取顶层 hidden → up_proj
#                    → 喂主模型 inputs_embeds (A'' 注入, 与训练一致)
#                    → 主模型 forward 一步, KV cache 续接
#                    → 算 entropy 监控
#   - 入口 h_in 在进入 latent 时锁定 (即显式段末位 hidden), latent 期间不变
#   - LR KV cache 跨 latent step 累积, 退出 latent 时清空


from config import HLRConfig
from model import build_latent_reasoner_from_main


class HLRInferenceEngine:

    def __init__(
        self,
        model_path: str,
        latent_reasoner_path: str,
        cfg: HLRConfig | None = None,
        entropy_window: int | None = None,
        max_latent_steps: int | None = None,
        min_latent_steps: int | None = None,
        latent_cooldown: int | None = None,
        entropy_quantile: float | None = None,
        min_entropy_samples: int | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 与 train_hlr 对齐: 强制关闭 add_bos_token, 避免训练/推理 tokenize 长度差 1
        if getattr(self.tokenizer, "add_bos_token", False):
            self.tokenizer.add_bos_token = False

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # 根据 cfg (或默认) 构造 LatentReasoner 后加载权重
        if cfg is None:
            cfg = HLRConfig()
        self.latent_reasoner = build_latent_reasoner_from_main(self.model, cfg)
        state = torch.load(latent_reasoner_path, map_location="cpu")
        self.latent_reasoner.load_state_dict(state)
        self.latent_reasoner = self.latent_reasoner.to(self.model.device).to(torch.bfloat16)
        self.latent_reasoner.eval()

        # Latent trigger 参数 (推理时使用, 必须与 entropy_profile.py 训练侧镜像一致)
        self.entropy_window = entropy_window if entropy_window is not None else cfg.entropy_window
        self.entropy_quantile = entropy_quantile if entropy_quantile is not None else cfg.entropy_quantile
        self.min_latent_steps = min_latent_steps if min_latent_steps is not None else cfg.min_latent_steps
        self.max_latent_steps = max_latent_steps if max_latent_steps is not None else cfg.max_latent_steps
        self.latent_cooldown = latent_cooldown if latent_cooldown is not None else cfg.latent_cooldown
        self.min_entropy_samples = min_entropy_samples if min_entropy_samples is not None else cfg.min_entropy_samples

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        log_p = F.log_softmax(logits.float(), dim=-1)
        entropy = -(log_p.exp() * log_p).sum(dim=-1)
        return entropy.item()

    def _sample(self, logits: torch.Tensor, temperature: float,
                top_p: float = 1.0, top_k: int = -1) -> int:
        """支持 temperature / top_p / top_k 联动 (与 paths.sh GEN_TEMPERATURE/TOP_P/TOP_K 对齐).

        temperature<=0 → greedy.
        top_k>0 → 截断到 top-k tokens.
        top_p<1.0 → nucleus sampling (累积 prob >= top_p 后停).
        """
        if temperature <= 0:
            return logits.argmax(dim=-1).item()

        scaled = logits / temperature

        # top-k 截断
        if top_k is not None and top_k > 0:
            k = min(top_k, scaled.size(-1))
            topk_vals, _ = torch.topk(scaled, k, dim=-1)
            min_keep = topk_vals[..., -1, None]
            scaled = torch.where(scaled < min_keep, torch.full_like(scaled, float("-inf")), scaled)

        probs = F.softmax(scaled, dim=-1)

        # top-p (nucleus) 截断
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumprob = torch.cumsum(sorted_probs, dim=-1)
            # 保留累积 prob <= top_p 的部分, 第一个超过 top_p 的也保留 (避免空集)
            sorted_keep = cumprob - sorted_probs <= top_p
            sorted_probs = sorted_probs * sorted_keep.to(sorted_probs.dtype)
            # 重新归一化
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            # 散回原顺序
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

        return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = -1):
        """
        生成解, 显式/latent 动态切换 (新方向: 压缩低熵段省算力).

        触发规则 (与 entropy_profile.py 镜像一致):
          进入 latent:
            1. 熵连续 entropy_window 步下降
            2. 当前熵 < seen_entropies 的 entropy_quantile 分位数
            3. 自上次退出 latent 起, 已走 >= latent_cooldown 个显式 token
            4. seen_entropies 累积 >= min_entropy_samples (running median 才稳)
          退出 latent:
            1. 已走 >= min_latent_steps 步 之后
            2. 熵连续 entropy_window 步上升
            3. OR latent_step_count >= max_latent_steps (强制)

        Returns:
            output_text, info_dict
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        # ── Prefill ──
        prefill_out = self.model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = prefill_out.past_key_values
        last_main_hidden = prefill_out.hidden_states[-1][:, -1, :]

        first_logits = prefill_out.logits[:, -1, :]
        first_id = self._sample(first_logits, temperature, top_p=top_p, top_k=top_k)
        generated_tokens = [first_id]
        current_input = torch.tensor([[first_id]], device=self.model.device)

        if first_id == self.tokenizer.eos_token_id:
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True), {
                "num_tokens": 1, "latent_steps": 0, "entropy_history": [],
            }

        # 状态变量
        in_latent = False
        latent_step_count = 0
        latent_past_kv = None
        latent_h_in = None

        seen_entropies: list[float] = []          # 整 generate 累积 (用于 running quantile)
        window_entropies: list[float] = []        # 用于趋势检测, 模式切换时重置
        explicit_count_since_exit = float("inf")  # 初始无穷大: 允许首次进入 latent

        full_entropy_history: list[float] = []
        total_latent_steps = 0

        def _running_quantile_threshold() -> float:
            sorted_e = sorted(seen_entropies)
            idx = min(int(len(sorted_e) * self.entropy_quantile), len(sorted_e) - 1)
            return sorted_e[idx]

        for _step in range(max_new_tokens - 1):
            if in_latent:
                # ── Latent 模式: LR incremental → 主模型 forward 1 步 ──
                layer_hiddens, latent_past_kv = self.latent_reasoner(
                    latent_h_in, k=1, past_kv=latent_past_kv
                )
                top_new = layer_hiddens[-1][:, -1:, :]
                latent_inputs_embed = self.latent_reasoner.up_proj(top_new)

                main_out = self.model(
                    inputs_embeds=latent_inputs_embed,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_kv = main_out.past_key_values
                last_main_hidden = main_out.hidden_states[-1][:, -1, :]

                logits = main_out.logits[:, -1, :]
                entropy = self._compute_entropy(logits)
                seen_entropies.append(entropy)
                window_entropies.append(entropy)
                full_entropy_history.append(entropy)

                latent_step_count += 1
                total_latent_steps += 1

                # 退出判断: 必须走过 min_latent_steps 之后才检测趋势退出
                should_exit = False
                if latent_step_count >= self.min_latent_steps:
                    if len(window_entropies) >= self.entropy_window + 1:
                        recent = window_entropies[-(self.entropy_window + 1):]
                        rising = all(
                            recent[i] < recent[i + 1] for i in range(self.entropy_window)
                        )
                        should_exit = rising
                if latent_step_count >= self.max_latent_steps:
                    should_exit = True  # 强制兜底

                if should_exit:
                    # 切回显式: 当前 logits 采样一个 token
                    in_latent = False
                    latent_step_count = 0
                    latent_past_kv = None
                    latent_h_in = None
                    window_entropies = []
                    explicit_count_since_exit = 0  # 重置 cooldown 计数

                    token_id = self._sample(logits, temperature, top_p=top_p, top_k=top_k)
                    generated_tokens.append(token_id)
                    current_input = torch.tensor([[token_id]], device=self.model.device)
                    if token_id == self.tokenizer.eos_token_id:
                        break

            else:
                # ── 显式模式: 主模型自回归 1 步 ──
                main_out = self.model(
                    input_ids=current_input,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_kv = main_out.past_key_values
                last_main_hidden = main_out.hidden_states[-1][:, -1, :]

                logits = main_out.logits[:, -1, :]
                entropy = self._compute_entropy(logits)
                seen_entropies.append(entropy)
                window_entropies.append(entropy)
                full_entropy_history.append(entropy)
                explicit_count_since_exit += 1

                token_id = self._sample(logits, temperature, top_p=top_p, top_k=top_k)
                generated_tokens.append(token_id)
                current_input = torch.tensor([[token_id]], device=self.model.device)
                if token_id == self.tokenizer.eos_token_id:
                    break

                # 进入 latent 判断 (新方向: 连续下降 + 低于分位数 + cooldown 过 + 样本足够)
                can_check = (
                    explicit_count_since_exit >= self.latent_cooldown
                    and len(seen_entropies) >= self.min_entropy_samples
                    and len(window_entropies) >= self.entropy_window + 1
                )
                if can_check:
                    recent = window_entropies[-(self.entropy_window + 1):]
                    falling = all(
                        recent[i] > recent[i + 1] for i in range(self.entropy_window)
                    )
                    threshold = _running_quantile_threshold()
                    below_threshold = entropy < threshold

                    if falling and below_threshold:
                        in_latent = True
                        latent_step_count = 0
                        latent_past_kv = None
                        latent_h_in = last_main_hidden
                        window_entropies = []

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text, {
            "num_tokens": len(generated_tokens),
            "latent_steps": total_latent_steps,
            "entropy_history": full_entropy_history,
        }
