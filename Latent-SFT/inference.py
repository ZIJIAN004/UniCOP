"""
Latent-SFT 推理脚本：COCONUT 式 hidden state 回传 + 熵趋势动态切换。

推理流程:
  1. 模型读取问题 prompt
  2. 进入 latent 模式（默认从 latent 开始）
  3. latent 模式: hidden state 直接回传为下一步输入，
     同时过 LM head 算熵（仅监控，不采样）
  4. 熵连续 K 步下降 → 退出 latent，用当前 LM head 输出采样 token
  5. 显式模式: 正常自回归
  6. 熵连续 K 步上升 → 重新进入 latent
  7. latent 步数达到上限 → 强制退出
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
            state = torch.load(latent_emb_path, map_location="cpu")
            num_tokens = state["embeddings"].shape[0]
            self.latent_emb = LatentEmbeddings(num_tokens, hidden_size)
            self.latent_emb.load_state_dict(state)
            self.latent_emb = self.latent_emb.to(self.model.device)

        self.entropy_window = entropy_window
        self.max_latent_steps = max_latent_steps

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
        return entropy.item()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, start_latent: bool = True):
        """
        生成解，支持 latent/显式动态切换。

        Args:
            prompt: 问题 prompt（已格式化）
            max_new_tokens: 最大新生成 token 数
            temperature: 采样温度 (0 = greedy)
            start_latent: 是否从 latent 模式开始
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        # 预填充 KV cache（处理 prompt），避免第一步 latent 浪费在 prompt 编码上
        prefill_out = self.model(
            input_ids=input_ids, use_cache=True, output_hidden_states=True,
        )
        past_key_values = prefill_out.past_key_values
        last_hidden = prefill_out.hidden_states[-1][:, -1, :]

        generated_tokens = []
        entropy_history = []

        in_latent = start_latent
        latent_step_count = 0

        # 非 latent 启动时，用 prefill logits 采样第一个 token
        if not start_latent:
            first_logits = prefill_out.logits[:, -1, :]
            first_id = self._sample(first_logits, temperature)
            generated_tokens.append(first_id)
            current_input = torch.tensor([[first_id]], device=self.model.device)
            if first_id == self.tokenizer.eos_token_id:
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True), {
                    "num_tokens": 1, "entropy_history": [],
                }
        else:
            current_input = None

        for step in range(max_new_tokens):
            if in_latent:
                # COCONUT 式: 用上一步的 hidden state 作为输入，shape (1, 1, hidden)
                outputs = self.model(
                    inputs_embeds=last_hidden.unsqueeze(1),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
            else:
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # 最后一层最后位置

            entropy = self._compute_entropy(logits)
            entropy_history.append(entropy)

            if in_latent:
                latent_step_count += 1

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

                # 进入条件: 熵连续 K 步上升
                if len(entropy_history) >= self.entropy_window + 1:
                    recent = entropy_history[-(self.entropy_window + 1):]
                    if all(recent[i] < recent[i + 1] for i in range(self.entropy_window)):
                        in_latent = True
                        latent_step_count = 0
                        entropy_history = []  # 重置，避免跨模式误触发

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text, {
            "num_tokens": len(generated_tokens),
            "entropy_history": entropy_history,
        }

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return logits.argmax(dim=-1).item()
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
