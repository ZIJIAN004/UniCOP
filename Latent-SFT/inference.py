"""
HLR 推理引擎 (Hierarchical Latent Reasoner) — online entropy 触发的混合解码.

触发规则镜像 entropy_profile.detect_latent_segments (训练侧):
  进入 latent (低熵段):
    - entropy_history 长度 >= min_entropy_samples
    - 距上次退 latent 至少 latent_cooldown 个显式 token
    - 连续 entropy_window 步熵下降
    - 当前熵 < entropy_history 的 entropy_quantile 分位数
  退出 latent:
    - latent step 数 >= max_latent_steps      OR
    - latent step 数 >= min_latent_steps 且连续 window 步熵上升

每次显式步:
  - 从 last_logits 采样 token, 累加 explicit_count
  - 主模型 forward 一步, 更新 last_logits / last_hidden / past_kv

每次 latent 步:
  - LR.forward(enter_hidden, k=1, past_kv=lr_past) → top hidden
  - up_proj → inputs_embeds 注入主模型 forward 一步, 拿 logits (不采样, 仅监控熵)
  - 更新 lr_past 和 main_past; last_hidden 不变 (h_in 复用 enter_hidden, 与训练一致)

bs=1.

返回 (text, info), info 字段:
  explicit_tokens:           显式 token 总数
  latent_steps:              latent 总步数
  latent_steps_as_tokens:    latent_steps × compression_ratio (等效替代的显式 token 数)
  total_equivalent_tokens:   explicit_tokens + latent_steps_as_tokens (公平对比 baseline 用)
  entropy_history:           显式段累积熵
  latent_segments:           [{enter_explicit_idx, steps, exit_entropy}]
  wall_time_sec:             generate() 耗时
  truncated:                 是否因 max_new_tokens 截断
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import HLRConfig
from model import build_latent_reasoner_from_main


class HLRInferenceEngine:
    def __init__(
        self,
        checkpoint_dir: str,
        base_model_path: str | None = None,
        cfg: HLRConfig | None = None,
        device: str = "cuda",
        merge_lora: bool = False,
    ):
        self.cfg = cfg if cfg is not None else HLRConfig()
        self.device = device
        ckpt = Path(checkpoint_dir)

        # ── tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 主模型: 自动检测 adapter or merged ──
        adapter_cfg = ckpt / "adapter_config.json"
        if adapter_cfg.exists():
            if base_model_path is None:
                with open(adapter_cfg, "r", encoding="utf-8") as f:
                    base_model_path = json.load(f)["base_model_name_or_path"]
            print(f"  [HLR engine] 加载 base model: {base_model_path}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
            from peft import PeftModel
            print(f"  [HLR engine] 加载 LoRA adapter: {checkpoint_dir}")
            model = PeftModel.from_pretrained(base, str(checkpoint_dir))
            if merge_lora:
                print("  [HLR engine] merge LoRA (推理加速)")
                model = model.merge_and_unload()
        else:
            print(f"  [HLR engine] 加载 merged 主模型: {checkpoint_dir}")
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)

        self.model = model
        self.model.eval()

        # ── LatentReasoner ──
        lr_path = ckpt / "latent_reasoner.pt"
        if not lr_path.exists():
            raise FileNotFoundError(
                f"latent_reasoner.pt 不在 {checkpoint_dir} (HLR 推理必须有)"
            )
        self.latent_reasoner = build_latent_reasoner_from_main(self.model, self.cfg)
        state = torch.load(lr_path, map_location="cpu")
        self.latent_reasoner.load_state_dict(state)
        self.latent_reasoner = self.latent_reasoner.to(device).to(torch.bfloat16)
        self.latent_reasoner.eval()
        print(f"  [HLR engine] LatentReasoner 加载完成")

        # ── 触发参数 ──
        self.window = self.cfg.entropy_window
        self.quantile = self.cfg.entropy_quantile
        self.min_latent_steps = self.cfg.min_latent_steps
        self.max_latent_steps = self.cfg.max_latent_steps
        self.cooldown = self.cfg.latent_cooldown
        self.min_samples = self.cfg.min_entropy_samples
        self.compression_ratio = self.cfg.latent_compression_ratio

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _entropy(logits: torch.Tensor) -> float:
        log_p = F.log_softmax(logits.float(), dim=-1)
        return -(log_p.exp() * log_p).sum(dim=-1).item()

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return logits.argmax(dim=-1).item()
        probs = F.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs.squeeze(0), num_samples=1).item()

    def _should_enter_latent(
        self, entropy_history: list[float], tokens_since_exit: int
    ) -> bool:
        if len(entropy_history) < max(self.window + 1, self.min_samples):
            return False
        if tokens_since_exit < self.cooldown:
            return False
        recent = entropy_history[-(self.window + 1):]
        falling = all(recent[i] > recent[i + 1] for i in range(self.window))
        if not falling:
            return False
        sorted_e = sorted(entropy_history)
        q_idx = min(int(len(sorted_e) * self.quantile), len(sorted_e) - 1)
        return entropy_history[-1] < sorted_e[q_idx]

    def _should_exit_latent(
        self, latent_entropies: list[float], steps_done: int
    ) -> bool:
        if steps_done >= self.max_latent_steps:
            return True
        if steps_done < self.min_latent_steps:
            return False
        if len(latent_entropies) < self.window + 1:
            return False
        recent = latent_entropies[-(self.window + 1):]
        return all(recent[i] < recent[i + 1] for i in range(self.window))

    # ── generate ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, dict]:
        """
        Args:
            prompt: str (raw text) 或 list[dict] (chat 格式, 走 apply_chat_template)
            max_new_tokens: 总 equivalent token 数上限 (explicit + latent×compression)
            temperature: 0 → greedy, >0 → 采样
        """
        start = time.perf_counter()

        if isinstance(prompt, list):
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        # ── prefill ──
        out = self.model(
            input_ids=input_ids, use_cache=True, output_hidden_states=True,
        )
        main_past = out.past_key_values
        last_logits = out.logits[:, -1, :]
        last_hidden = out.hidden_states[-1][:, -1, :]  # [1, H_main]

        generated_tokens: list[int] = []
        entropy_history: list[float] = []
        latent_segments: list[dict] = []
        explicit_count = 0
        latent_step_count = 0
        tokens_since_exit = self.cooldown   # 允许首次触发 (但 history 不足会先 block)
        truncated = False
        hit_eos = False
        eos_id = self.tokenizer.eos_token_id

        while True:
            equivalent = explicit_count + latent_step_count * self.compression_ratio
            if equivalent >= max_new_tokens:
                truncated = True
                break

            # ── 是否进 latent ──
            if self._should_enter_latent(entropy_history, tokens_since_exit):
                enter_at = explicit_count
                enter_hidden = last_hidden  # 锁定: 整段 LR 都用同一 h_in (与训练一致)
                lr_past = None
                latent_ents: list[float] = []
                steps_done = 0

                while True:
                    # LR 一步
                    layer_hiddens, lr_past = self.latent_reasoner(
                        enter_hidden, k=1, past_kv=lr_past,
                    )
                    top_h = layer_hiddens[-1]                              # [1, 1, lr_hidden]
                    inj = self.latent_reasoner.up_proj(top_h)              # [1, 1, H_main]

                    # 主模型一步 (inputs_embeds 模式, 不采样 token)
                    out = self.model(
                        inputs_embeds=inj.to(next(self.model.parameters()).dtype),
                        past_key_values=main_past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    main_past = out.past_key_values
                    cur_logits = out.logits[:, -1, :]
                    # last_hidden 在 latent 段内不更新 LR 入口 (enter_hidden 锁定),
                    # 但段末位的 main hidden 是退出后第一个显式步要不要用? 不需要,
                    # 因为退出后是从 cur_logits 采样, last_hidden 由那个 explicit token
                    # 自己重新算.

                    latent_ents.append(self._entropy(cur_logits))
                    steps_done += 1
                    latent_step_count += 1

                    # 退出判断
                    if self._should_exit_latent(latent_ents, steps_done):
                        break
                    # equivalent token 上限保护
                    eq2 = explicit_count + latent_step_count * self.compression_ratio
                    if eq2 >= max_new_tokens:
                        break

                # 退出 latent: 用段末位 logits 作为下一步显式采样的 last_logits
                last_logits = cur_logits
                # last_hidden 用段末位主模型 hidden (供下一个 latent 段做 h_in, 也供
                # 后续显式步 forward 不依赖, 因为显式步会自己算新 hidden)
                last_hidden = out.hidden_states[-1][:, -1, :]

                latent_segments.append({
                    "enter_explicit_idx": enter_at,
                    "steps": steps_done,
                    "exit_entropy": latent_ents[-1] if latent_ents else None,
                })
                tokens_since_exit = 0
                # 不立即采样, 回 while 头, 跳过 should_enter (cooldown 重置), 直接显式
                continue

            # ── 显式模式: 采样下一个 token ──
            next_id = self._sample(last_logits, temperature)
            generated_tokens.append(next_id)
            explicit_count += 1
            tokens_since_exit += 1
            entropy_history.append(self._entropy(last_logits))

            if next_id == eos_id:
                hit_eos = True
                break

            # forward 下一步
            next_tensor = torch.tensor([[next_id]], device=self.device)
            out = self.model(
                input_ids=next_tensor,
                past_key_values=main_past,
                use_cache=True,
                output_hidden_states=True,
            )
            main_past = out.past_key_values
            last_logits = out.logits[:, -1, :]
            last_hidden = out.hidden_states[-1][:, -1, :]

        text_out = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

        return text_out, {
            "explicit_tokens": explicit_count,
            "latent_steps": latent_step_count,
            "latent_steps_as_tokens": latent_step_count * self.compression_ratio,
            "total_equivalent_tokens": explicit_count + latent_step_count * self.compression_ratio,
            "latent_segments": latent_segments,
            "entropy_history": entropy_history,
            "wall_time_sec": time.perf_counter() - start,
            "truncated": truncated,
            "hit_eos": hit_eos,
        }


def main():
    """Smoke / manual test: 加载 checkpoint, 跑一条 prompt."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="HLR checkpoint dir (含 adapter_config.json + latent_reasoner.pt)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="base 模型路径, 不传则从 adapter_config.json 读")
    parser.add_argument("--prompt", type=str, default="Briefly say hello.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--merge_lora", action="store_true")
    args = parser.parse_args()

    engine = HLRInferenceEngine(
        checkpoint_dir=args.checkpoint,
        base_model_path=args.base_model,
        merge_lora=args.merge_lora,
    )
    text, info = engine.generate(
        [{"role": "user", "content": args.prompt}],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("\n" + "=" * 60)
    print("OUTPUT:")
    print(text)
    print("=" * 60)
    print(f"explicit={info['explicit_tokens']}  "
          f"latent_steps={info['latent_steps']} "
          f"(≈{info['latent_steps_as_tokens']} explicit tokens)  "
          f"total_equiv={info['total_equivalent_tokens']}")
    print(f"latent_segments={len(info['latent_segments'])}  "
          f"wall={info['wall_time_sec']:.2f}s  "
          f"truncated={info['truncated']}  hit_eos={info['hit_eos']}")


if __name__ == "__main__":
    main()
