"""
HLR 推理引擎 (Hierarchical Latent Reasoner) — batched online entropy 触发的混合解码.

触发规则镜像 entropy_profile.detect_latent_segments (训练侧):
  进入 latent (低熵段):
    - entropy_history 长度 >= min_entropy_samples
    - 距上次退 latent 至少 latent_cooldown 个显式 token
    - 连续 entropy_window 步熵下降
    - 当前熵 < entropy_history 的 entropy_quantile 分位数
  退出 latent:
    - latent step 数 >= max_latent_steps      OR
    - latent step 数 >= min_latent_steps 且连续 window 步熵上升

Batched 实现:
  所有样本统一走 inputs_embeds 模式, 每步同步推进:
    - 显式 sample: input_emb = main_model.embedding(sampled_token)
    - latent  sample: input_emb = LR.up_proj(LR.forward(enter_hidden, k=1, past))
    - DONE   sample: input_emb = embedding(pad), 输出丢弃
  主模型 KV cache past_len 对所有 sample 共享 (每步加 1).
  LR past_kv per-sample 独立 (只在 latent 状态的 sample 才推进).

Per-sample state:
  mode ∈ {EXPLICIT, LATENT, DONE}
  enter_hidden: 进 latent 时锁定的 main hidden (整段 LR 都用同一 h_in, 与训练一致)
  lr_past, latent_steps_done, latent_entropies
  entropy_history, tokens_since_exit, generated_tokens
  explicit_count, latent_step_count, latent_segments_meta

返回 list of (text, info), info 字段与单条一致.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hlr_config import HLRConfig
from model import build_latent_reasoner_from_main


# Per-sample state modes
_EXPLICIT = 0
_LATENT = 1
_DONE = 2


class _SampleState:
    """Per-sample 状态容器."""
    __slots__ = (
        "mode", "enter_hidden", "lr_past",
        "latent_steps_done", "latent_entropies",
        "entropy_history", "tokens_since_exit",
        "generated_tokens", "explicit_count", "latent_step_count",
        "latent_segments_meta", "truncated", "hit_eos",
    )

    def __init__(self, cooldown_init: int):
        self.mode = _EXPLICIT
        self.enter_hidden = None
        self.lr_past = None
        self.latent_steps_done = 0
        self.latent_entropies = []
        self.entropy_history = []
        # 允许首次触发, 但 history 不足会先 block
        self.tokens_since_exit = cooldown_init
        self.generated_tokens = []
        self.explicit_count = 0
        self.latent_step_count = 0
        self.latent_segments_meta = []
        self.truncated = False
        self.hit_eos = False


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

        # ── tokenizer (left padding 必须, 否则 batched last-pos 取不到 valid logits) ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ── 主模型 ──
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
        self.model_dtype = next(self.model.parameters()).dtype

        # 取主模型 input embedding (用于显式 token → embedding lookup, 走统一 inputs_embeds 路径)
        # 处理 PEFT 包装
        inner = self.model
        while not hasattr(inner, "get_input_embeddings") and hasattr(inner, "module"):
            inner = inner.module
        self.embed_layer = inner.get_input_embeddings()

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
    def _entropy(logits_1d: torch.Tensor) -> float:
        """logits_1d: [V]"""
        log_p = F.log_softmax(logits_1d.float(), dim=-1)
        return -(log_p.exp() * log_p).sum(dim=-1).item()

    @staticmethod
    def _sample(logits_1d: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return logits_1d.argmax(dim=-1).item()
        probs = F.softmax(logits_1d.float() / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _should_enter_latent(self, s: _SampleState) -> bool:
        if len(s.entropy_history) < max(self.window + 1, self.min_samples):
            return False
        if s.tokens_since_exit < self.cooldown:
            return False
        recent = s.entropy_history[-(self.window + 1):]
        falling = all(recent[i] > recent[i + 1] for i in range(self.window))
        if not falling:
            return False
        sorted_e = sorted(s.entropy_history)
        q_idx = min(int(len(sorted_e) * self.quantile), len(sorted_e) - 1)
        return s.entropy_history[-1] < sorted_e[q_idx]

    def _should_exit_latent(self, s: _SampleState) -> bool:
        if s.latent_steps_done >= self.max_latent_steps:
            return True
        if s.latent_steps_done < self.min_latent_steps:
            return False
        if len(s.latent_entropies) < self.window + 1:
            return False
        recent = s.latent_entropies[-(self.window + 1):]
        return all(recent[i] < recent[i + 1] for i in range(self.window))

    # ── batched generate ────────────────────────────────────────────

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> list[tuple[str, dict]]:
        """
        Batched HLR generation.

        Args:
            prompts: list of str (raw text) 或 list[dict] (chat 格式)
            max_new_tokens: 总 equivalent token 上限 (per sample, explicit + latent*compression)
            temperature: 0 → greedy, >0 → 采样

        Returns: list of (text, info) 与 prompts 一一对应.
        """
        start = time.perf_counter()
        B = len(prompts)
        device = self.device

        # ── 文本化 + apply chat template + 补 think 防御 ──
        texts: list[str] = []
        for p in prompts:
            if isinstance(p, list):
                t = self.tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True,
                )
            else:
                t = p
            if not t.rstrip().endswith("<think>"):
                t += "<think>\n"
            texts.append(t)

        # ── batched tokenize (left pad) ──
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
        )
        input_ids = enc.input_ids.to(device)              # [B, P_max]
        attention_mask = enc.attention_mask.to(device)    # [B, P_max]
        prompt_len = input_ids.shape[1]

        # ── prefill ──
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        main_past = out.past_key_values
        last_logits = out.logits[:, -1, :]                # [B, V]
        last_hidden = out.hidden_states[-1][:, -1, :]     # [B, H]

        # ── per-sample state init ──
        states: list[_SampleState] = [_SampleState(self.cooldown) for _ in range(B)]

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # ── 主循环: 每 iter 决定 input_emb → batched forward 一次 ──
        while True:
            # 终止 1: 全部 DONE
            if all(s.mode == _DONE for s in states):
                break
            # 终止 2: 所有未 done sample 都到 max equivalent tokens (per sample)
            #         对各 sample 单独标 truncated, 然后 break 总循环 (全 truncated 后)
            all_exceeded = True
            for s in states:
                if s.mode == _DONE:
                    continue
                eq = s.explicit_count + s.latent_step_count * self.compression_ratio
                if eq < max_new_tokens:
                    all_exceeded = False
                    break
            if all_exceeded:
                for s in states:
                    if s.mode != _DONE:
                        s.truncated = True
                        s.mode = _DONE
                break

            # Phase A: per-sample 决策 input_emb
            input_emb_list = []
            for i, s in enumerate(states):
                if s.mode == _DONE:
                    input_emb_list.append(
                        self.embed_layer(
                            torch.tensor([pad_id], device=device, dtype=torch.long)
                        )  # [1, H]
                    )
                    continue

                # 提前 cap: 个别 sample 超 max_new_tokens 就提前 done
                eq = s.explicit_count + s.latent_step_count * self.compression_ratio
                if eq >= max_new_tokens:
                    s.truncated = True
                    s.mode = _DONE
                    input_emb_list.append(
                        self.embed_layer(
                            torch.tensor([pad_id], device=device, dtype=torch.long)
                        )
                    )
                    continue

                if s.mode == _EXPLICIT:
                    if self._should_enter_latent(s):
                        # 进 latent: 不 sample 上一个 explicit 预测 (丢弃), 直接 LR step 1
                        s.mode = _LATENT
                        s.enter_hidden = last_hidden[i:i + 1].clone()  # [1, H]
                        s.lr_past = None
                        s.latent_entropies = []
                        s.latent_steps_done = 1
                        s.latent_step_count += 1
                        s.latent_segments_meta.append({
                            "enter_explicit_idx": s.explicit_count,
                        })
                        layer_h, s.lr_past = self.latent_reasoner(
                            s.enter_hidden, k=1, past_kv=None,
                        )
                        inj = self.latent_reasoner.up_proj(layer_h[-1])  # [1, 1, H]
                        input_emb_list.append(inj.squeeze(0))            # [1, H]
                    else:
                        # 继续显式: sample token from last_logits, input = embedding(token)
                        token = self._sample(last_logits[i], temperature)
                        s.generated_tokens.append(token)
                        s.explicit_count += 1
                        s.entropy_history.append(self._entropy(last_logits[i]))
                        s.tokens_since_exit += 1
                        if token == eos_id:
                            s.hit_eos = True
                            s.mode = _DONE
                            input_emb_list.append(
                                self.embed_layer(
                                    torch.tensor([pad_id], device=device, dtype=torch.long)
                                )
                            )
                            continue
                        tok_t = torch.tensor([token], device=device, dtype=torch.long)
                        input_emb_list.append(self.embed_layer(tok_t))   # [1, H]

                elif s.mode == _LATENT:
                    # 上一步 forward 是 latent step, last_logits[i] 是其 logits
                    s.latent_entropies.append(self._entropy(last_logits[i]))

                    if self._should_exit_latent(s):
                        # 退出: 段元数据收尾, sample explicit token from last_logits
                        s.latent_segments_meta[-1].update({
                            "steps": s.latent_steps_done,
                            "exit_entropy": s.latent_entropies[-1],
                        })
                        s.mode = _EXPLICIT
                        s.tokens_since_exit = 0
                        token = self._sample(last_logits[i], temperature)
                        s.generated_tokens.append(token)
                        s.explicit_count += 1
                        s.entropy_history.append(self._entropy(last_logits[i]))
                        s.tokens_since_exit += 1
                        if token == eos_id:
                            s.hit_eos = True
                            s.mode = _DONE
                            input_emb_list.append(
                                self.embed_layer(
                                    torch.tensor([pad_id], device=device, dtype=torch.long)
                                )
                            )
                            continue
                        tok_t = torch.tensor([token], device=device, dtype=torch.long)
                        input_emb_list.append(self.embed_layer(tok_t))
                    else:
                        # 继续 latent
                        s.latent_steps_done += 1
                        s.latent_step_count += 1
                        layer_h, s.lr_past = self.latent_reasoner(
                            s.enter_hidden, k=1, past_kv=s.lr_past,
                        )
                        inj = self.latent_reasoner.up_proj(layer_h[-1])
                        input_emb_list.append(inj.squeeze(0))

            # 再次检查是否全 DONE (sample EOS 可能在本 iter 触发)
            if all(s.mode == _DONE for s in states):
                break

            # Phase B: stack + batched forward
            stacked = torch.stack(input_emb_list, dim=0)   # [B, 1, H]
            stacked = stacked.to(self.model_dtype)

            # attention_mask 扩展 1 列 (新的位置, 全部 valid)
            new_col = torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_col], dim=1)

            out = self.model(
                inputs_embeds=stacked,
                attention_mask=attention_mask,
                past_key_values=main_past,
                use_cache=True,
                output_hidden_states=True,
            )
            main_past = out.past_key_values
            last_logits = out.logits[:, -1, :]
            last_hidden = out.hidden_states[-1][:, -1, :]

        # ── 收集结果 ──
        total_wall = time.perf_counter() - start
        # 按样本均分 wall (粗略, 用于 hlr_summary 累加;
        # 真正 per-sample 时间在 batched 推理里没法精确分)
        per_sample_wall = total_wall / B
        results: list[tuple[str, dict]] = []
        for s in states:
            text = self.tokenizer.decode(s.generated_tokens, skip_special_tokens=False)
            info = {
                "explicit_tokens":        s.explicit_count,
                "latent_steps":           s.latent_step_count,
                "latent_steps_as_tokens": s.latent_step_count * self.compression_ratio,
                "total_equivalent_tokens": s.explicit_count + s.latent_step_count * self.compression_ratio,
                "latent_segments":        s.latent_segments_meta,
                "entropy_history":        s.entropy_history,
                "wall_time_sec":          per_sample_wall,
                "truncated":              s.truncated,
                "hit_eos":                s.hit_eos,
            }
            results.append((text, info))
        return results

    def generate(self, prompt, max_new_tokens: int = 4096, temperature: float = 0.0):
        """单条 wrapper, 返回 (text, info)."""
        return self.generate_batch([prompt], max_new_tokens, temperature)[0]


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
