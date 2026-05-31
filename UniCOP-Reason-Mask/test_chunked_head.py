"""change A 等价性测试: 分块 LM head (_logp_from_hidden_chunked) vs 原 log_softmax.gather。

默认用一个极小的合成 Qwen3 (不加载任何真实 4B/7B 权重), 只验证逻辑:
  [1] free function 数学 + 梯度 与 log_softmax(F.linear).gather 等价
  [2] 真实模型: backbone+slice 路径 logits == model(logits_to_keep)[:,:-1] logits
  [3] 端到端 per-token logp 等价
  [4] 梯度能经新路径回流到 LoRA 参数, 且 lm_head 保持 frozen 无梯度

用法 (在 UniCOP-Reason-Mask 目录, 远程 conda 环境):
    python test_chunked_head.py
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model

import grpo_prm_trainer as G

# 强制把 chunk 调小, 让 C=13 跨多个 chunk, 真正测试分块循环
G._LOGP_CHUNK_SIZE = 4

torch.manual_seed(0)

cfg = Qwen3Config(
    vocab_size=257, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, max_position_embeddings=128, tie_word_embeddings=False,
)
peft = get_peft_model(
    Qwen3ForCausalLM(cfg),
    LoraConfig(r=4, lora_alpha=8,
               target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
               task_type="CAUSAL_LM"),
)

B, P, C = 2, 5, 13
S = P + C
ids = torch.randint(1, cfg.vocab_size, (B, S))
am = torch.ones(B, S, dtype=torch.long)
comp = ids[:, P:]                                          # [B, C] completion_ids

results = []

# ── [1] free function 数学 + 梯度 ──────────────────────────────────────
h = torch.randn(B, C, cfg.hidden_size, dtype=torch.float64, requires_grad=True)
W = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.float64)
ref = torch.log_softmax(F.linear(h, W), -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
got = G._logp_from_hidden_chunked(h, comp, W)
d1 = (ref - got).abs().max().item()
gref = torch.autograd.grad(ref.sum(), h, retain_graph=True)[0]
ggot = torch.autograd.grad(got.sum(), h)[0]
dg = (gref - ggot).abs().max().item()
ok1 = d1 < 1e-9 and dg < 1e-9
results.append(ok1)
print(f"[1] free-fn  logp diff={d1:.2e}  grad diff={dg:.2e}  -> {'PASS' if ok1 else 'FAIL'}")

# ── [2] 真实模型: backbone+slice vs model(logits_to_keep) ──────────────
peft.eval()
with torch.no_grad():
    out = peft(input_ids=ids, attention_mask=am, logits_to_keep=C + 1)
    ref_logits = out.logits[:, :-1, :]                    # [B, C, V] 原路径
causal = peft.get_base_model()
backbone, lm_head = causal.model, causal.lm_head
with torch.no_grad():
    bb = backbone(input_ids=ids, attention_mask=am, use_cache=False)
    hidden = bb.last_hidden_state
    hc = hidden[:, P - 1:-1, :]                            # [B, C, H] 新路径切片
    new_logits = lm_head(hc)
d2 = (ref_logits - new_logits).abs().max().item()
ok2 = d2 < 1e-3 and ref_logits.shape == new_logits.shape
results.append(ok2)
print(f"[2] real-model logits diff={d2:.2e}  ref{tuple(ref_logits.shape)} "
      f"new{tuple(new_logits.shape)}  -> {'PASS' if ok2 else 'FAIL'}")

# ── [3] 端到端 logp 等价 ──────────────────────────────────────────────
ref_logp = torch.log_softmax(ref_logits.float(), -1).gather(
    -1, comp.unsqueeze(-1)).squeeze(-1)
new_logp = G._logp_from_hidden_chunked(hc, comp, lm_head.weight.detach())
d3 = (ref_logp - new_logp).abs().max().item()
ok3 = d3 < 1e-3
results.append(ok3)
print(f"[3] end-to-end logp diff={d3:.2e}  -> {'PASS' if ok3 else 'FAIL'}")

# ── [4] 梯度回流 LoRA + lm_head frozen ────────────────────────────────
peft.train()
bb = backbone(input_ids=ids, attention_mask=am, use_cache=False)
hc = bb.last_hidden_state[:, P - 1:-1, :]
logp = torch_checkpoint.checkpoint(
    G._logp_from_hidden_chunked, hc, comp, lm_head.weight.detach(),
    use_reentrant=False)
(-logp.mean()).backward()
lg = [p.grad for n, p in peft.named_parameters()
      if p.requires_grad and "lora" in n.lower()]
nz = sum(1 for g in lg if g is not None and g.abs().sum().item() > 0)
head_frozen = (lm_head.weight.requires_grad is False) and (lm_head.weight.grad is None)
ok4 = len(lg) > 0 and nz == len(lg) and head_frozen
results.append(ok4)
print(f"[4] LoRA grads nonzero {nz}/{len(lg)};  lm_head frozen={head_frozen}  "
      f"-> {'PASS' if ok4 else 'FAIL'}")

print("\n=== " + ("ALL PASS ✅" if all(results) else "SOME FAILED ❌") + " ===")
