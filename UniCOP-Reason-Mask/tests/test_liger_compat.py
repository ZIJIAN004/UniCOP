"""Liger Kernel 兼容性 / 正确性测试 (RMSNorm + RoPE + SwiGLU)。

验证 train.py 里 USE_LIGER 路径启用的三个加速 kernel 在 Qwen3 架构 + 本项目 LoRA
配置上是否正常:
  1. 前向数值等价: baseline vs Liger-patched 的 logits 一致 (top-1 一致率≈1) →
     证明 rms_norm / rope / swiglu 三个 kernel 算得对、与架构兼容。
  2. patch 真生效: patch 后 RMSNorm / MLP 被换成 Liger 实现。
  3. LoRA × SwiGLU 反传: 用本项目 LoRA 配置 (target 含 gate/up/down_proj) wrap 后,
     backward 后这三个 proj 的 LoRA adapter 梯度非零 → SwiGLU 融合没旁路掉 LoRA。

默认用一个"迷你 Qwen3"(随机初始化, 2 层, 词表 1024) → 不下载、不加载 7B/4B,
GPU 占用几 MB、秒级完成。kernel 兼容性是架构级的, 结论适用于真实 Qwen3-4B-thinking。

可选: TEST_MODEL=/path/to/qwen3-4b python tests/test_liger_compat.py
       → 改为加载真实模型 + 真实 tokenizer + 2 个真实 CVRP-20 实例做全量验证。

需要一张 GPU (Liger 是 Triton/CUDA kernel) + 含 liger-kernel 的 unicop 环境。
运行 (GPU 节点):  cd UniCOP-Reason-Mask && python tests/test_liger_compat.py
"""
import os
import gc
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

TEST_MODEL = os.environ.get("TEST_MODEL", "").strip()    # 空 = 用合成迷你 Qwen3
SYNTHETIC = (TEST_MODEL == "")
SEED = 1234
N_INSTANCES = 2
PROBLEM_SIZE = 20


def _banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def _is_liger(module) -> bool:
    """判断模块是否被 Liger patch: 兼容"替换类"和"替换 forward 方法"两种方式。"""
    if "Liger" in type(module).__name__:
        return True
    fwd = getattr(module, "forward", None)
    return "liger" in (getattr(fwd, "__module__", "") or "").lower()


def _resolve_apply_liger(model_type: str):
    """按 model_type 选 patch 函数 (与 train.py 逻辑一致)。"""
    if model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as fn
    elif model_type == "qwen2":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as fn
    else:
        raise RuntimeError(f"未适配 model_type={model_type} (仅 qwen2/qwen3)")
    return fn


# ── 模型 / 输入构造: 合成迷你模型 vs 真实模型两种模式 ─────────────────────
def make_synthetic_config():
    from transformers import Qwen3Config
    return Qwen3Config(
        vocab_size=1024, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, max_position_embeddings=1024, tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )


def build_inputs(model_type_or_cfg):
    """返回 (input_ids, attention_mask) cuda tensors。

    合成模式: 随机 token; 真实模式: 2 个真实 CVRP-20 prompt (左 padding)。
    """
    if SYNTHETIC:
        g = torch.Generator().manual_seed(SEED)
        ids = torch.randint(0, 1024, (N_INSTANCES, 64), generator=g)
        attn = torch.ones_like(ids)
        return ids.cuda(), attn.cuda()
    # 真实模式
    from transformers import AutoTokenizer
    from problems.cvrp import CVRP
    tok = AutoTokenizer.from_pretrained(TEST_MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    prob, rng, texts = CVRP(), np.random.default_rng(SEED), []
    for _ in range(N_INSTANCES):
        inst = prob.generate_instance(PROBLEM_SIZE, rng)
        texts.append(tok.apply_chat_template(
            prob.build_prompt(inst), tokenize=False, add_generation_prompt=True))
    enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True)
    return enc.input_ids.cuda(), enc.attention_mask.cuda()


_SYN_CFG = make_synthetic_config() if SYNTHETIC else None
_SYN_STATE = {}   # 缓存合成模型权重, 保证 baseline / patched 权重完全一致


def load_model():
    if SYNTHETIC:
        torch.manual_seed(SEED)
        m = AutoModelForCausalLM.from_config(_SYN_CFG, torch_dtype=torch.bfloat16)
        if _SYN_STATE:                       # 第二次构造时载入第一次的权重, 确保一致
            m.load_state_dict(_SYN_STATE)
        else:
            _SYN_STATE.update({k: v.clone() for k, v in m.state_dict().items()})
        return m.cuda().eval()
    return AutoModelForCausalLM.from_pretrained(
        TEST_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()


def get_model_type():
    if SYNTHETIC:
        return "qwen3"
    return AutoConfig.from_pretrained(TEST_MODEL, trust_remote_code=True).model_type


@torch.no_grad()
def forward_logits(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits.float()


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    results = {}
    if not torch.cuda.is_available():
        print("✗ 需要 GPU (Liger 是 Triton/CUDA kernel), 当前无 CUDA。")
        sys.exit(2)

    model_type = get_model_type()
    mode = "合成迷你 Qwen3 (随机权重, 2 层)" if SYNTHETIC else f"真实模型 {TEST_MODEL}"
    _banner(f"模式: {mode}  | model_type={model_type}")

    input_ids, attn = build_inputs(model_type)
    valid = attn.bool()
    print(f"输入 shape = {tuple(input_ids.shape)}  | 有效 token = {int(valid.sum())}")

    # ── 1. baseline 前向 (patch 之前) ────────────────────────────────────
    _banner("Step 1/4: baseline 前向 (未 patch)")
    base = load_model()
    logits_base = forward_logits(base, input_ids, attn).cpu()
    free(base)
    print("✓ baseline logits 已存")

    # ── 2. 模块级 patch (复刻 train.py) + 重新构造 ───────────────────────
    _banner("Step 2/4: apply_liger (rms_norm+rope+swiglu, CE/FLCE 关) + 重建模型")
    _resolve_apply_liger(model_type)(
        rms_norm=True, rope=True, swiglu=True,
        cross_entropy=False, fused_linear_cross_entropy=False)
    model = load_model()

    layer0 = model.model.layers[0]
    rms_patched = _is_liger(layer0.input_layernorm)
    swiglu_patched = _is_liger(layer0.mlp)
    print(f"  input_layernorm = {type(layer0.input_layernorm).__name__:24s} "
          f"{'✓ Liger' if rms_patched else '✗ 未换'}")
    print(f"  mlp             = {type(layer0.mlp).__name__:24s} "
          f"{'✓ Liger' if swiglu_patched else '✗ 未换'}")
    print("  (RoPE 是函数级 patch, 不在模块名体现, 由 Step 3 数值等价间接验证)")
    results["rms_norm_patched"] = rms_patched
    results["swiglu_patched"] = swiglu_patched

    # ── 3. 前向数值等价 (rms_norm + rope + swiglu 综合正确性) ─────────────
    _banner("Step 3/4: 前向数值等价")
    logits_liger = forward_logits(model, input_ids, attn).cpu()
    vb = valid.cpu()
    base_sel, liger_sel = logits_base[vb], logits_liger[vb]
    delta = liger_sel - base_sel
    # 主判据: 相对误差 ||Δ|| / ||base||。对 logits 是否平坦不敏感, 比 argmax 稳健
    # (随机权重的合成模型 logits 较平, argmax 会对微小 bf16 差异过敏 → 误判)。
    rel_err = (delta.norm() / base_sel.norm().clamp(min=1e-6)).item()
    max_abs = delta.abs().max().item()
    top1_agree = (base_sel.argmax(-1) == liger_sel.argmax(-1)).float().mean().item()
    print(f"  相对误差 ||Δ||/||base|| = {rel_err:.5f}  (阈值 < 0.05)")
    print(f"  max|Δ| = {max_abs:.4f}   top-1 一致率 = {top1_agree:.4f} (参考)")
    parity_ok = rel_err < 0.05
    results["forward_parity"] = parity_ok
    print(f"  {'✓ 数值等价' if parity_ok else '✗ 偏离过大, 可能 kernel 不兼容'}")

    # ── 4. LoRA × SwiGLU 反传 (gate/up/down_proj 梯度非零) ────────────────
    _banner("Step 4/4: LoRA × SwiGLU 反传 (adapter 梯度非零)")
    from peft import LoraConfig, get_peft_model
    from config import config
    lora_cfg = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    model(input_ids=input_ids, attention_mask=attn).logits.float().mean().backward()

    watch = ["gate_proj", "up_proj", "down_proj"]   # SwiGLU 融合涉及的三个 proj
    grad_norm = {k: 0.0 for k in watch}
    seen = {k: False for k in watch}
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" in name and p.grad is not None:
            for k in watch:
                if k in name:
                    grad_norm[k] += float(p.grad.detach().float().norm().item())
                    seen[k] = True
    lora_ok = True
    for k in watch:
        ok = seen[k] and grad_norm[k] > 0.0
        lora_ok = lora_ok and ok
        print(f"  {k:11s} LoRA grad_norm = {grad_norm[k]:.4e}   "
              f"{'✓' if ok else '✗ 梯度 0/缺失 → SwiGLU 旁路了 LoRA!'}")
    results["lora_swiglu_backward"] = lora_ok
    free(model)

    # ── 汇总 ─────────────────────────────────────────────────────────────
    _banner("结果汇总")
    label = {
        "rms_norm_patched":     "1a. RMSNorm 已替换为 Liger",
        "swiglu_patched":       "1b. SwiGLU 已替换为 Liger",
        "forward_parity":       "2.  前向数值等价 (rms_norm+rope+swiglu)",
        "lora_swiglu_backward": "3.  LoRA×SwiGLU 反传梯度非零",
    }
    all_ok = True
    for k in ["rms_norm_patched", "swiglu_patched", "forward_parity",
              "lora_swiglu_backward"]:
        ok = results.get(k, False)
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label[k]}")
    print("\n" + ("✅ 三部分全部正常, 可放心 export USE_LIGER=1 LIGER_SWIGLU=1"
                  if all_ok else "❌ 有项未通过, 见上面 ✗ 详情"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
