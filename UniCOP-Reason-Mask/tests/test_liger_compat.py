"""Liger Kernel 兼容性 / 正确性测试 (RMSNorm + RoPE + SwiGLU)。

验证 train.py 里 USE_LIGER 路径启用的三个加速 kernel 在本项目模型 + LoRA 栈上:
  1. 前向数值等价: baseline vs Liger-patched 的 logits 一致 (top-1 一致率≈1) →
     证明 rms_norm / rope / swiglu 三个 kernel 算得对、与模型架构兼容。
  2. patch 真生效: patch 后 RMSNorm / MLP 模块被换成 Liger 类。
  3. LoRA × SwiGLU 反传: 用本项目 LoRA 配置 (target 含 gate/up/down_proj) wrap 后,
     backward 后这三个 proj 的 LoRA adapter 梯度非零 → SwiGLU 融合没旁路掉 LoRA。

仅用 2 个真实 CVRP-20 实例当输入。需要一张 GPU + 完整 unicop 环境 (含 liger-kernel)。
完整复刻 train.py 的 patch 路径 (模块级 patch + 重新 from_pretrained)。

运行 (在 GPU 节点的 unicop 环境):
    cd UniCOP-Reason-Mask && python tests/test_liger_compat.py
    # 可选: TEST_MODEL=/path/to/model python tests/test_liger_compat.py
"""
import os
import gc
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from config import config
from problems.cvrp import CVRP
import numpy as np

N_INSTANCES = 2          # 用 1-2 个真实问题实例即可
PROBLEM_SIZE = 20
SEED = 1234

MODEL_NAME = os.environ.get("TEST_MODEL", config.model_name)


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


def build_inputs(tokenizer):
    """造 N_INSTANCES 个真实 CVRP-20 prompt, tokenize 成 batch (左 padding)。"""
    prob = CVRP()
    rng = np.random.default_rng(SEED)
    texts = []
    for _ in range(N_INSTANCES):
        inst = prob.generate_instance(PROBLEM_SIZE, rng)
        msgs = prob.build_prompt(inst)
        texts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    enc = tokenizer(texts, return_tensors="pt", add_special_tokens=False,
                    padding=True)
    return enc.input_ids, enc.attention_mask


@torch.no_grad()
def forward_logits(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits.float()


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    results = {}

    if not torch.cuda.is_available():
        print("✗ 需要 GPU, 当前无 CUDA。请在 GPU 节点运行。")
        sys.exit(2)

    model_type = AutoConfig.from_pretrained(
        MODEL_NAME, trust_remote_code=True).model_type
    _banner(f"模型: {MODEL_NAME}  | model_type={model_type}  | "
            f"{N_INSTANCES}×CVRP-{PROBLEM_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    input_ids, attn = build_inputs(tokenizer)
    input_ids, attn = input_ids.cuda(), attn.cuda()
    valid = attn.bool()                       # 只比非 padding 位置
    print(f"输入 shape = {tuple(input_ids.shape)} (含左 padding), "
          f"有效 token = {int(valid.sum())}")

    # ── 1. baseline 前向 (patch 之前, 必须先做) ──────────────────────────
    _banner("Step 1/4: baseline 前向 (未 patch)")
    base = load_model()
    logits_base = forward_logits(base, input_ids, attn).cpu()
    free(base)
    print("✓ baseline logits 已存")

    # ── 2. 模块级 patch (复刻 train.py) + 重新加载 ───────────────────────
    _banner("Step 2/4: 模块级 apply_liger (rms_norm+rope+swiglu, CE/FLCE 关) + 重载")
    apply_liger = _resolve_apply_liger(model_type)
    apply_liger(rms_norm=True, rope=True, swiglu=True,
                cross_entropy=False, fused_linear_cross_entropy=False)
    model = load_model()

    # 验证三部分之 rms_norm + swiglu 模块确实被替换成 Liger 类
    layer0 = model.model.layers[0]
    norm_cls = type(layer0.input_layernorm).__name__
    mlp_cls = type(layer0.mlp).__name__
    rms_patched = _is_liger(layer0.input_layernorm)
    swiglu_patched = _is_liger(layer0.mlp)
    print(f"  input_layernorm 类 = {norm_cls}   {'✓ Liger' if rms_patched else '✗ 未换'}")
    print(f"  mlp 类           = {mlp_cls}   {'✓ Liger' if swiglu_patched else '✗ 未换'}")
    print("  (RoPE 是函数级 patch, 不在模块名体现, 由 Step 3 数值等价间接验证)")
    results["rms_norm_patched"] = rms_patched
    results["swiglu_patched"] = swiglu_patched

    # ── 3. 前向数值等价 ──────────────────────────────────────────────────
    _banner("Step 3/4: 前向数值等价 (rms_norm + rope + swiglu 综合正确性)")
    logits_liger = forward_logits(model, input_ids, attn).cpu()
    # 只看有效位置
    vb = valid.cpu()
    diff = (logits_liger - logits_base)[vb]
    base_sel = logits_base[vb]
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    # top-1 (下一 token 预测) 一致率: kernel 等价时应≈1.0
    top1_base = logits_base[vb].argmax(-1)
    top1_liger = logits_liger[vb].argmax(-1)
    top1_agree = (top1_base == top1_liger).float().mean().item()
    scale = base_sel.abs().mean().item()
    print(f"  logits 量级(mean|·|)   = {scale:.3f}")
    print(f"  max|Δ| = {max_abs:.4f}   mean|Δ| = {mean_abs:.5f}")
    print(f"  top-1 一致率           = {top1_agree:.4f}")
    # 判定: top-1 一致率高 (bf16 跨多层有累积噪声, 用 argmax 一致性为主判据)
    parity_ok = (top1_agree >= 0.99) and (mean_abs < max(0.5, 0.05 * scale))
    results["forward_parity"] = parity_ok
    print(f"  {'✓ 数值等价' if parity_ok else '✗ 偏离过大, 可能 kernel 不兼容'}")

    # ── 4. LoRA × SwiGLU 反传 (gate/up/down_proj 梯度非零) ────────────────
    _banner("Step 4/4: LoRA × SwiGLU 反传 (adapter 梯度非零)")
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    out = model(input_ids=input_ids, attention_mask=attn)
    loss = out.logits.float().mean()          # 触达所有层, 含 MLP
    loss.backward()

    watch = ["gate_proj", "up_proj", "down_proj"]   # SwiGLU 融合涉及的三个 proj
    grad_norm = {k: 0.0 for k in watch}
    grad_seen = {k: False for k in watch}
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" in name and p.grad is not None:
            for k in watch:
                if k in name:
                    grad_norm[k] += float(p.grad.detach().float().norm().item())
                    grad_seen[k] = True
    lora_ok = True
    for k in watch:
        ok = grad_seen[k] and grad_norm[k] > 0.0
        lora_ok = lora_ok and ok
        print(f"  {k:11s} LoRA grad_norm = {grad_norm[k]:.4e}   "
              f"{'✓' if ok else '✗ 梯度为 0/缺失 → SwiGLU 旁路了 LoRA!'}")
    results["lora_swiglu_backward"] = lora_ok
    free(model)

    # ── 汇总 ─────────────────────────────────────────────────────────────
    _banner("结果汇总")
    label = {
        "rms_norm_patched":     "1a. RMSNorm 模块已替换",
        "swiglu_patched":       "1b. SwiGLU 模块已替换",
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
