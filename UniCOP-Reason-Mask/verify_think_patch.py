"""验证 think patch 是否生效 — 在 yangzhihan 主机直接跑。

背景: instruct 基座 (Qwen3-4B-Instruct-2507) 的 chat_template 末尾不带 <think>,
SFT stage2 手动补了 <think>\n 教模型, 但旧 GRPO/eval 没补 → 分布断裂导致训练塌缩、
eval 输出 <think></think> 重复。prompt_think_patch.py 修复后, 本脚本确认 patch 生效。

用法 (cwd = UniCOP-Reason-Mask):
    python verify_think_patch.py                 # 只验证 tokenizer patch (秒级, 无需 GPU)
    python verify_think_patch.py --gen           # 额外加载模型跑贪心生成对比 (需 1 GPU)
    python verify_think_patch.py --model <path>  # 换模型 (默认 SFT 产物 = GRPO 起点)
        # 想看塌缩模型补 think 后是否好转, 把 --model 指向 v6_complete/merged_model
        # (预期仍差: 模型已在错误分布上训废, 证明必须重训)
"""
import argparse
import sys

from transformers import AutoTokenizer

from prompt_think_patch import patch_chat_template_for_think

# GRPO 起点 = instruct SFT 产物 (chat_template 不带 <think>, 正是要 patch 的对象)
SFT_MODEL = ("/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/"
             "output_sft_qwen3_instruct_template_cvrp20/final_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SFT_MODEL, help="模型路径 (默认 SFT 产物)")
    ap.add_argument("--gen", action="store_true", help="额外加载模型跑贪心生成对比 (需 GPU)")
    args = ap.parse_args()

    print(f"模型: {args.model}\n")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sys_p = ("You are a logistics route planning expert solving CVRP.\n"
             "Before answering, reason step by step inside <think>...</think>.")
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": "Plan routes for a small CVRP instance."}]

    before = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("=" * 64)
    print("[BEFORE patch] prompt 末尾 60 字符:")
    print(" ", repr(before[-60:]))
    print("  末尾是 <think>? ->", before.rstrip().endswith("<think>"))

    patched = patch_chat_template_for_think(tok)
    print("=" * 64)
    print(f"patch_chat_template_for_think 返回: {patched}  "
          f"(True=instruct基座已patch / False=thinking基座无需patch)")

    after = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("[AFTER patch] prompt 末尾 60 字符:")
    print(" ", repr(after[-60:]))
    print("  末尾是 <think>? ->", after.rstrip().endswith("<think>"))
    print("=" * 64)

    if not after.rstrip().endswith("<think>"):
        print("❌ patch 未生效: 末尾仍无 <think>。jinja 不兼容, 需换 patch 方式, 告知 Claude。")
        sys.exit(1)
    print("✅ patch 生效: prompt 末尾已补 <think>, 与 SFT stage2 对齐。可以重训/重 eval。")

    if args.gen:
        import torch
        from transformers import AutoModelForCausalLM
        print("\n加载模型跑贪心生成对比 (复现 bo1 口径)...")
        m = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda")
        for tag, ct in [("不补 <think> (旧口径)", before), ("补 <think> (新口径)", after)]:
            ids = tok(ct, return_tensors="pt", add_special_tokens=False).to(m.device)
            out = m.generate(**ids, max_new_tokens=200, do_sample=False)
            gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False)
            print(f"\n=== {tag} 贪心输出前 200 token ===")
            print(repr(gen))


if __name__ == "__main__":
    main()
