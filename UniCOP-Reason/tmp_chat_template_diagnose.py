#!/usr/bin/env python
"""
诊断 chat_template 在 SFT 训练 vs 推理时的 <think> token 行为是否一致。

用法 (服务器上):
    cd /Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason
    python tmp_chat_template_diagnose.py
    # 或指定 model_path:
    python tmp_chat_template_diagnose.py --model_path /path/to/merged_model

诊断三种可能:
    A) 推理单 <think> + SFT 双 <think>      → 分布错位(最可能的 bug)
    B) 推理单 <think> + SFT 单 <think>      → 分布一致,</think> 忘写是温度/SFT 质量问题
    C) 推理无 <think>  + SFT 单 <think>     → 模型该自己生成 <think> 但没生成
"""
import argparse
import glob
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None,
                    help="merged_model 路径。默认自动 glob 最新 bak")
    args = ap.parse_args()

    # 1. 解析 model_path
    if args.model_path:
        model_path = args.model_path
    else:
        pattern = "/Data04/yangzhihan/lzj/UniCOP-Distill.bak_*/output_sft_r1_v2/merged_model"
        candidates = sorted(glob.glob(pattern), reverse=True)
        if not candidates:
            sys.exit(f"❌ 自动 glob 失败: {pattern}\n"
                     "   请手动指定 --model_path")
        model_path = candidates[0]

    print(f"[model_path] {model_path}\n")
    if not os.path.isdir(model_path):
        sys.exit(f"❌ 路径不存在: {model_path}")

    # 2. 加载 tokenizer
    print("加载 tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  tokenizer class: {type(tok).__name__}")
    print(f"  vocab size:      {tok.vocab_size}")

    # 看 <think> / </think> 是不是单 special token
    think_ids = tok.encode("<think>", add_special_tokens=False)
    think_end_ids = tok.encode("</think>", add_special_tokens=False)
    print(f"  '<think>'  encode 为 {len(think_ids)} 个 token: {think_ids}")
    print(f"  '</think>' encode 为 {len(think_end_ids)} 个 token: {think_end_ids}")
    print()

    # 3. 构造推理 prompt (只有 system + user, add_generation_prompt=True)
    prompt_msgs = [
        {"role": "system", "content": "You are a route planning expert solving TSP."},
        {"role": "user",   "content": "Plan a route for TSP with 3 nodes: 0, 1, 2."},
    ]
    text_eval = tok.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    # 4. 构造 SFT 训练文本 (有 assistant, add_generation_prompt=False)
    sft_msgs = prompt_msgs + [
        {"role": "assistant",
         "content": "<think>\nI need to find the shortest route.\n</think>\nRoute: 0 -> 1 -> 2 -> 0"},
    ]
    text_sft = tok.apply_chat_template(
        sft_msgs, tokenize=False, add_generation_prompt=False
    )

    # 5. 打印两段对比
    print("=" * 80)
    print("A. 推理 prompt 末尾 200 字 (inference 时模型看到的 context 末尾)")
    print("=" * 80)
    print(repr(text_eval[-200:]))
    print()

    # 6. 定位 SFT 训练文本里 assistant 段
    print("=" * 80)
    print("B. SFT 训练文本里 assistant 段前 250 字")
    print("=" * 80)
    # 找最后一个 "assistant" 出现的位置附近
    assistant_idx = text_sft.rfind("assistant")
    if assistant_idx == -1:
        print(f"⚠️ 找不到 'assistant' 关键词,打印全文:")
        print(repr(text_sft))
    else:
        # 往前 20 字，往后 250 字
        start = max(0, assistant_idx - 20)
        print(repr(text_sft[start:assistant_idx + 250]))
    print()

    # 7. 自动诊断
    print("=" * 80)
    print("C. 自动诊断")
    print("=" * 80)

    # 检测推理 prompt 末尾是否有 <think>
    eval_tail = text_eval.rstrip()
    eval_has_think = eval_tail.endswith("<think>") or "<think>" in text_eval[-30:]

    # 检测 SFT assistant 段是否双 <think>
    if assistant_idx != -1:
        assistant_segment = text_sft[assistant_idx:assistant_idx + 300]
        think_count_in_assistant = assistant_segment.count("<think>")
    else:
        think_count_in_assistant = text_sft.count("<think>")

    print(f"推理 prompt 末尾是否有 '<think>': {eval_has_think}")
    print(f"SFT assistant 段里 '<think>' 出现次数: {think_count_in_assistant}")
    print()

    if eval_has_think and think_count_in_assistant >= 2:
        print("🔴 场景 A: 推理单 <think> + SFT 双 <think> → 分布错位 (最可能的 bug)")
        print("   修复: 在 train_sft.py 里剥离 output 开头的 <think>,然后重训 SFT")
        print("   或短期 hack: 推理 prompt 手动再加一个 <think>\\n")
    elif eval_has_think and think_count_in_assistant == 1:
        print("🟢 场景 B: 推理单 <think> + SFT 单 <think> → 分布一致")
        print("   chat_template 正确处理了 output 开头的 <think>")
        print("   => UNPARSED 样本缺 </think> 的原因: 温度 drift / SFT 量不足 / 其他")
    elif not eval_has_think and think_count_in_assistant >= 1:
        print("🟡 场景 C: 推理无 <think> + SFT 有 <think>")
        print("   chat_template 不在 prompt 末尾加 <think>,模型需要自己生成")
        print("   但 observed completion 不以 <think> 开头 → SFT 没学好")
        print("   修复: 推理侧也在 prompt 末尾手动加 <think>\\n")
    else:
        print("⚠️ 意外情况: eval_has_think=False AND think_count_in_assistant=0")
        print("   chat_template 可能根本不识别 <think> → tokenizer 配置异常")
        print("   建议先检查 tokenizer_config.json 的 chat_template 字段")

    print()
    print("=" * 80)
    print("D. chat_template 原文 (前 500 字)")
    print("=" * 80)
    template = tok.chat_template or "(空)"
    print(template[:500])
    if len(template) > 500:
        print(f"... (共 {len(template)} 字, 后略)")


if __name__ == "__main__":
    main()
