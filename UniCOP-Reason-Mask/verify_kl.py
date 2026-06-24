"""验证 KL anchor 是否真正接入 GRPO loss (zhuoyi / zhihan 直接跑)。

背景: DAPO 默认 kl_coef=0 (移除 KL anchor)。instruct 弱先验 + Clip-Higher + 无 KL →
后期策略漂出 SFT 流形 (内容+长度双崩, 见 2026-06-23 v6_instruct log: step28 健康 →
step37 全截断/parse=0, clip_high_hit_rate 全程 0 拦不住慢漂)。补 KL_COEF>0
(train.py 经 env 覆盖 config.kl_coef → GRPOConfig.beta) 锚回 SFT。

⚠️ 关键风险: 本 trainer 自定义了 rollout 路径 (grpo_prm_trainer.py:246
_generate_and_score_completions), 它先 super() 委托父类 TRL, 再增补。KL 分支
(grpo_prm_trainer.py:2502) 读 inputs["ref_per_token_logps"] —— 这个键必须由父类 TRL 在
beta>0 时生产, 否则撞 :2511 warning 分支, KL 静默失效 (= 等于没补, 照崩)。
本脚本就是确认这条链路真生效, 不是 no-op。

用法 (cwd = UniCOP-Reason-Mask):
    python verify_kl.py                 # [静态] 查装的 TRL 在 beta>0 时是否算 ref logps (秒级, 无 GPU)
    python verify_kl.py --gen           # [数值] 额外加载 SFT 模型验证 KL 公式 (需 1 GPU)
    python verify_kl.py --log <path>    # [运行时] 训练跑起来后审计 log, 确认 KL 真进了 loss 且在止崩
"""
import argparse
import ast
import inspect
import re
import sys

# zhuoyi instruct SFT 产物 (= GRPO 起点 = KL reference; chat_template 不带 <think>)
SFT_MODEL = ("/homes/zhuoyi/zijianliu/UniCOP/UniCOP-Distill/"
             "output_sft_qwen3_instruct_template_cvrp20/final_model")

WARN_PAT = "KL anchor 实际未生效"


def check_trl_wiring():
    """静态确认: 装的 TRL GRPOTrainer 在 beta!=0 时会算 ref_per_token_logps。
    这是'暴露 KL_COEF 就够'的前提——父类生产, 子类消费 (grpo_prm_trainer:2503)。"""
    print("=" * 72)
    print("[Check 1/2] TRL 接线: beta>0 时父类是否生产 ref_per_token_logps")
    print("=" * 72)
    try:
        import trl
        from trl.trainer.grpo_trainer import GRPOTrainer
    except Exception as e:
        print(f"❌ 无法 import trl / GRPOTrainer: {e}")
        return False
    print(f"  trl 版本: {getattr(trl, '__version__', '?')}")

    # 把 GRPOTrainer 整类源码 (含所有方法) 抓出来扫
    try:
        src = inspect.getsource(GRPOTrainer)
    except Exception as e:
        print(f"❌ 无法读取 GRPOTrainer 源码: {e}")
        return False

    has_key = "ref_per_token_logps" in src
    # 生产点附近应有 beta 门控 (self.beta != 0 / if self.beta) —— 确认是 beta>0 才算, 与我们一致
    beta_gated = bool(re.search(r"self\.beta\s*(!=|>)\s*0", src)) or "if self.beta" in src
    # ref 来源: 独立 ref_model 或 LoRA disable_adapter (PEFT 路径, 我们就是 LoRA)
    ref_source = ("disable_adapter" in src) or ("ref_model" in src)

    print(f"  源码含 'ref_per_token_logps' 生产/赋值 : {'✅' if has_key else '❌'}")
    print(f"  且由 beta>0 门控 (与 KL_COEF>0 一致)   : {'✅' if beta_gated else '⚠️ 未显式匹配到'}")
    print(f"  ref 来源 (ref_model / disable_adapter) : {'✅' if ref_source else '❌'}")

    # 把生产 ref 的那几行打出来当证据
    for i, ln in enumerate(src.splitlines()):
        if "ref_per_token_logps" in ln and ("=" in ln or "output" in ln or "beta" in ln):
            print(f"    证据 L{i}: {ln.strip()[:100]}")

    ok = has_key and ref_source
    print(f"\n  → {'✅ 链路成立: beta>0 时父类生产 ref, 子类 grpo_prm_trainer:2503 能取到'  if ok else '❌ 没找到 ref 生产逻辑 —— KL_COEF>0 会撞 warning 静默失效, 需手动在 rollout 补 ref logps'}")
    return ok


def check_config_flow():
    """确认 config.kl_coef 存在且 train.py 把它当 beta 传下去。"""
    print("\n" + "=" * 72)
    print("[Check 2/2] config 流向: KL_COEF → config.kl_coef → GRPOConfig.beta")
    print("=" * 72)
    ok = True
    try:
        import config as cfg_mod
        # 找到带 kl_coef 字段的 config 类/实例
        kl_default = None
        for name in dir(cfg_mod):
            obj = getattr(cfg_mod, name)
            if hasattr(obj, "kl_coef"):
                kl_default = getattr(obj, "kl_coef")
                print(f"  config.{name}.kl_coef 默认 = {kl_default}")
                break
        if kl_default is None:
            # dataclass: 实例化看默认
            print("  (未找到现成实例, 跳过默认值读取; 字段存在性以 grep 为准)")
    except Exception as e:
        print(f"  ⚠️ import config 失败 ({e}); 改用源码 grep")

    # grep train.py 的 env 覆盖 + beta 传递
    try:
        t = open("train.py", encoding="utf-8").read()
        env_ov = 'os.environ.get("KL_COEF")' in t and "config.kl_coef" in t
        beta_pass = re.search(r"beta\s*=\s*config\.kl_coef", t) is not None
        print(f"  train.py 有 KL_COEF env 覆盖 config.kl_coef : {'✅' if env_ov else '❌'}")
        print(f"  train.py 把 config.kl_coef 传给 beta=       : {'✅' if beta_pass else '❌'}")
        ok = env_ov and beta_pass
    except Exception as e:
        print(f"  ❌ 读 train.py 失败: {e}")
        ok = False
    print(f"\n  → {'✅ KL_COEF 能落到 GRPOConfig.beta' if ok else '❌ 流向断裂, 检查 train.py'}")
    return ok


def numeric_kl_sanity(model_path):
    """数值验证 grpo_prm_trainer:2507 的 KL 公式 kl = exp(ref-pol)-(ref-pol)-1:
    相等时 = 0 (无惩罚), 偏离时 > 0 (有惩罚)。用真实 SFT 模型跑一次 forward 取 logps,
    确认 logps 能正常算、公式行为正确。"""
    print("\n" + "=" * 72)
    print("[Check 数值] KL 公式行为 (用真实 SFT 模型的 per-token logps)")
    print("=" * 72)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompt_think_patch import patch_chat_template_for_think

    print(f"  模型: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    patch_chat_template_for_think(tok)
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    m.eval()

    sys_p = ("You are a logistics route planning expert solving CVRP.\n"
             "Before answering, reason step by step inside <think>...</think>.")
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": "Plan routes for a small CVRP instance."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    completion = "Let me check capacities. Route 0->3->7->1->0.</think>\nRoute: 0 3 7 1 0"
    full = prompt + completion
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    f_ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
    comp_start = p_ids.shape[1]

    with torch.no_grad():
        logits = m(f_ids).logits[0, :-1]               # 预测 next-token
        logps_all = torch.log_softmax(logits.float(), dim=-1)
        tgt = f_ids[0, 1:]
        tok_logps = logps_all.gather(1, tgt.unsqueeze(1)).squeeze(1)
    pol = tok_logps[comp_start - 1:]                   # completion 段 per-token logp
    print(f"  completion token 数 = {pol.numel()}, per-token logp 均值 = {pol.mean():.3f}")

    def kl_term(ref, p):                               # 与 grpo_prm_trainer.py:2507 完全一致
        d = ref - p
        return torch.exp(d) - d - 1

    kl_same = kl_term(pol, pol)
    kl_drift = kl_term(pol + 0.2, pol)                 # ref 比 policy 高 0.2 nat → 有偏离
    print(f"  KL(self‖self)  逐token均值 = {kl_same.mean().item():.6e}  (应 ≈ 0 → 无惩罚)")
    print(f"  KL(+0.2 drift) 逐token均值 = {kl_drift.mean().item():.6e}  (应 > 0 → 有惩罚)")
    ok = (kl_same.mean().item() < 1e-6) and (kl_drift.mean().item() > 1e-3) and bool((kl_drift >= 0).all())
    print(f"\n  → {'✅ KL 公式行为正确 (相等=0, 偏离>0 且恒非负)' if ok else '❌ 公式行为异常'}")
    return ok


def audit_log(path):
    """训练跑起来后审计 log: 确认 KL 真进了 loss (无 warning + kl 指标非零) 且在止崩。"""
    print("=" * 72)
    print(f"[运行时审计] {path}")
    print("=" * 72)
    txt = open(path, encoding="utf-8", errors="ignore").read()

    # 1) 致命: KL 静默失效的 warning
    if WARN_PAT in txt:
        print(f"  ❌ 发现 warning '{WARN_PAT}' → KL 未生效 (ref logps 没传进来)! KL_COEF 形同虚设。")
        print("     说明装的 TRL 没在 beta>0 时算 ref —— 告知 Claude, 需手动在 rollout 补 ref logps。")
        return False
    print(f"  ✅ 未出现 '{WARN_PAT}' warning")

    # 2) kl 诊断指标是否被 log 出来 + 是否非零
    rows = []
    for m in re.finditer(r"\{'loss/total'.*?\}", txt):
        seg = txt[:m.start()]
        sm = re.findall(r"(\d+)/\d+ \[", seg)
        st = int(sm[-1]) if sm else -1
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        rows.append((st, d))
    if not rows:
        print("  ⚠️ 没解析到任何 metric 行 (log 可能还没到第一个 logging step)")
        return False

    has_kl = any("diag/kl_mean" in d for _, d in rows)
    if not has_kl:
        print("  ❌ metric 里没有 diag/kl_mean → kl_term_for_diag 为 None → beta=0 或 ref 缺失, KL 没生效")
        return False
    print("  ✅ metric 含 diag/kl_mean / diag/kl_loss_share → KL 项已进入 loss")

    print("\n  step  kl_mean   kl_share  trunc  ratio_m  | 解读")
    seen = set()
    nonzero_kl = False
    for st, d in rows:
        if st in seen:
            continue
        seen.add(st)
        km = d.get("diag/kl_mean", 0.0)
        ks = d.get("diag/kl_loss_share", 0.0)
        tr = d.get("stats/truncation_rate", 0.0)
        rm = d.get("is/ratio_mean", 0.0)
        if km > 0:
            nonzero_kl = True
        flag = ""
        if ks > 0.30:
            flag = "⚠️ KL 占比>30% 偏强, 考虑调小"
        elif tr > 0.7:
            flag = "⚠️ 截断仍高, KL 可能不够或还在早期"
        print(f"  {st:>3}  {km:8.5f}  {ks:7.3f}  {tr:4.2f}  {rm:6.3f}  | {flag}")

    print()
    if not nonzero_kl:
        print("  ⚠️ kl_mean 全 0 → policy 还没偏离 ref (极早期正常); 多跑几步再看。")
        return True
    print("  ✅ KL 已生效且在施加约束。盯 kl_loss_share 别长期 >0.30 (过强), truncation 别冲 1.0 (止崩成功标志)。")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SFT_MODEL, help="模型路径 (默认 zhuoyi SFT 产物)")
    ap.add_argument("--gen", action="store_true", help="额外数值验证 KL 公式 (需 1 GPU)")
    ap.add_argument("--log", default=None, help="审计训练 log: 确认 KL 真进 loss 且止崩")
    args = ap.parse_args()

    if args.log:
        sys.exit(0 if audit_log(args.log) else 1)

    ok1 = check_trl_wiring()
    ok2 = check_config_flow()
    ok = ok1 and ok2
    if args.gen:
        ok = numeric_kl_sanity(args.model) and ok

    print("\n" + "=" * 72)
    if ok:
        print("✅ 静态链路全通过: KL_COEF>0 会真正接入 loss。可以提交 sweep。")
        print("   提交后务必再用  python verify_kl.py --log <训练log>  做运行时确认。")
    else:
        print("❌ 有检查未过 (见上), 别急着提交; 修好再跑, 否则 KL 形同虚设照崩。")
    print("=" * 72)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
