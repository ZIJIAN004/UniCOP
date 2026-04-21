# UniCOP

LLM × Combinatorial Optimization 的端到端研究仓库。收录两个配套项目：

## 子项目

### `UniCOP-Distill/`
冷启动 SFT。用 Gemini 2.5 Pro 生成高质量 COP 推理链蒸馏数据，SFT 训练一个"懂 COP 推理"的基座，供后续 GRPO 使用。

- `generate_chains.py` — 调 Vertex AI Gemini 生成思维链
- `clean_single.py` — 数据清洗
- `train_sft.py` — SFT 训练入口
- `run_distill.sh` — 自动化 pipeline
- `lkh_solver.py` — LKH 求解器对照答案

### `UniCOP-Reason/`
GRPO + POMO PRM 训练主体。双信号（terminal 标量 + PRM per-token）联合优化 LLM。

- `train.py` — GRPO 训练入口
- `grpo_prm_trainer.py` — 双信号 GRPO Trainer
- `pomo_prm.py` — POMO Process Reward Model（TSP/CVRP/VRPTW 走 POMO，TSPTW 走 PIP-D）
- `terminal_reward.py` — 四维 terminal reward (parse + coverage + constraint + format)
- `problems/` — TSP/CVRP/TSPTW/TSPDL/VRPTW 问题定义
- `utils/vllm_ngram_processor.py` — vLLM V1 自定义 LogitsProcessor，移植 HF `no_repeat_ngram_size=6`
- `evaluate.py` — 多后端评估（HF local / vLLM / Vertex AI Gemini）
- `auto_train.sh` — 训练自动化脚本（vLLM server mode + GPU 调度）
- `preflight.sh` — 训练前 7 级 sanity check

## 训练路线

```
UniCOP-Distill:  基座模型 → SFT(Gemini 链) → merged_model
    ↓
UniCOP-Reason:   merged_model → GRPO + POMO PRM → final_model
    ↓
evaluate:        final_model × (TSP/CVRP/TSPTW/VRPTW) × (n=20/50/100)
```

## 引用外部依赖（不在本仓库）

- **POMO-Baseline**：TSP/CVRP/VRPTW 的 POMO checkpoints 来源
- **PIP-D baseline (Bi et al., NeurIPS 2024)**：TSPTW 的 PRM backbone

两者路径在 `auto_train.sh` 顶部变量里指定。

## 目标

AAAI 2027 投稿（DDL 2026-08 初）。
