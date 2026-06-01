#!/bin/bash
# ── yangzhihan (A*STAR, 直连无 SLURM) 4B 甜点位启动器 ──────────────────────────
# 用法: 挂 tmux 后 `bash run_zhihan_v5.sh` (yangzhihan 无 sbatch, submit_*.sh 仅 zhuoyi)。
# sweep 实测甜点位 = 1 vLLM + 2 卡训练 (无 NVLink 下梯度同步中转最少, 每-completion 最低)。
# 不写死 GPU 索引: 仅把 NEED_TRAIN_PROC 锁到 2, 让 run 脚本的 zhihan 动态挑卡分支
# 自动选 2 张空闲训练卡 + 1 张 vLLM (避开被别人占的卡)。
set -euo pipefail
cd "$(dirname "$0")"

export NEED_TRAIN_PROC=2     # 动态挑卡: 2 张训练 + 1 张 vLLM (甜点位)
export PER_DEVICE_BATCH=4    # B=4, 4×2=8 % num_gen8 = 0 ✓
export ZERO_STAGE=2          # 甜点位用 ZeRO-2; 万一长序列 OOM 改成 3 (run 脚本支持 env 覆盖)
# num_gen=8 走 config 默认 (信号质量不能降到 4)

bash run_grpo_cvrp20_v5.sh
