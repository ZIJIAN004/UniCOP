#!/bin/bash
# OpenRLHF 0.10.2 GRPO 训练 · TSP n=20 · DeepSeek-R1-Distill-Qwen-7B · LoRA
#
# CLI 格式: 0.10.2 层级式 dot notation
#
# 预期资源 (与 auto_all.sh 阶段 4 对齐):
#   - 1 卡 vLLM rollout
#   - 3 卡训练 (ZeRO-3 + offload_optimizer)
#
# 用法:
#   # 终端 1: 起 reward server
#   python reward/remote_reward_server.py --problem_type tsp --problem_size 20 --port 5000 \
#       --pomo_ckpt_dir /home/ntu/lzj/POMO-Baseline/result \
#       --pomo_baseline_dir /home/ntu/lzj/POMO-Baseline
#
#   # 终端 2: 跑训练
#   bash configs/train_grpo_tsp20_7b.sh

set -euo pipefail

# ── 路径（从 paths.sh 获取） ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")/paths.sh"

WORK_DIR="$REASON_DIR/openrlhf"
MODEL_BASE="$DISTILL_DIR/output/merged_model"
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ SFT merged model 不存在: $MODEL_BASE"
    exit 1
fi
echo "[MODEL_BASE] $MODEL_BASE"
DATA_TRAIN="$WORK_DIR/data/processed/tsp20_train.jsonl"
OUTPUT_DIR="$WORK_DIR/output/tsp20_7b_grpo_lora_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

REWARD_URL="http://localhost:5000/get_reward"

# ── GPU 调度 (4 卡, 与 auto_all.sh 阶段 4 对齐) ─────────────────────
export CUDA_VISIBLE_DEVICES="0,1,2,3"
N_GPUS=4
VLLM_GPUS=1
TRAIN_GPUS=3

# ── 训练超参 (与父目录 config.py 对齐) ───────────────────────────────
MAX_LEN=4864
MAX_NEW_TOKENS=4096
NUM_SAMPLES=8
LR=5e-6
KL_COEF=0.01
CLIP_EPS_LOW=0.20                # DAPO
CLIP_EPS_HIGH=0.28               # DAPO clip-higher

# LoRA
LORA_RANK=64
LORA_ALPHA=128

# ── CUDA_HOME (paths.sh 已 export，此处确保显式) ─────────────────────

# ── NoRepeatNgram: 通过环境变量传参 ─────────────────────────────────
export NO_REPEAT_NGRAM_SIZE=6
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

# ── 启动 Ray 集群 (单节点) ───────────────────────────────────────────
ray stop || true
ray start --head --num-gpus=$N_GPUS

# ── OpenRLHF 0.10.2 GRPO 训练命令 ──────────────────────────────────
# 7B ZeRO-3 + gradient_checkpointing + offload_optimizer
# micro_batch_size=1 + batch_size=24 → grad_accum=8 (3卡×8=24)
# rollout_batch_size 降到 64 (相比 1.5B 的 128) 避免 vLLM 单卡 OOM

python -m openrlhf.cli.train_ppo_ray \
    --ref.num_nodes 1 \
    --ref.num_gpus_per_node $TRAIN_GPUS \
    --actor.num_nodes 1 \
    --actor.num_gpus_per_node $TRAIN_GPUS \
    --vllm.num_engines $VLLM_GPUS \
    --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.85 \
    --vllm.enable_prefix_caching \
    --vllm.sync_backend nccl \
    --train.colocate_all \
    --algo.advantage.estimator group_norm \
    --algo.advantage.gamma 1.0 \
    --algo.kl.init_coef $KL_COEF \
    --algo.kl.use_loss \
    --algo.kl.estimator k3 \
    --actor.eps_clip_low_high $CLIP_EPS_LOW $CLIP_EPS_HIGH \
    --rollout.n_samples_per_prompt $NUM_SAMPLES \
    --actor.model_name_or_path "$MODEL_BASE" \
    --ckpt.output_dir "$OUTPUT_DIR" \
    --ckpt.path "$OUTPUT_DIR/ckpt" \
    --ckpt.max_ckpt_num 3 \
    --ckpt.save_steps 50 \
    --ckpt.save_hf \
    --logger.logging_steps 1 \
    --eval.steps 100 \
    --train.micro_batch_size 1 \
    --train.batch_size 24 \
    --rollout.micro_batch_size 2 \
    --rollout.batch_size 64 \
    --train.max_epochs 1 \
    --train.num_episodes 3 \
    --data.max_len $MAX_LEN \
    --rollout.max_new_tokens $MAX_NEW_TOKENS \
    --actor.adam.lr $LR \
    --data.prompt_dataset "$DATA_TRAIN" \
    --data.input_key messages \
    --data.label_key instance_id \
    --data.apply_chat_template \
    --reward.remote_url "$REWARD_URL" \
    --ds.lora.rank $LORA_RANK \
    --ds.lora.alpha $LORA_ALPHA \
    --ds.lora.target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
    --actor.gradient_checkpointing_enable \
    --ds.attn_implementation flash_attn_2 \
    --ds.zero_stage 3 \
    --ds.offload_optimizer \
    --ds.param_dtype bf16 \
    --logger.use_wandb \
    --logger.wandb_project "UniCOP-Reason-OpenRLHF" \
    --logger.wandb_run_name "tsp20_7b_grpo_lora_$(date +%Y%m%d_%H%M%S)"

# ── 清理 ─────────────────────────────────────────────────────────────
ray stop
