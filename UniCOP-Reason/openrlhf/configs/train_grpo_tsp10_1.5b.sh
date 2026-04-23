#!/bin/bash
# OpenRLHF 0.10.2 GRPO 训练 · TSP n=10 · DeepSeek-R1-Distill-Qwen-1.5B · LoRA
#
# CLI 格式: 0.10.2 层级式 dot notation (--algo.*, --actor.*, --ds.*, etc.)
# 参考: https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_dapo_ray_hybrid_engine.sh
#
# 预期资源:
#   - 1 卡 vLLM rollout
#   - 1 卡 POMO reward server (远程, 另起终端)
#   - 6 卡训练
#
# 用法:
#   # 终端 1: 起 reward server
#   python reward/remote_reward_server.py --problem_type tsp --problem_size 10 --port 5000
#
#   # 终端 2: 跑训练
#   bash configs/train_grpo_tsp10_1.5b.sh

set -euo pipefail

# ── 路径 ─────────────────────────────────────────────────────────────
WORK_DIR="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason/openrlhf"
MODEL_BASE="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/output_sft_auto_20260423_024302/merged_model"
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ SFT merged model 不存在: $MODEL_BASE"
    exit 1
fi
echo "[MODEL_BASE] $MODEL_BASE"
DATA_TRAIN="$WORK_DIR/data/processed/tsp10_train.jsonl"
OUTPUT_DIR="$WORK_DIR/output/tsp10_1.5b_grpo_lora_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

REWARD_URL="http://localhost:5000/get_reward"

# ── GPU 调度 ─────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
N_GPUS=8
VLLM_GPUS=1
TRAIN_GPUS=7

# ── 训练超参 (与父目录 config.py 对齐) ───────────────────────────────
MAX_LEN=4864                     # prompt (768) + completion (4096)
MAX_NEW_TOKENS=4096              # completion 上限
NUM_SAMPLES=8                    # GRPO 组内样本数
LR=5e-6
KL_COEF=0.01
CLIP_EPS_LOW=0.20                # DAPO
CLIP_EPS_HIGH=0.28               # DAPO clip-higher

# LoRA
LORA_RANK=64
LORA_ALPHA=128

# ── CUDA_HOME (DeepSpeed 编译/检查需要) ──────────────────────────────
export CUDA_HOME=/Data04/yangzhihan/envs/analog_env
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# ── NoRepeatNgram: 通过环境变量传参 ─────────────────────────────────
# ngram_processor.py 会读取此环境变量作为默认 ngram_size
export NO_REPEAT_NGRAM_SIZE=6
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

# ── 启动 Ray 集群 (单节点) ───────────────────────────────────────────
ray stop || true
ray start --head --num-gpus=$N_GPUS

# ── OpenRLHF 0.10.2 GRPO 训练命令 ──────────────────────────────────
# 所有 CLI 参数已转换为 0.10.2 层级式 dot notation
# --data.label_key instance_id → reward server 的 labels 参数会收到 instance_id 字符串
# --algo.kl.use_loss + --algo.kl.estimator k3 → GRPO/DAPO 推荐 KL 配置

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
    --train.batch_size 64 \
    --rollout.micro_batch_size 4 \
    --rollout.batch_size 128 \
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
    --ds.param_dtype bf16 \
    --logger.use_wandb \
    --logger.wandb_project "UniCOP-Reason-OpenRLHF" \
    --logger.wandb_run_name "tsp10_1.5b_grpo_lora_$(date +%Y%m%d_%H%M%S)"

# ── 清理 ─────────────────────────────────────────────────────────────
ray stop
