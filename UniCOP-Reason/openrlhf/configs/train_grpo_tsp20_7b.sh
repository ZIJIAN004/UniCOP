#!/bin/bash
# OpenRLHF GRPO 训练 · TSP n=20 · DeepSeek-R1-Distill-Qwen-7B · LoRA
#
# 预期资源 (与 auto_all.sh 阶段 4 对齐):
#   - 1 卡 vLLM rollout
#   - 3 卡训练 (ZeRO-3 + offload_optimizer)
#
# 用法:
#   # 终端 1: 起 reward server
#   python reward/remote_reward_server.py --problem_type tsp --problem_size 20 --port 5000 \
#       --pomo_ckpt_dir /Data04/yangzhihan/lzj/POMO-Baseline/result \
#       --pomo_baseline_dir /Data04/yangzhihan/lzj/POMO-Baseline
#
#   # 终端 2: 跑训练
#   bash configs/train_grpo_tsp20_7b.sh

set -euo pipefail

# ── 路径 ─────────────────────────────────────────────────────────────
WORK_DIR="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason/openrlhf"
MODEL_BASE="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/output_sft_auto_20260423_024302/merged_model"
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
VLLM_GPUS=1          # 1 卡 rollout
TRAIN_GPUS=3         # 3 卡训练

# ── 训练超参 (与父目录 config.py 对齐) ───────────────────────────────
MAX_PROMPT_LEN=768
MAX_COMPLETION_LEN=4096
NUM_GENERATIONS=8
LR=5e-6
KL_COEF=0.01
CLIP_EPS_LOW=0.20               # DAPO
CLIP_EPS_HIGH=0.28              # DAPO clip-higher

# LoRA
LORA_RANK=64
LORA_ALPHA=128

# ── CUDA_HOME (DeepSpeed 编译/检查需要) ──────────────────────────────
export CUDA_HOME=/Data04/yangzhihan/envs/analog_env
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# ── NoRepeatNgram 通过环境变量注册到 vLLM ────────────────────────────
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"
export VLLM_LOGITS_PROCESSORS="openrlhf.custom.ngram_processor:NoRepeatNgramProcessor"

# ── 启动 Ray 集群 (单节点) ───────────────────────────────────────────
ray stop || true
ray start --head --num-gpus=$N_GPUS

# ── OpenRLHF GRPO 训练命令 ──────────────────────────────────────────
# 7B ZeRO-3 + gradient_checkpointing + offload_optimizer
# micro_train_batch_size=1 + train_batch_size=24 → grad_accum=8 (3卡×8=24)
# rollout_batch_size 降到 64 (相比 1.5B 的 128) 避免 vLLM 单卡 OOM

python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node $TRAIN_GPUS \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $TRAIN_GPUS \
    --vllm_num_engines $VLLM_GPUS \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.85 \
    --colocate_all_models \
    --enable_prefix_caching \
    --advantage_estimator group_norm \
    --num_generations $NUM_GENERATIONS \
    --pretrain "$MODEL_BASE" \
    --save_path "$OUTPUT_DIR" \
    --ckpt_path "$OUTPUT_DIR/ckpt" \
    --max_ckpt_num 3 \
    --save_steps 50 \
    --logging_steps 1 \
    --eval_steps 100 \
    --micro_train_batch_size 1 \
    --train_batch_size 24 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 64 \
    --max_epochs 1 \
    --num_episodes 3 \
    --prompt_max_len $MAX_PROMPT_LEN \
    --generate_max_len $MAX_COMPLETION_LEN \
    --actor_learning_rate $LR \
    --init_kl_coef $KL_COEF \
    --eps_clip_low $CLIP_EPS_LOW \
    --eps_clip_high $CLIP_EPS_HIGH \
    --prompt_data "$DATA_TRAIN" \
    --input_key messages \
    --apply_chat_template \
    --remote_rm_url "$REWARD_URL" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
    --flash_attn \
    --gradient_checkpointing \
    --zero_stage 3 \
    --bf16 \
    --use_wandb \
    --wandb_project "UniCOP-Reason-OpenRLHF" \
    --wandb_run_name "tsp20_7b_grpo_lora_$(date +%Y%m%d_%H%M%S)" \
    --generation_kwargs '{"extra_args": {"no_repeat_ngram_size": 6}}'

# ── 清理 ─────────────────────────────────────────────────────────────
ray stop
