#!/bin/bash
# OpenRLHF GRPO 训练 · TSP n=10 · DeepSeek-R1-Distill-Qwen-1.5B · LoRA
#
# 预期资源:
#   - 1 卡 vLLM rollout
#   - 1 卡 POMO reward server (远程)
#   - 6 卡训练 (FSDP2 分片)
#
# 用法:
#   # 终端 1: 起 reward server
#   python reward/remote_reward_server.py --problem_type tsp --problem_size 10 --port 5000
#
#   # 终端 2 (另一个): 跑训练
#   bash configs/train_grpo_tsp10_1.5b.sh

set -euo pipefail

# ── 路径 ─────────────────────────────────────────────────────────────
WORK_DIR="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason/openrlhf"
# 与父目录 auto_train.sh 一致: 扫 UniCOP-Distill.bak_* 里最新一份 SFT 产物
MODEL_BASE=$(ls -d /Data04/yangzhihan/lzj/UniCOP-Distill.bak_*/output_sft_r1_v2/merged_model 2>/dev/null | sort -r | head -1)
if [ -z "$MODEL_BASE" ]; then
    echo "❌ 找不到 /Data04/yangzhihan/lzj/UniCOP-Distill.bak_*/output_sft_r1_v2/merged_model"
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
VLLM_GPUS=1          # 1 卡 rollout
TRAIN_GPUS=7         # 7 卡训练 (POMO reward server 放在 CPU 上跑或者复用 vllm 卡)

# ── 训练超参 (与父目录 config.py 对齐) ───────────────────────────────
MAX_PROMPT_LEN=768
MAX_COMPLETION_LEN=4096
NUM_GENERATIONS=8               # GRPO 组内样本数
LR=1e-6
KL_COEF=0.01
CLIP_EPS_LOW=0.20               # DAPO
CLIP_EPS_HIGH=0.28              # DAPO clip-higher

# LoRA
LORA_RANK=64
LORA_ALPHA=128

# ── NoRepeatNgram 通过环境变量注册到 vLLM ────────────────────────────
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"
export VLLM_LOGITS_PROCESSORS="openrlhf.custom.ngram_processor:NoRepeatNgramProcessor"

# ── 启动 Ray 集群 (单节点) ───────────────────────────────────────────
ray stop || true
ray start --head --num-gpus=$N_GPUS

# ── OpenRLHF GRPO 训练命令 ──────────────────────────────────────────
# 关键参数说明:
#   --algo grpo                 选 GRPO 算法 (也可换 dapo / reinforce++_baseline)
#   --advantage_estimator group_norm    GRPO 的组归一化 advantage
#   --eps_clip_low/high         DAPO 非对称 clip
#   --remote_rm_url             远程 reward server
#   --actor_num_gpus_per_node   训练并行度
#   --vllm_num_engines          vLLM rollout 并行度
#   --vllm_gpu_memory_utilization 0.85   与父目录 VLLM_GPU_MEM_UTIL 一致
#   --enable_prefix_caching     GRPO 组内同 prompt prefix 复用
#   --lora_rank / lora_alpha    LoRA 配置
#   --flash_attn                启用 FA2
#   --gradient_checkpointing    激活重计算
#   --zero_stage 3              DeepSpeed ZeRO-3 (也可换 --use_fsdp2)
#   --generation_kwargs         透传给 SamplingParams, 其中 extra_args 进入 ngram processor

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
    --train_batch_size 64 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 128 \
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
    --wandb_run_name "tsp10_1.5b_grpo_lora_$(date +%Y%m%d_%H%M%S)" \
    --generation_kwargs '{"extra_args": {"no_repeat_ngram_size": 6}}'

# ── 清理 ─────────────────────────────────────────────────────────────
ray stop
