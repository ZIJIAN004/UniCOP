#!/bin/bash
# OpenRLHF 0.10.2 GRPO 训练 · CVRP n=20 · DeepSeek-R1-Distill-Qwen-7B (SFT-hybrid-cvrp20) · LoRA
#
# 资源 (5 卡 zhuoyi A5000 24GB):
#   - GPU 0:  vLLM rollout + reward server (POMO PRM ~10MB,与 vLLM 共卡)
#   - GPU 1-4: ZeRO-3 + LoRA + offload_optimizer 训练
#
# 设计要点:
#   - 基座模型直接用 SFT-hybrid-cvrp20 产物 (与 SFT 同分布,无 OOD 风险)
#   - reward server 在脚本内后台启动,EXIT trap 清理
#   - max_new_tokens=4096 与现有 7B 模板对齐
#   - n_samples=8 GRPO 组内样本数,batch_size=32 = 4 卡 × grad_accum 8

set -euo pipefail

# ── 路径(从 paths.sh 获取 POMO/CUDA_HOME/LKH 等;不用其推导 BASE_MODEL) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")/paths.sh"

WORK_DIR="$REASON_DIR/openrlhf"

# 用户指定的 SFT-hybrid-cvrp20 完整权重 (run_sft_hybrid_cvrp20.sh Step 2 已合并 LoRA)
MODEL_BASE="$DISTILL_DIR/output_sft_hybrid_cvrp20/final_model"
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ SFT 模型不存在: $MODEL_BASE"
    exit 1
fi
if [ ! -f "$MODEL_BASE/config.json" ]; then
    echo "❌ $MODEL_BASE 下无 config.json,可能仍是 LoRA adapter,先合并"
    exit 1
fi
echo "[MODEL_BASE] $MODEL_BASE"

DATA_TRAIN="$WORK_DIR/data/processed/cvrp20_train.jsonl"
INSTANCES_TRAIN="$WORK_DIR/data/processed/cvrp20_train_instances.json"
for f in "$DATA_TRAIN" "$INSTANCES_TRAIN"; do
    if [ ! -f "$f" ]; then
        echo "❌ 数据文件不存在: $f"
        echo "   先跑: python $WORK_DIR/data/prepare_dataset.py --problem_type cvrp --problem_size 20 --num_train 20000"
        exit 1
    fi
done
echo "[DATA] $DATA_TRAIN"

OUTPUT_DIR="$WORK_DIR/output/cvrp20_7b_grpo_lora_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

# ── 通知 (Server 酱) ──────────────────────────────────────────────────
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

# ── GPU 分配 ─────────────────────────────────────────────────────────
# SLURM 把 5 张卡映射到 CUDA_VISIBLE_DEVICES=0,1,2,3,4 (从 SLURM 视角看是连续的)
N_GPUS=5
VLLM_GPUS=1
TRAIN_GPUS=4

# ── 训练超参 ────────────────────────────────────────────────────────
MAX_LEN=4864                     # 与现有 7B 模板对齐
MAX_NEW_TOKENS=4096
NUM_SAMPLES=8                    # GRPO 组内样本数
LR=5e-6
KL_COEF=0.01
CLIP_EPS_LOW=0.20                # DAPO
CLIP_EPS_HIGH=0.28               # DAPO clip-higher

LORA_RANK=64
LORA_ALPHA=128

# ── reward server ──────────────────────────────────────────────────
REWARD_PORT=5000
REWARD_URL="http://127.0.0.1:${REWARD_PORT}/get_reward"
REWARD_LOG="$LOG_DIR/reward_server_cvrp20_$(date +%Y%m%d_%H%M%S).log"
REWARD_PID=""

cleanup() {
    local exit_code=$?
    if [ -n "$REWARD_PID" ] && kill -0 "$REWARD_PID" 2>/dev/null; then
        echo "[cleanup] 关闭 reward server pid=$REWARD_PID"
        kill "$REWARD_PID" 2>/dev/null || true
        wait "$REWARD_PID" 2>/dev/null || true
    fi
    ray stop 2>/dev/null || true
    if [ "$exit_code" != "0" ]; then
        notify "❌ UniCOP CVRP20 GRPO 异常退出" \
"退出码: $exit_code
时间: $(date '+%Y-%m-%d %H:%M:%S')
日志末尾:
$(tail -n 30 "$REWARD_LOG" 2>/dev/null || echo '(无 reward 日志)')"
    fi
}
trap cleanup EXIT INT TERM

# ── NoRepeatNgram + PYTHONPATH ─────────────────────────────────────
export NO_REPEAT_NGRAM_SIZE=6
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

# ── 启动 reward server (后台,共 GPU 0 即 vLLM 卡) ────────────────────
echo "[$(date '+%H:%M:%S')] 启动 reward server (cvrp n=20, port $REWARD_PORT)"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
    python "$WORK_DIR/reward/remote_reward_server.py" \
    --problem_type cvrp \
    --problem_size 20 \
    --port $REWARD_PORT \
    --pomo_ckpt_dir "$POMO_CKPT_DIR" \
    --pomo_baseline_dir "$POMO_BASELINE_DIR" \
    > "$REWARD_LOG" 2>&1 &
REWARD_PID=$!
echo "  reward server pid=$REWARD_PID, log=$REWARD_LOG"

# 等 reward server 就绪 (POMO ckpt 加载约 10-30s)
WAITED=0
while [ $WAITED -lt 120 ]; do
    if ! kill -0 "$REWARD_PID" 2>/dev/null; then
        echo "❌ reward server 启动失败,详见 $REWARD_LOG"
        tail -n 50 "$REWARD_LOG"
        exit 1
    fi
    if curl -s "http://127.0.0.1:${REWARD_PORT}/health" | grep -q '"ok":true'; then
        echo "[$(date '+%H:%M:%S')] ✓ reward server 就绪 (用时 ${WAITED}s)"
        break
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done
if [ $WAITED -ge 120 ]; then
    echo "❌ reward server 启动超时"
    exit 1
fi

# ── 启动 Ray 集群 ─────────────────────────────────────────────────
ray stop || true
ray start --head --num-gpus=$N_GPUS

notify "🚀 CVRP20 GRPO 启动" \
"基座: $MODEL_BASE
数据: $DATA_TRAIN
GPU: 1 vLLM + $TRAIN_GPUS 训练
QOS: long, 5 卡
开始: $(date '+%Y-%m-%d %H:%M:%S')"

# ── OpenRLHF 0.10.2 GRPO 训练命令 ──────────────────────────────────
# 7B + ZeRO-3 + offload_optimizer + grad_ckpt
# micro_batch=1, batch_size=32 → grad_accum=8 (4 卡 × 8 = 32)
# rollout_batch=64 (与 7B 模板对齐),CVRP n=20 prompt 短显存压力小

python -m openrlhf.cli.train_ppo_ray \
    --ref.num_nodes 1 \
    --ref.num_gpus_per_node $TRAIN_GPUS \
    --actor.num_nodes 1 \
    --actor.num_gpus_per_node $TRAIN_GPUS \
    --vllm.num_engines $VLLM_GPUS \
    --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.80 \
    --vllm.enable_prefix_caching \
    --vllm.sync_backend nccl \
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
    --train.batch_size 32 \
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
    --logger.wandb_run_name "cvrp20_7b_grpo_lora_$(date +%Y%m%d_%H%M%S)"

TRAIN_EC=$?

if [ $TRAIN_EC -eq 0 ]; then
    notify "✅ CVRP20 GRPO 训练完成" \
"ckpt: $OUTPUT_DIR
结束: $(date '+%Y-%m-%d %H:%M:%S')"
fi

exit $TRAIN_EC
