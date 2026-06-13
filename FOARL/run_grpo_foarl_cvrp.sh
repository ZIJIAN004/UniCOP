#!/bin/bash
# run_grpo_foarl_cvrp.sh — FOARL CVRP GRPO RL, 直连主机版 (zhihan 无 SLURM, 挂 tmux 跑)
#   对标 submit_grpo_foarl_cvrp.sh(zhuoyi/sbatch), 但: 动态挑卡 + 无 preflight/sbatch。
#   流程同 submit: [1/3] 训练(1 vLLM + 6 训练) → [2/3] merge → [3/3] BO1 eval。
#   batch/采样/数据/奖励权重默认全对齐 Mask (见 submit 注释)。
#
#   ⚠️ 先激活环境再跑 (zhihan conda 由你的 shell 提供):
#        conda activate /Data04/yangzhihan/envs/unicop   # 或 conda activate unicop
#   ⚠️ 必填 MODEL = merged SFT 模型目录 (RL 起点)。
#   用法 (挂 tmux):
#     tmux new -s foarl
#     MODEL=/Data04/yangzhihan/lzj/UniCOP/FOARL/output_sft_foarl_cvrp20/merged \
#       bash run_grpo_foarl_cvrp.sh
#   sanity: SANITY=1 MODEL=<merged> bash run_grpo_foarl_cvrp.sh
#   手动指定卡: VLLM_GPU=7 TRAIN_GPUS_CSV=0,1,2,3,4,5 MODEL=<merged> bash run_grpo_foarl_cvrp.sh

set -uo pipefail
_SELF="$(cd "$(dirname "$0")" && pwd)"

# paths.sh 取 HOST_ID/UNICOP_ROOT/MASK_DIR/CUDA_HOME/NCCL 默认 (BASE_MODEL_TYPE 仅借它走 host 分支,
# FOARL 用自己的 MODEL/采样, 不依赖 paths.sh 的模型)。
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source "$(dirname "$_SELF")/paths.sh"

command -v accelerate >/dev/null 2>&1 || { echo "[FATAL] accelerate 不在 PATH。先 conda activate unicop 再跑。"; exit 1; }
command -v trl >/dev/null 2>&1        || { echo "[FATAL] trl 不在 PATH (vllm-serve 需要)。先激活环境。"; exit 1; }

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# zhihan 单机默认开 P2P/SHM(=0, 更快); 其它主机禁(=1, 否则 ZeRO-3 init hang)。可 env 覆盖。
if [ "${HOST_ID:-}" = "astar-zhihan" ]; then _NCCL_DEFAULT=0; else _NCCL_DEFAULT=1; fi
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-$_NCCL_DEFAULT}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-$_NCCL_DEFAULT}"

# ── 可覆盖参数 (默认对齐 Mask, 见 submit_grpo_foarl_cvrp.sh) ─────────────
MODEL="${MODEL:-}"
DATA="${DATA:-data/foarl_cvrp20_mask1000.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_grpo_foarl_cvrp20}"
S="${S:-8}"; LR="${LR:-2e-5}"; BETA="${BETA:-0.05}"; EPS="${EPS:-0.1}"; EPS_HIGH="${EPS_HIGH:-0.28}"
PDTB="${PDTB:-4}"; GA="${GA:-8}"; EPOCHS="${EPOCHS:-1}"; MAX_STEPS="${MAX_STEPS:--1}"
# ⚠️ 不能叫 TEMP: 系统/conda 常把 TEMP 设成临时目录, ${TEMP:-0.7} 会拿到路径而非默认
GEN_TEMP="${GEN_TEMP:-0.7}"; TOP_P="${TOP_P:-0.8}"; TOP_K="${TOP_K:-20}"
ALPHA="${ALPHA:-1.0}"; W_PARSE="${W_PARSE:-0.2}"; W_DEPOT="${W_DEPOT:-0.1}"; W_COV="${W_COV:-0.1}"; W_CAP="${W_CAP:-0.6}"
SANITY="${SANITY:-0}"
VLLM_PORT="${VLLM_PORT:-8005}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.80}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-360}"
# merge + BO1 eval
DO_EVAL="${DO_EVAL:-1}"; EVAL_TP="${EVAL_TP:-4}"; EVAL_NUM_TEST="${EVAL_NUM_TEST:-1000}"
EVAL_MAXLEN="${EVAL_MAXLEN:-1024}"; EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.8}"
# GPU 动态挑选
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-22528}"   # 单卡 free ≥ 此值才算空闲
NEED_TRAIN_PROC="${NEED_TRAIN_PROC:-6}"          # 期望训练进程数 (对齐 Mask=6; <6 则 batch 不再 192, 会 WARN)

cd "$_SELF"
[ -n "$MODEL" ] || { echo "[FATAL] MODEL 为空。export MODEL=<merged SFT 目录> 后再跑。"; exit 1; }
[ -d "$MODEL" ] || { echo "[FATAL] 基座不存在: $MODEL"; exit 1; }
if [ ! -f "$DATA" ]; then
    echo "[FATAL] 数据不存在: $DATA"
    echo "  zhihan 上预生成: python build_foarl_cvrp_data_mask1000.py --out $DATA --num 1000 --seed 42 --n 20 --workers 30"
    exit 1
fi

# ── GPU 选择: 手动指定优先; 否则 zhihan 动态挑空闲卡 (1 vLLM + 训练) ──────
if [ -n "${VLLM_GPU:-}" ] && [ -n "${TRAIN_GPUS_CSV:-}" ]; then
    TRAIN_PROC="${TRAIN_PROC:-$(echo "$TRAIN_GPUS_CSV" | awk -F, '{print NF}')}"
    echo "[GPU] 手动指定: vLLM=GPU $VLLM_GPU | 训练=GPU $TRAIN_GPUS_CSV ($TRAIN_PROC 进程)"
else
    _free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v th="$GPU_MIN_FREE_MIB" '{gsub(/ /,"",$1);gsub(/ /,"",$2)} $2+0>=th {print $1}')
    _free_gpus=($_free); _nfree=${#_free_gpus[@]}
    echo "[GPU] 空闲卡(free≥${GPU_MIN_FREE_MIB}MiB): ${_free_gpus[*]:-无}  (共 $_nfree 张)"
    if [ "$_nfree" -ge $((NEED_TRAIN_PROC + 1)) ]; then
        TRAIN_PROC="$NEED_TRAIN_PROC"
    else
        TRAIN_PROC=$(( _nfree - 1 )); TRAIN_PROC=$(( TRAIN_PROC - (TRAIN_PROC % 2) ))   # 留 1 给 vLLM, 训练向下取偶
        echo "[GPU] ⚠️ 空闲卡不足 $((NEED_TRAIN_PROC+1)) 张, 降级到 1 vLLM + $TRAIN_PROC 训练"
    fi
    [ "$TRAIN_PROC" -ge 2 ] || { echo "[FATAL] 空闲卡不足 (需 ≥3: 1 vLLM + 2 训练), 当前 $_nfree 张"; exit 1; }
    _sel=("${_free_gpus[@]:0:$((TRAIN_PROC + 1))}")
    TRAIN_GPUS_CSV=$(IFS=,; echo "${_sel[*]:0:$TRAIN_PROC}")
    VLLM_GPU="${_sel[$TRAIN_PROC]}"
    echo "[GPU] 自动分配: vLLM=GPU $VLLM_GPU | 训练=GPU $TRAIN_GPUS_CSV ($TRAIN_PROC 进程)"
fi
[ "$TRAIN_PROC" = "6" ] || echo "[WARN] 训练进程=$TRAIN_PROC≠6 → 每次更新 batch = $((PDTB*TRAIN_PROC*GA)) completions ≠ Mask 的 192, 受控对比会失真!"

# 全局批可整除性
GLOBAL_BATCH=$(( PDTB * TRAIN_PROC * GA ))
[ $(( GLOBAL_BATCH % S )) -eq 0 ] || { echo "[FATAL] 全局批 $GLOBAL_BATCH 不能被 S=$S 整除"; exit 1; }
echo "[批检查] 全局生成批=$GLOBAL_BATCH completions = $(( GLOBAL_BATCH / S )) prompts/更新, S=$S → OK"

SANITY_FLAG=""; LOG_STEPS=5; RESUME_FLAG="--resume_from_checkpoint auto"
if [ "$SANITY" = "1" ]; then
    SANITY_FLAG="--max_samples 64"; LOG_STEPS=1; RESUME_FLAG=""; OUTPUT_DIR="${OUTPUT_DIR}_sanity"
    echo "[sanity] 输出改到独立目录: $OUTPUT_DIR (不续 ckpt)"
fi

LOG_DIR="$_SELF/logs"; mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/foarl_vllm_$(date +%Y%m%d_%H%M%S).log"
VLLM_PID=""
stop_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] 关闭 vLLM (pid=$VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}
trap 'stop_vllm' EXIT INT TERM

echo "############## FOARL CVRP GRPO RL (zhihan, 1 vLLM + $TRAIN_PROC 训练) ##############  $(date '+%F %T')"
echo "  MODEL=$MODEL | DATA=$DATA | OUT=$OUTPUT_DIR"
echo "  GRPO: S=$S LR=$LR BETA=$BETA EPS=[$EPS,$EPS_HIGH] PDTB=$PDTB GA=$GA EPOCHS=$EPOCHS | 采样 T=$GEN_TEMP p=$TOP_P k=$TOP_K"

echo "[$(date '+%H:%M:%S')] 启动 vLLM | GPU=$VLLM_GPU port=$VLLM_PORT (log: $VLLM_LOG)"
CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    trl vllm-serve --model "$MODEL" --tensor_parallel_size 1 --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --dtype bfloat16 --enable_prefix_caching True > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
_waited=0
while [ "$_waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[FATAL] vLLM 退出, 见 $VLLM_LOG"; tail -n 60 "$VLLM_LOG" || true; exit 1; }
    curl -s "http://localhost:${VLLM_PORT}/health/" >/dev/null 2>&1 && { echo "[$(date '+%H:%M:%S')] ✓ vLLM 就绪 (${_waited}s)"; break; }
    sleep 5; _waited=$(( _waited + 5 ))
done
[ "$_waited" -lt "$VLLM_STARTUP_TIMEOUT" ] || { echo "[FATAL] vLLM 超时, 见 $VLLM_LOG"; exit 1; }

TRAIN_LOG="$LOG_DIR/foarl_train_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%H:%M:%S')] 启动训练 ($TRAIN_PROC 卡: $TRAIN_GPUS_CSV) | log: $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS_CSV" \
accelerate launch --num_processes "$TRAIN_PROC" --main_process_port 29611 \
    train_grpo_foarl.py \
    --model "$MODEL" --data "$DATA" --output_dir "$OUTPUT_DIR" \
    --lora_rank 64 --lora_alpha 128 --num_generations "$S" \
    --lr "$LR" --beta "$BETA" --epsilon "$EPS" --epsilon_high "$EPS_HIGH" \
    --batch_size "$PDTB" --grad_accum "$GA" --epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
    --max_prompt_length 1536 --max_completion_length 1000 \
    --temperature "$GEN_TEMP" --top_p "$TOP_P" --top_k "$TOP_K" \
    --alpha "$ALPHA" --omega_parse "$W_PARSE" --omega_depot "$W_DEPOT" \
    --omega_coverage "$W_COV" --omega_capacity "$W_CAP" \
    --use_vllm --vllm_server_host localhost --vllm_server_port "$VLLM_PORT" \
    --zero_stage 3 --gradient_checkpointing --save_steps 200 --logging_steps "$LOG_STEPS" \
    $RESUME_FLAG $SANITY_FLAG 2>&1 | tee "$TRAIN_LOG"
EC=${PIPESTATUS[0]}
stop_vllm   # 训练 vLLM 关掉, 腾卡给 eval

[ "$EC" -eq 0 ] || { echo "[FATAL] 训练非零退出 (exit=$EC), 跳过 merge+eval"; exit "$EC"; }
echo "############## [1/3] ✓ 训练完成: $OUTPUT_DIR/final_model ##############  $(date '+%F %T')"
if [ "$DO_EVAL" != "1" ] || [ "$SANITY" = "1" ]; then
    echo "(DO_EVAL=$DO_EVAL SANITY=$SANITY) 跳过 merge+eval, 结束。"; exit 0
fi

ADAPTER="$OUTPUT_DIR/final_model"; MERGED="$OUTPUT_DIR/merged_model"
[ -f "$ADAPTER/adapter_config.json" ] || { echo "[FATAL] 找不到 RL adapter: $ADAPTER/adapter_config.json"; exit 1; }

echo "############## [2/3] merge RL LoRA → merged_model ##############  $(date '+%F %T')"
if [ -d "$MERGED" ] && [ -f "$MERGED/config.json" ]; then
    echo "[2/3] merged_model 已存在, 跳过: $MERGED"
else
    python ../tools/merge_lora.py --adapter "$ADAPTER" --output "$MERGED" --device cpu \
        || { echo "[FATAL] merge 失败"; exit 1; }
fi
_W=$(find "$MERGED" -maxdepth 1 -name '*.safetensors' -size +0c 2>/dev/null | head -1)
{ [ -f "$MERGED/config.json" ] && [ -n "$_W" ]; } || { echo "[FATAL] merged 校验失败 (空壳?): $MERGED"; ls -la "$MERGED" || true; exit 1; }
echo "[2/3] ✓ merge 完成且权重非空: $MERGED"

echo "############## [3/3] BO1 eval (foarl/instruct) ##############  $(date '+%F %T')"
MERGED_ABS="$(readlink -f "$MERGED")"; EVAL_SAVE="$(readlink -f "$OUTPUT_DIR")/eval_bo1"; mkdir -p "$EVAL_SAVE"
# eval 用训练腾空的卡: 取 训练卡+vLLM卡 前 EVAL_TP 张
EVAL_GPU="${EVAL_GPU:-$(echo "${TRAIN_GPUS_CSV},${VLLM_GPU}" | tr ',' '\n' | head -n "$EVAL_TP" | paste -sd, -)}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "  eval GPU=$EVAL_GPU (TP=$EVAL_TP) | 测试集 seed=9999 × $EVAL_NUM_TEST (与 Mask RL_BO1 可比)"
cd "$MASK_DIR"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python evaluate.py \
    --backend vllm --model_path "$MERGED_ABS" --tp_size "$EVAL_TP" --vllm_gpu_mem_util "$EVAL_GPU_MEM" \
    --problem cvrp --problem_size 20 --num_test "$EVAL_NUM_TEST" \
    --prompt_mode foarl --model_type instruct \
    --num_samples 1 --max_completion_length "$EVAL_MAXLEN" \
    --save_dir "$EVAL_SAVE" --run_tag foarl_RL_BO1
EVAL_EC=$?

echo "============================================================"
if [ "$EVAL_EC" -eq 0 ]; then
    echo "  ✅ FOARL train→merge→BO1 eval 全部完成  $(date '+%F %T')"
    echo "  RL 模型:  $MERGED_ABS"
    echo "  BO1 结果: $EVAL_SAVE/foarl_RL_BO1.json  (对比 Mask RL_BO1, 同 seed=9999 1000 实例)"
else
    echo "  ⚠️ eval 非零退出 (exit=$EVAL_EC)"
fi
echo "============================================================"
exit "$EVAL_EC"
