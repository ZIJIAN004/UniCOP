#!/bin/bash
# submit_grpo_foarl_cvrp.sh — FOARL CVRP Stage-2 GRPO RL (zhuoyi SLURM, 1 vLLM + 6 训练)
#   基座: Stage-1 SFT 的 **merged** 模型 (LoRA 须先 merge 回基座)
#   奖励: foarl_reward_cvrp 规则奖励 R^P = R_f(可行性) + R_o(最优性), 无 reward model/PRM
#   栈:   trl vllm-serve(1卡生成) + accelerate ZeRO-3(6卡训练) + TRL GRPOTrainer + LoRA
#   流程: [1/3] 训练 → [2/3] merge RL LoRA(tools/merge_lora.py, 校验非空) → [3/3] BO1 eval
#         (evaluate.py --prompt_mode foarl --model_type instruct, seed=9999 1000 实例,
#          与 Mask RL_BO1 同一批可比)。DO_EVAL=0 只训练; SANITY=1 也跳过 merge+eval。
#
#   ★ 受控对比 (vs UniCOP-Reason-Mask): 三处与 Mask 逐一对齐, 只留"方法"这一变量:
#     · 拓扑   = 1 vLLM + 6 训练 (qos=large, --gpus=7)
#     · batch  = per_device 4 × 6 卡 × grad_accum 8 = 192 completions / 24 prompts 每次更新
#                (与 Mask v6 完全相同; num_generations=8 一致)
#     · 数据   = Mask 同一批 1000 条实例 (seed=42, build_foarl_cvrp_data_mask1000.py 产出)
#     · 采样   = Qwen3-Instruct-2507 官方 temperature=0.7 top_p=0.8 top_k=20
#     方法差异(各自固有): FOARL=Instruct 非思维 + 规则奖励; Mask=Thinking + POMO-PRM。
#
#   ⚠️ 必填: MODEL = merged SFT 模型目录, 提交前 export MODEL=<路径>。
#   ⚠️ 数据须先预生成 (PyVRP 求参考解, 在 login 节点跑即可, 纯 CPU, 别占 GPU 作业):
#        python build_foarl_cvrp_data_mask1000.py \
#          --mask_dir ../UniCOP-Reason-Mask --distill_dir ../UniCOP-Distill \
#          --out data/foarl_cvrp20_mask1000.jsonl --num 1000 --seed 42 --n 20 --timeout 5
#   sanity: SANITY=1 MODEL=<merged> sbatch submit_grpo_foarl_cvrp.sh  (只取 64 条 + 看 reward 探针)
#   提交:   MODEL=<merged> sbatch submit_grpo_foarl_cvrp.sh

#SBATCH --qos=large
#SBATCH --gpus=7
#SBATCH --job-name=zijia_foarl_grpo_cvrp
#SBATCH --comment="zijianliu, FOARL CVRP stage2 GRPO RL, do not cancel"
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/homes/zhuoyi/zijianliu/UniCOP/FOARL/foarl_grpo_cvrp_%j.log
#SBATCH --error=/homes/zhuoyi/zijianliu/UniCOP/FOARL/foarl_grpo_cvrp_%j.err
#   注: 不加静态 --exclude, 改用 preflight_gpu.sh 动态排除被占节点 (与 Mask 同口径,
#       见 reference_zhuoyi_flaky_node)。

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ── 可覆盖参数 ─────────────────────────────────────────────────────────
# ⚠️ MODEL 留空: 必须指向 Stage-1 SFT 的 merged 模型目录, 提交前 export MODEL=<路径>
MODEL="${MODEL:-}"
DATA="${DATA:-data/foarl_cvrp20_mask1000.jsonl}"   # Mask 同 1000 实例
OUTPUT_DIR="${OUTPUT_DIR:-./output_grpo_foarl_cvrp20}"

# ── GPU 拓扑 (sbatch --gpus=7 下 SLURM 暴露逻辑卡 0..6) ──────────────────
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-6}"          # 训练进程数 (= Mask 训练卡数)
TRAIN_GPUS_CSV="${TRAIN_GPUS_CSV:-0,1,2,3,4,5}"   # 前 6 张做训练
VLLM_GPU_IDX="${VLLM_GPU_IDX:-6}"              # 第 7 张做 vLLM 生成
# 端口走低位 (< ip_local_port_range 起点): Mask 实测高端口(8006/826x)会被 vLLM init 那 ~19s
#   里的 outgoing socket(NCCL/torch.dist) 当临时端口抢占 → uvicorn LISTEN 时 address in use。
#   8005 与 Mask 的 8004 错开 (本 job 独占整节点, 本不会撞, 仅图保险)。
VLLM_PORT="${VLLM_PORT:-8005}"
# gpu_memory_utilization=0.80 是 Mask 在 4B 模型上踩出来的甜点: 0.85 → CUDA graph capture
#   阶段 OOM (capture 的 ~4.7GiB 在 util 预算之外); 0.60 → KV cache 砍半使并发 < num_gen=8
#   触发 RECOMPUTE 抢占损坏 rollout。OOM 别靠降 util 绕 (会跌破并发), 降 PDTB 或 num_gen。
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.80}"
# max_model_len 必须 ≥ max_prompt_length + max_completion_length (+overhead), 否则 vLLM 拒绝;
#   且越小并发越高。CVRP20 FOARL prompt ~<1k tok, 故 1536(prompt)+1000(completion)+overhead ≈ 3072。
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-360}"

# ── GRPO 超参 (对齐 Mask: 4×6×8 = 192 completions / 24 prompts 每次更新) ──
S="${S:-8}"                       # num_generations (官方=Mask=8)
LR="${LR:-2e-5}"                  # 用户指定 (官方默认 1e-6)
BETA="${BETA:-0.05}"              # KL 系数 (官方 0.05)
EPS="${EPS:-0.1}"                 # 裁剪下界 (官方 0.1)
EPS_HIGH="${EPS_HIGH:-0.28}"      # 裁剪上界 (官方 0.28)
PDTB="${PDTB:-4}"                 # per-device completions (对齐 Mask=4); 全局批=PDTB×卡×GA 须被 S 整除
GA="${GA:-8}"                     # 梯度累积 (官方=Mask=8)
EPOCHS="${EPOCHS:-1}"             # 官方=Mask sweep=1
MAX_STEPS="${MAX_STEPS:--1}"
# 采样 (Qwen3-Instruct-2507 官方)
# ⚠️ 不能叫 TEMP: 系统/conda 常把 TEMP 设成临时目录, ${TEMP:-0.7} 会拿到该路径而非默认值
GEN_TEMP="${GEN_TEMP:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
# FOARL 奖励权重 (官方 rewards.py CVRP weights, 论文附录 A.3.3 同值)
ALPHA="${ALPHA:-1.0}"             # 最优性权重 (官方 1/(1+gap) → α=1.0)
W_PARSE="${W_PARSE:-0.2}"         # 官方 parse=0.2
W_DEPOT="${W_DEPOT:-0.1}"         # 官方 depot=0.1
W_COV="${W_COV:-0.1}"             # 官方 coverage=0.1
W_CAP="${W_CAP:-0.6}"             # 官方 capacity=0.6
SANITY="${SANITY:-0}"             # 1 = 只取 64 条做 sanity
# ── 训练后 merge + BO1 eval (镜像 Mask submit_grpo_cvrp20_v6_eval.sh; 训练成功才跑) ──
DO_EVAL="${DO_EVAL:-1}"           # 1 = 训练后自动 merge + BO1 eval; 0 = 只训练
EVAL_GPU="${EVAL_GPU:-0}"         # eval 单卡即可 (tp=1); 训练完腾空的任一卡
# ⚠️ EVAL_TP 默认=1 (原=4 致崩): vLLM 0.7.3 无 Qwen3 原生实现→回退 Transformers backend,
#    该 backend 在 tensor_parallel>1 多 worker 下生成期崩 (Worker died exit -15, issue
#    #17630/#39774)。单卡 tp=1 不起 worker, 整类问题消失; 4B 单卡 KV cache 足够, 贪心结果不受 tp 影响。
#    Mask v6_eval 用 tp=4 是因为经 run_eval_matrix 包装, 此处直调 evaluate.py 无那层防护, 故回退默认 tp=1。
EVAL_TP="${EVAL_TP:-1}"
EVAL_NUM_TEST="${EVAL_NUM_TEST:-1000}"  # seed=9999 冻结测试集; 与 Mask RL_BO1 同一批 → 可比, 勿改小
EVAL_MAXLEN="${EVAL_MAXLEN:-1024}"      # FOARL 答案短(~百 tok); 1024 足够 (instruct 默认仅 512)
EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.8}"

# ⚠️ 先 conda activate 再 set -u (cuda-nvcc_activate.sh 引用未设的 NVCC_PREPEND_FLAGS,
#    nounset 下会 unbound variable 挂掉)。不能靠 source ~/.bashrc 激活: 非交互 sbatch shell
#    下 .bashrc 提前 return, conda hook 不生效 → conda/accelerate/trl 全 command not found。
#    直接 source miniforge 的 conda.sh (zhuoyi 是 miniforge3, 见 address.md/paths.sh)。
__CONDA_SH="/homes/zhuoyi/miniforge3/etc/profile.d/conda.sh"
[ -f "$__CONDA_SH" ] || { echo "[FATAL] 找不到 conda.sh: $__CONDA_SH"; exit 1; }
source "$__CONDA_SH"
conda activate /homes/zhuoyi/miniforge3/envs/unicop

# ── DeepSpeed CUDA 扩展(CPUAdam, ZeRO-3 offload_optimizer 用)JIT 编译需 CUDA_HOME + dev 头 ──
#   缺失会在主模型 deepspeed.initialize 阶段报 "cuda_runtime.h: No such file or directory"。
#   复刻 paths.sh:142-155 的 zhuoyi 配置: conda env 把头/库放在 targets/x86_64-linux/{include,lib},
#   torch 默认只看 $CUDA_HOME/include → 须把真目录加进 CPATH/LIBRARY_PATH/LD_LIBRARY_PATH。
#   (Mask run_v5 经 source paths.sh 拿到这些; 本 submit 自包含故内联, 与 paths.sh 同值。)
export CUDA_HOME="/homes/zhuoyi/miniforge3/envs/unicop"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
_CUDA_TARGETS="$CUDA_HOME/targets/x86_64-linux"
if [ -d "$_CUDA_TARGETS/include" ]; then
    export CPATH="$_CUDA_TARGETS/include:${CPATH:-}"
    export LIBRARY_PATH="$_CUDA_TARGETS/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$_CUDA_TARGETS/lib:$LD_LIBRARY_PATH"
fi

command -v accelerate >/dev/null 2>&1 || { echo "[FATAL] accelerate 不在 PATH, unicop 未激活"; exit 1; }
command -v trl >/dev/null 2>&1 || { echo "[FATAL] trl CLI 不在 PATH (vllm-serve 需要); 确认 trl 已装"; exit 1; }
cd /homes/zhuoyi/zijianliu/UniCOP/FOARL
set -uo pipefail

# zhuoyi 多卡须禁 P2P/SHM 否则 ZeRO-3 init hang (reference_zhuoyi_nccl_topology)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

echo "############## FOARL CVRP GRPO RL (1 vLLM + ${NUM_TRAIN_GPUS} 训练) ##############  $(date '+%F %T')"
echo "  MODEL=$MODEL"
echo "  DATA=$DATA | OUT=$OUTPUT_DIR"
echo "  拓扑: vLLM=GPU $VLLM_GPU_IDX(port $VLLM_PORT) | 训练=GPU $TRAIN_GPUS_CSV ($NUM_TRAIN_GPUS 进程)"
echo "  GRPO: S=$S LR=$LR BETA=$BETA EPS=[$EPS,$EPS_HIGH] PDTB=$PDTB GA=$GA EPOCHS=$EPOCHS"
echo "  采样: T=$GEN_TEMP top_p=$TOP_P top_k=$TOP_K | 奖励: ALPHA=$ALPHA W(p=$W_PARSE d=$W_DEPOT cov=$W_COV cap=$W_CAP) | SANITY=$SANITY"

# ── 前置检查 ──────────────────────────────────────────────────────────
if [ -z "$MODEL" ]; then
    echo "[FATAL] MODEL 为空。export MODEL=<merged SFT 模型目录> 后重投 (SFT 产物是 adapter, 须先 merge)。"
    exit 1
fi
[ -d "$MODEL" ] || { echo "[FATAL] 基座不存在: $MODEL"; exit 1; }
if [ ! -f "$DATA" ]; then
    echo "[FATAL] 数据不存在: $DATA"
    echo "  先在 login 节点(纯 CPU, 别占 GPU)预生成 Mask 同 1000 实例数据:"
    echo "    python build_foarl_cvrp_data_mask1000.py \\"
    echo "      --mask_dir ../UniCOP-Reason-Mask --distill_dir ../UniCOP-Distill \\"
    echo "      --out $DATA --num 1000 --seed 42 --n 20 --timeout 5"
    exit 1
fi

# 全局批可整除性自检 (TRL 硬约束: 全局生成批 % num_generations == 0)
GLOBAL_BATCH=$(( PDTB * NUM_TRAIN_GPUS * GA ))
if [ $(( GLOBAL_BATCH % S )) -ne 0 ]; then
    echo "[FATAL] 全局批 $GLOBAL_BATCH (=PDTB$PDTB×卡$NUM_TRAIN_GPUS×GA$GA) 不能被 S=$S 整除, 调参后重投。"
    exit 1
fi
echo "[批检查] 全局生成批=$GLOBAL_BATCH completions = $(( GLOBAL_BATCH / S )) prompts/更新, S=$S → OK"

SANITY_FLAG=""
LOG_STEPS=5
RESUME_FLAG="--resume_from_checkpoint auto"
if [ "$SANITY" = "1" ]; then
    SANITY_FLAG="--max_samples 64"
    LOG_STEPS=1
    RESUME_FLAG=""
    OUTPUT_DIR="${OUTPUT_DIR}_sanity"
    echo "[sanity] 输出改到独立目录: $OUTPUT_DIR (不续 ckpt)"
fi

# ── GPU 占用预检: 分到的卡若被别人占着 → 排除本节点重投, 本 job 退出 (与 Mask 同口径) ──
#    起 vLLM/CUDA 之前查一遍, 避免落到 flaky/被占节点上 vLLM CUDA graph capture OOM。
export SUBMIT_SCRIPT="$(pwd)/submit_grpo_foarl_cvrp.sh"
export BASE_EXCLUDE=""
source "$(pwd)/preflight_gpu.sh"
preflight_gpu_or_resubmit

# ── vLLM server 生命周期 ───────────────────────────────────────────────
VLLM_LOG="/homes/zhuoyi/zijianliu/UniCOP/FOARL/foarl_vllm_${SLURM_JOB_ID:-manual}.log"
VLLM_PID=""

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] 关闭 vLLM (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}
trap 'stop_vllm' EXIT INT TERM

echo "[$(date '+%H:%M:%S')] 启动 vLLM server | GPU=$VLLM_GPU_IDX port=$VLLM_PORT (log: $VLLM_LOG)"
# flag 名沿用 Mask 实测可用的那套 (其 vllm_serve_logprobs.py 即转发给 trl vllm-serve):
#   --tensor_parallel_size / --gpu_memory_utilization / --max_model_len / --dtype / --enable_prefix_caching
CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDX" \
    trl vllm-serve --model "$MODEL" \
        --tensor_parallel_size 1 \
        --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --dtype bfloat16 \
        --enable_prefix_caching True \
        > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# 健康等待: 进程存活 + /health/ 就绪
_waited=0
while [ "$_waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[FATAL] vLLM 进程退出, 见 $VLLM_LOG"; tail -n 60 "$VLLM_LOG" || true; exit 1
    fi
    if curl -s "http://localhost:${VLLM_PORT}/health/" >/dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] ✓ vLLM 就绪 (用时 ${_waited}s)"; break
    fi
    sleep 5; _waited=$(( _waited + 5 ))
done
if [ "$_waited" -ge "$VLLM_STARTUP_TIMEOUT" ]; then
    echo "[FATAL] vLLM 启动超时 ${VLLM_STARTUP_TIMEOUT}s, 见 $VLLM_LOG"; tail -n 60 "$VLLM_LOG" || true; exit 1
fi

# ── 训练 (6 卡, 连 vLLM server 生成) ────────────────────────────────────
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS_CSV" \
accelerate launch --num_processes "$NUM_TRAIN_GPUS" --main_process_port 29611 \
    train_grpo_foarl.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank 64 --lora_alpha 128 \
    --num_generations "$S" \
    --lr "$LR" --beta "$BETA" --epsilon "$EPS" --epsilon_high "$EPS_HIGH" \
    --batch_size "$PDTB" --grad_accum "$GA" \
    --epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
    --max_prompt_length 1536 --max_completion_length 1000 \
    --temperature "$GEN_TEMP" --top_p "$TOP_P" --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --omega_parse "$W_PARSE" --omega_depot "$W_DEPOT" \
    --omega_coverage "$W_COV" --omega_capacity "$W_CAP" \
    --use_vllm --vllm_server_host localhost --vllm_server_port "$VLLM_PORT" \
    --zero_stage 3 --gradient_checkpointing \
    --save_steps 200 --logging_steps "$LOG_STEPS" \
    $RESUME_FLAG \
    $SANITY_FLAG
EC=$?
stop_vllm   # 训练用的 trl vllm-serve 关掉, 腾出 7 卡给 eval

# ====================================================================
if [ "$EC" -ne 0 ]; then
    echo "[FATAL] 训练非零退出 (exit=$EC), 跳过 merge + eval"
    exit "$EC"
fi
echo "############## [1/3] ✓ 训练完成: $OUTPUT_DIR/final_model ##############  $(date '+%F %T')"

if [ "$DO_EVAL" != "1" ] || [ "$SANITY" = "1" ]; then
    echo "(DO_EVAL=$DO_EVAL SANITY=$SANITY) 跳过 merge+eval, 结束。"
    exit 0
fi

ADAPTER="$OUTPUT_DIR/final_model"
MERGED="$OUTPUT_DIR/merged_model"
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "[FATAL] 找不到 RL LoRA adapter: $ADAPTER/adapter_config.json"; exit 1
fi

# ── [2/3] merge RL LoRA → merged_model (团队 canonical 脚本, 含 vocab resize) ──
echo "############## [2/3] merge RL LoRA → merged_model ##############  $(date '+%F %T')"
if [ -d "$MERGED" ] && [ -f "$MERGED/config.json" ]; then
    echo "[2/3] merged_model 已存在, 跳过: $MERGED"
else
    python ../tools/merge_lora.py --adapter "$ADAPTER" --output "$MERGED" --device cpu \
        || { echo "[FATAL] merge 失败"; exit 1; }
fi
# 校验非空权重 (防 ZeRO-3+LoRA 空壳, 见 CLAUDE.md 代码自审)
_W=$(find "$MERGED" -maxdepth 1 -name '*.safetensors' -size +0c 2>/dev/null | head -1)
if [ ! -f "$MERGED/config.json" ] || [ -z "$_W" ]; then
    echo "[FATAL] merged_model 校验失败 (缺 config.json 或权重为空): $MERGED"; ls -la "$MERGED" || true; exit 1
fi
echo "[2/3] ✓ merge 完成且权重非空: $MERGED"

# ── [3/3] BO1 eval (FOARL 臂: prompt_mode=foarl, model_type=instruct, num_samples=1) ──
#    复用 Mask 的 evaluate.py: seed=9999 的 1000 实例冻结测试集, 与 Mask RL_BO1 同一批 → 可比。
#    evaluate.py 在 Mask 目录 (import 它的 config/problems), 故 cd 过去跑; 用绝对路径喂模型/存档。
echo "############## [3/3] BO1 eval ##############  $(date '+%F %T')"
MERGED_ABS="$(readlink -f "$MERGED")"
EVAL_SAVE="$(readlink -f "$OUTPUT_DIR")/eval_bo1"
mkdir -p "$EVAL_SAVE"
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # tp>1 vLLM worker 必须 spawn (evaluate.py 已 init CUDA)
cd /homes/zhuoyi/zijianliu/UniCOP/UniCOP-Reason-Mask
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python evaluate.py \
    --backend vllm --model_path "$MERGED_ABS" --tp_size "$EVAL_TP" \
    --vllm_gpu_mem_util "$EVAL_GPU_MEM" \
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
    echo "  ⚠️ eval 非零退出 (exit=$EVAL_EC), 见日志"
fi
echo "============================================================"
exit "$EVAL_EC"
