#!/bin/bash
# run_grpo_cvrp20_v5.sh — GRPO + POMO PRM · CVRP n=20 · reward_scheme=v5
#   GPU: 1 vLLM + 6 训练 (默认动态挑卡, submit/env 可覆盖); 实测吞吐随卡数提升 (per-completion 6卡<4卡<2卡)
#   · v4 + hardgate distance + cov/cons 加权 A_feas
#
# v5 设计 (修 v4 7414 run 信号弱 + 冷启动):
#   - A_feas 加回 parse + cov + cons(hardgate cov_gate_v5=1.0) + format
#     cov+cons 占权重 82% (w_p=0.5, w_cov=2.5, w_cons=2.0, w_f=0.5)
#   - A_outcome 用 raw prob.get_tour_distance (不 repair) on strict
#     fully_feasible 子集 (parse + cov=1 + cons=1 + format=1, 子集 >=2 才启用)
#   - 前期 fully_feasible<2 时 A_outcome=0, 完全靠 A_feas + PRM 机会成本推可行性
#   - PRM 复用 v4: absolute base + tanh(R_step), 违例/重复及之后 step 游离
#   - 配 LR 2e-5 + warmup_ratio 0.01 (5 step) 加快收敛 (v4 7414 run grad_norm=0.05
#     远低于 clip 阈值, LR 信号空间充足)
#
# 输出目录: output_v5 (跟 output_v3/v4/mask 隔离)
# 端口 8004 错开 hardgate (8001) / v4 (8002) / mask (8002) / 6gpu (8000)
#
# SBATCH 提交:
#   sbatch submit_grpo_cvrp20_v5.sh
# 手动:
#   bash run_grpo_cvrp20_v5.sh

set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
# BASE_MODEL_TYPE: 选哪条 SFT 产物作为 RL 起点 (不是加载原始基座!):
#   qwen3_thinking (默认) → output_sft_qwen3_template_cvrp20/final_model          (Qwen3-4B-Thinking SFT)
#   qwen3_instruct        → output_sft_qwen3_instruct_template_cvrp20/final_model (Qwen3-4B-Instruct SFT, 对齐 FOARL instruct 范式)
#   r1_distill            → output_sft_hybrid_cvrp20/final_model                  (DeepSeek-R1-7B SFT)
# paths.sh 据此设采样参数 GEN_TEMPERATURE/TOP_P/TOP_K, trainer 自动读 env
export BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-qwen3_thinking}"
source "$(dirname "$_SELF_DIR")/paths.sh"

WORK_DIR="$MASK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/grpo_cvrp20_v5_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "日志文件: $LOG_FILE"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# NCCL 传输默认按主机分流 (HOST_ID 来自 paths.sh):
#   zhihan (astar-zhihan, 单机) → 默认开 P2P/SHM (=0): 集合通信走卡间 P2P/共享内存而非禁用回退, 比禁用快;
#   zhuoyi 等其它 → 默认禁 (=1), 否则 init hang 30min (reference_zhuoyi_nccl_topology)。
# 仍可用 env 覆盖: NCCL_P2P_DISABLE=1 bash ... (zhihan 万一 init hang 时退回)。
if [ "${HOST_ID:-}" = "astar-zhihan" ]; then
    _NCCL_DEFAULT=0
else
    _NCCL_DEFAULT=1
fi
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-$_NCCL_DEFAULT}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-$_NCCL_DEFAULT}"
echo "[NCCL] HOST_ID=${HOST_ID:-?}  P2P_DISABLE=$NCCL_P2P_DISABLE  SHM_DISABLE=$NCCL_SHM_DISABLE  (0=开/快, 1=禁/稳)"

# ── Liger Kernel 提速 (RMSNorm/RoPE/SwiGLU), 默认开启 ───────────────────
# 兼容性已验证: tests/test_liger_compat.py 4/4 PASS (含 LoRA×SwiGLU 反传)。
# 只融合模型 forward 层算子 (~10-20% 吞吐), 与自定义 GRPO loss 解耦; train.py 里
# 已强制关 cross_entropy/FLCE。单次禁用: USE_LIGER=0 bash run_grpo_cvrp20_v5.sh
export USE_LIGER="${USE_LIGER:-1}"
export LIGER_SWIGLU="${LIGER_SWIGLU:-1}"

# ── 梯度重计算开关 (A/B 测速用)。默认开; GRAD_CKPT=0 关掉 (省 backward 重算, 吃显存) ──
GRAD_CKPT="${GRAD_CKPT:-1}"
if [ "$GRAD_CKPT" = "1" ]; then
    GC_FLAG="--gradient_checkpointing"
else
    GC_FLAG=""
    echo "[GC] gradient_checkpointing 已关闭 (GRAD_CKPT=0)"
fi

# ── 配置 ──────────────────────────────────────────────────────────────
PROBLEM="cvrp"
SIZE=20

case "$BASE_MODEL_TYPE" in
    r1_distill)     MODEL_BASE="$DISTILL_DIR/output_sft_hybrid_cvrp20/final_model" ;;
    qwen3_thinking) MODEL_BASE="$DISTILL_DIR/output_sft_qwen3_template_cvrp20/final_model" ;;
    qwen3_instruct) MODEL_BASE="$DISTILL_DIR/output_sft_qwen3_instruct_template_cvrp20/final_model" ;;
    *) echo "❌ 未知 BASE_MODEL_TYPE='$BASE_MODEL_TYPE'"; exit 1 ;;
esac
if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ 基座模型不存在: $MODEL_BASE"
    exit 1
fi
echo "[RL 起点] SFT 产物 (非原始基座): $MODEL_BASE"
echo "[BASE_MODEL_TYPE=$BASE_MODEL_TYPE] qwen3_thinking→Qwen3-4B-Thinking SFT | qwen3_instruct→Qwen3-4B-Instruct SFT | r1_distill→R1-7B SFT"

# ── GPU 选择 ──────────────────────────────────────────────────────────
# zhihan 默认动态挑卡: nvidia-smi 扫所有卡, 用 free 显存 ≥ GPU_MIN_FREE_MIB 的空闲卡,
# 不再写死 GPU 索引——某张卡被别人占了也能自动避开, 不会整个跑不起来。
#   1 张做 vLLM + 其余做训练(训练进程数取 ≤可用-1 的最大偶数, 满足 4×n%8==0, 上限 NEED_TRAIN_PROC)。
# 手动覆盖: 显式 export VLLM_GPU + TRAIN_GPUS_CSV (+TRAIN_PROC) 即跳过自动挑卡。
# 其他主机(zhuoyi 在 sbatch 下 SLURM 已隔离卡)维持原固定分配。
# 注: GPU_MIN_FREE_MIB=22528(22G, used≤2048) 比 check_gpu_idle 的 used≤2000 略松,
#     但真正的空闲卡 used 只有几~几百 MiB, 两个判据都轻松通过。
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-22528}"   # 22G; 单卡 free ≥ 此值才算空闲
NEED_TRAIN_PROC="${NEED_TRAIN_PROC:-6}"          # 期望训练进程数(须偶数)

if [ -n "${VLLM_GPU:-}" ] && [ -n "${TRAIN_GPUS_CSV:-}" ]; then
    TRAIN_PROC="${TRAIN_PROC:-$(echo "$TRAIN_GPUS_CSV" | awk -F, '{print NF}')}"
    echo "[GPU] 手动指定: vLLM=GPU $VLLM_GPU | 训练=GPU $TRAIN_GPUS_CSV ($TRAIN_PROC 进程)"
elif [ "$HOST_ID" = "astar-zhihan" ]; then
    _free_list=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v th="$GPU_MIN_FREE_MIB" '{gsub(/ /,"",$1);gsub(/ /,"",$2)} $2+0>=th {print $1}') || true
    _free_gpus=($_free_list)
    _nfree=${#_free_gpus[@]}
    echo "[GPU] 空闲卡(free≥${GPU_MIN_FREE_MIB}MiB): ${_free_gpus[*]:-无}  (共 $_nfree 张)"
    if [ "$_nfree" -ge $((NEED_TRAIN_PROC + 1)) ]; then
        TRAIN_PROC="$NEED_TRAIN_PROC"
    else
        _avail=$((_nfree - 1))                       # 留 1 张给 vLLM
        TRAIN_PROC=$(( _avail - (_avail % 2) ))      # 训练进程向下取偶数
        echo "[GPU] ⚠️ 空闲卡不足 $((NEED_TRAIN_PROC + 1)) 张, 降级到 1 vLLM + $TRAIN_PROC 训练"
    fi
    if [ "$TRAIN_PROC" -lt 2 ]; then
        echo "[FATAL] 空闲卡不足(需至少 3 张: 1 vLLM + 2 训练), 当前仅 $_nfree 张 free≥${GPU_MIN_FREE_MIB}MiB"
        nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true
        exit 1
    fi
    _sel=("${_free_gpus[@]:0:$((TRAIN_PROC + 1))}")
    TRAIN_GPUS_CSV=$(IFS=,; echo "${_sel[*]:0:$TRAIN_PROC}")   # 前 TRAIN_PROC 张做训练
    VLLM_GPU="${_sel[$TRAIN_PROC]}"                            # 第 TRAIN_PROC+1 张做 vLLM
    echo "[GPU] 自动分配: vLLM=GPU $VLLM_GPU | 训练=GPU $TRAIN_GPUS_CSV ($TRAIN_PROC 进程)"
else
    VLLM_GPU=6
    TRAIN_GPUS_CSV="0,1,2,3,4,5"
    TRAIN_PROC=6
fi

# ZeRO stage 可 env 覆盖。当前 CVRP20 配置(max_completion≤3584, B=4, 24G 卡)ZeRO-2 放得下(不分片参数,
# 每卡常驻基座 8.3GB)。⚠️ 早期 6144 completion 下 ZeRO-2 会 OOM, 现已随 max_completion 6144→3584 缓解;
# ZeRO-3 留给更大规模(更长序列/更大 batch/更省显存)分片基座参数。
# DS_OFFLOAD=0(见 train.py make_deepspeed_config): 去掉 CPU offload, 优化器/参数留 GPU 省 PCIe 搬运;
# 实测提速有限(非早期预估的 ~50%)——bwd 通信(无快速互联下的梯度同步)才是单 step 大头(~88%)。
ZERO_STAGE="${ZERO_STAGE:-3}"
NUM_TRAIN="${NUM_TRAIN:-1000}"   # 一个 epoch 的 instance 数, 可 env 覆盖。total_steps = NUM_TRAIN×epochs(2)÷(4×训练卡数); 例 1000×2÷(4×6)=83 (6 卡)
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$WORK_DIR/output_v5}"   # 可 env 覆盖, 消融实验用独立目录避免与主实验打架

VLLM_PORT="${VLLM_PORT:-8004}"   # 可 env 覆盖, 多 scheme 并行时错开端口 (v6 用 8006)

# REWARD_SCHEME: env 覆盖 reward_scheme (不设则走 config 默认 v5)。v6 wrapper 用它切 v6。
REWARD_SCHEME_FLAG=""
if [ -n "${REWARD_SCHEME:-}" ]; then
    REWARD_SCHEME_FLAG="--reward_scheme ${REWARD_SCHEME}"
    echo "[REWARD_SCHEME] 覆盖为 ${REWARD_SCHEME}"
fi
# gpu_memory_utilization = 0.80 是 zhihan 24G 卡 + 这个 4B 模型的唯一甜点值。两边都是悬崖:
#   ▲ 太高(≥0.85): CUDA graph capture 要 ~4.68GiB, 且这块显存在 util 预算之外
#     (capture 之后才发生)。可用 = 23.69×(1-util): 0.85→3.55GiB < 4.68 → capture_end OOM;
#     0.80→4.74GiB ≥ 4.68 → 仅 0.06GiB 余量但能过。所以 0.80 是能开 capture 的上限。
#   ▼ 太低(≤0.6): KV cache 被砍半 → Maximum concurrency 4.66x < num_generations=8,
#     一个 prompt 的 8 条采样装不下 → 疯狂 RECOMPUTE 抢占(实测 51 次 vs 0.80 时 1 次),
#     prefix caching + 重算损坏约一半 rollout → parse/coverage 从 1.0 腰斩到 0.5
#     (2026-05-26 实测对比 211256@0.80 vs 124129@0.60)。
#   ✓ 0.80: KV 9.98GiB → concurrency 8.87x ≥ 8, 不抢占, parse~1.0; capture 也刚好放得下。
# OOM 别再靠降 util 绕(会跌破 concurrency<8); concurrency 不够就降 num_generations 或开 enforce_eager。
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.80}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-5248}"   # ≥ max_prompt(1280)+max_completion(3584)+overhead; 8192→5248 同步收紧, 顺带提 vLLM 并发
VLLM_DTYPE=bfloat16
VLLM_STARTUP_TIMEOUT=300

# 单卡已用显存超过此值(MiB)即视为"非空闲",拒绝启动。
# 空闲卡通常只占几 MiB~几百 MiB；任何残留模型进程都是 GB 级,2000 足够区分。
GPU_MEM_USED_THRESHOLD_MIB="${GPU_MEM_USED_THRESHOLD_MIB:-2000}"

SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
notify() {
    local title="${1:0:100}"
    local desp="${2:-}"
    curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
        --data-urlencode "title=$title" \
        --data-urlencode "desp=${desp:0:500}" > /dev/null 2>&1 || true
}

VLLM_LOG="$LOG_DIR/vllm_${PROBLEM}${SIZE}_v5_$(date +%Y%m%d_%H%M%S).log"
VLLM_PID=""

start_vllm_server() {
    echo "[$(date '+%H:%M:%S')] 启动 vLLM server | GPU=$VLLM_GPU | port=$VLLM_PORT"
    PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    CUDA_HOME="$CUDA_HOME" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
        python "$WORK_DIR/utils/vllm_serve_logprobs.py" \
        --model "$MODEL_BASE" \
        --tensor_parallel_size 1 \
        --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --dtype "$VLLM_DTYPE" \
        --enable_prefix_caching True \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    local waited=0
    while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] ✗ vLLM server 启动失败,详见 $VLLM_LOG"
            tail -n 50 "$VLLM_LOG" || true
            return 1
        fi
        if curl -s "http://localhost:${VLLM_PORT}/health/" > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] ✓ vLLM server 就绪 (pid=$VLLM_PID, 用时 ${waited}s)"
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
    done
    echo "[$(date '+%H:%M:%S')] ✗ vLLM server 启动超时 (${VLLM_STARTUP_TIMEOUT}s)"
    kill "$VLLM_PID" 2>/dev/null || true
    return 1
}

stop_vllm_server() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] 关闭 vLLM server (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}

# ── GPU 空闲检查 (防止往被占用的卡上启 vLLM 导致 CUDA graph capture OOM) ──
# 根因: vLLM gpu_memory_utilization 是"按总显存×util"的全局上限,不看当前空闲量。
#       若目标卡已被别的进程(别人的 job / 上次 crash 残留)占用,vLLM 仍按 0.85×总显存
#       预算 KV cache,等到 CUDA graph capture 要额外 scratch 时物理空闲不够 → capture_end OOM。
# 在启 vLLM 前把"vLLM 卡 + 训练卡"全查一遍,非空闲直接 fail-fast。
check_gpu_idle() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] 找不到 nvidia-smi,跳过 GPU 空闲检查"
        return 0
    fi

    local smi_out
    smi_out="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -z "$smi_out" ]; then
        echo "[WARN] nvidia-smi 查询为空,跳过 GPU 空闲检查"
        return 0
    fi

    local gpus_to_check
    gpus_to_check="$(echo "${VLLM_GPU},${TRAIN_GPUS_CSV}" | tr ',' ' ')"
    echo "[$(date '+%H:%M:%S')] GPU 空闲检查 (阈值 ${GPU_MEM_USED_THRESHOLD_MIB} MiB) | 待用: GPU ${VLLM_GPU}(vLLM) + ${TRAIN_GPUS_CSV}(训练)"

    local busy=0 g used total
    for g in $gpus_to_check; do
        used="$(echo "$smi_out"  | awk -F',' -v idx="$g" '{gsub(/ /,"",$1);gsub(/ /,"",$2)} $1==idx {print $2}')"
        total="$(echo "$smi_out" | awk -F',' -v idx="$g" '{gsub(/ /,"",$1);gsub(/ /,"",$3)} $1==idx {print $3}')"
        if [ -z "$used" ]; then
            echo "  [FAIL] GPU $g 在 nvidia-smi 中不存在 (可见 GPU 数不足?)"
            busy=$((busy+1))
        elif [ "$used" -gt "$GPU_MEM_USED_THRESHOLD_MIB" ]; then
            echo "  [FAIL] GPU $g 非空闲: 已用 ${used} / ${total} MiB"
            busy=$((busy+1))
        else
            echo "  [OK  ] GPU $g 空闲: 已用 ${used} / ${total} MiB"
        fi
    done

    if [ "$busy" -ne 0 ]; then
        echo ""
        echo "[FATAL] $busy 张待用 GPU 非空闲,拒绝启动 (否则 vLLM 大概率在 CUDA graph capture 阶段 OOM)"
        echo "  当前 GPU 占用进程:"
        nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv 2>/dev/null | sed 's/^/    /' || true
        echo "  排查:"
        echo "    1) 上次 crash 残留僵尸进程? 清理: pkill -u \"\$USER\" -f vllm_serve_logprobs.py ; pkill -u \"\$USER\" -f accelerate.commands.launch"
        echo "    2) 手动跑落到别人占用的物理卡? 改用 sbatch submit_grpo_cvrp20_v5.sh,或指定空闲卡:"
        echo "       VLLM_GPU=<空闲> TRAIN_GPUS_CSV=<空闲列表> TRAIN_PROC=<卡数> bash $0"
        echo "    3) 确认是误判(其他卡有合法占用但不影响)? 临时放宽: GPU_MEM_USED_THRESHOLD_MIB=<更大值> bash $0"
        notify "❌ CVRP20 GRPO v5 GPU 非空闲,未启动" \
"有 $busy 张待用 GPU 已被占用 (阈值 ${GPU_MEM_USED_THRESHOLD_MIB} MiB)
时间: $(date '+%Y-%m-%d %H:%M:%S')
详见日志: $LOG_FILE"
        return 1
    fi
    echo "[$(date '+%H:%M:%S')] ✓ 所有待用 GPU 空闲"
    return 0
}

TRAINING_COMPLETED=0
on_exit() {
    local exit_code=$?
    stop_vllm_server
    if [ "$TRAINING_COMPLETED" != "1" ] && [ "$exit_code" != "0" ]; then
        notify "❌ CVRP20 GRPO v5 异常退出" \
"退出码: $exit_code
时间: $(date '+%Y-%m-%d %H:%M:%S')
日志末尾:
$(tail -n 20 "$LOG_FILE" 2>/dev/null || echo '(无日志)')"
    fi
}
trap 'on_exit' EXIT INT TERM

cd "$WORK_DIR"

echo "============================================================"
echo "  GPU 拓扑"
echo "============================================================"
nvidia-smi topo -m 2>&1 || echo "(nvidia-smi topo unavailable)"
echo ""
_SCHEME="${REWARD_SCHEME:-v5}"   # 实际生效的 reward_scheme (v6 wrapper 经 env 设 v6); 不再硬编码 v5
echo "============================================================"
echo "  GRPO + POMO PRM · CVRP n=$SIZE · 1 vLLM + $TRAIN_PROC 训练 · reward_scheme=${_SCHEME}"
echo "  BASE_MODEL_TYPE: $BASE_MODEL_TYPE  (T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K)"
echo "  RL 起点:   $MODEL_BASE (SFT 产物, 非原始基座)"
echo "  GPU:       1 vLLM (GPU $VLLM_GPU) + $TRAIN_PROC 训练 (GPU $TRAIN_GPUS_CSV)"
echo "  ZeRO:      stage $ZERO_STAGE | gradient_checkpointing on"
echo "  Reward:    ${_SCHEME}  (A_out=A_feas+A_outcome; v6 复用 v5 的 A_out, 只改 PRM per-step 变换)"
echo "             A_feas/A_outcome 实际权重以 train.py 的 'Reward scheme:' 打印 + config 为准 (此 banner 不再硬编码)"
if [ "$_SCHEME" = "v6" ]; then
echo "             PRM(v6) = 批级截尾标准化 + sigmoid((R-mu)/s) ∈(0,1), proc_alpha_v6=${PROC_ALPHA_V6:-1000}"
else
echo "             PRM(${_SCHEME}) = absolute base 1.5 + tanh(R_step)"
fi
echo "  LR:        2e-5 (v4 加倍, 配 warmup 5 step 快收敛)"
echo "  Warmup:    0.01 × 500 step = 5 step"
echo "  输出目录:  $OUTPUT_DIR_BASE"
_PDB="${PER_DEVICE_BATCH:-4}"   # 跟 train.py 的 PER_DEVICE_BATCH 覆盖一致
_NUM_GEN="${NUM_GEN:-8}"        # 跟 train.py 的 NUM_GEN 覆盖一致 (固定 8: 降到 4 信号太差, 不可用于正式训练)
echo "  整除检查:  per_device_batch ($_PDB) × num_gpus ($TRAIN_PROC) = $(( _PDB * TRAIN_PROC )),  整除 num_generations ($_NUM_GEN) ? $(( (_PDB * TRAIN_PROC) % _NUM_GEN == 0 ))"
echo "  时间:      $(date)"
echo "============================================================"

if [ $(( _PDB * TRAIN_PROC % _NUM_GEN )) -ne 0 ]; then
    echo "[FATAL] 整除失败: per_device_batch ($_PDB) × num_gpus ($TRAIN_PROC) = $(( _PDB * TRAIN_PROC )) 必须整除 num_generations=$_NUM_GEN (调 PER_DEVICE_BATCH 或 TRAIN_PROC 或 NUM_GEN)"
    exit 1
fi

if ! check_gpu_idle; then
    echo "[FATAL] GPU 空闲检查未通过,中止 (避免 vLLM CUDA graph capture OOM)"
    exit 1
fi

notify "🚀 CVRP20 GRPO ${_SCHEME} 启动" \
"reward_scheme: ${_SCHEME}
LR 2e-5, warmup 5 step
基座: $MODEL_BASE
GPU: 1 vLLM + $TRAIN_PROC 训练
开始: $(date '+%Y-%m-%d %H:%M:%S')"

if ! start_vllm_server; then
    echo "[FATAL] vLLM server 启动失败"
    exit 1
fi

TRAIN_LOG="$LOG_DIR/train_${PROBLEM}${SIZE}_v5_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%H:%M:%S')] 启动训练 ($TRAIN_PROC 卡: GPU=$TRAIN_GPUS_CSV)"
echo "  log: $TRAIN_LOG"

PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_HOME="$CUDA_HOME" \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS_CSV" \
    python -m accelerate.commands.launch --num_processes "$TRAIN_PROC" "$WORK_DIR/train.py" \
    --problem "$PROBLEM" \
    --problem_size "$SIZE" \
    $REWARD_SCHEME_FLAG \
    --num_train "$NUM_TRAIN" \
    --model "$MODEL_BASE" \
    --num_gpus "$TRAIN_PROC" \
    --zero_stage "$ZERO_STAGE" \
    $GC_FLAG \
    --output_dir "$OUTPUT_DIR_BASE" \
    --pomo_ckpt_dir "$POMO_CKPT_DIR" \
    --pomo_baseline_dir "$POMO_BASELINE_DIR" \
    --pipd_ckpt_dir "$PIPD_CKPT_DIR" \
    --pipd_dir "$PIPD_DIR" \
    --vllm_server_host "localhost" \
    --vllm_server_port "$VLLM_PORT" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EC=${PIPESTATUS[0]}

if [ $TRAIN_EC -eq 0 ]; then
    notify "✅ CVRP20 GRPO v5 训练完成" \
"output: $OUTPUT_DIR_BASE/${PROBLEM}_n${SIZE}/final_model
结束: $(date '+%Y-%m-%d %H:%M:%S')"
    TRAINING_COMPLETED=1
fi

stop_vllm_server

echo ""
echo "============================================================"
echo "  完成! exit=$TRAIN_EC  $(date)"
echo "  训练日志: $TRAIN_LOG"
echo "  vLLM 日志: $VLLM_LOG"
echo "  模型输出: $OUTPUT_DIR_BASE/${PROBLEM}_n${SIZE}/final_model"
echo "============================================================"

exit $TRAIN_EC
