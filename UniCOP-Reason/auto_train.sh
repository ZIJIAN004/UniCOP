#!/bin/bash
# auto_train.sh — 自动化 GRPO+POMO PRM 训练 + 训练后评估
# 训练矩阵：TSP / VRPTW × n ∈ {20, 50, 100} = 6 个任务
#
# 调度策略（每个任务）：
#   阶段 1：等待 4 张空闲卡 → 1 张跑 vLLM server，3 张 ZeRO-3 + LoRA 训练
#   阶段 2：若 OOM → 等待 8 张空闲卡 → 1 vLLM + 7 训练重试
#   阶段 3：若训练正常结束 OR 非 OOM 错误退出 → 检查模型文件
#           存在 → 等 1 张空闲卡 → 跑 evaluate.py
#           OOM 失败 → 跳过 eval
#
# 架构：
#   - vLLM server 独立 1 卡，加载基座模型 + 接受 LoRA 热更新
#   - 训练进程通过 HTTP 和 vLLM 通信做 rollout（比 HF generate 快 10-30x）
#   - LoRA 每 step 同步到 vLLM（仅 ~40M 参数，秒级）
#
# 显存优化：
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（碎片化显存修复）
#   - --gradient_checkpointing（激活重计算，砍 30~50% 激活内存）
#   - --zero_stage 3（权重+优化器+梯度跨卡分片）

WORK_DIR="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason"
LOG_DIR="$WORK_DIR/logs"
EVAL_RESULT_DIR="$WORK_DIR/eval_results_auto_train"
mkdir -p "$LOG_DIR" "$EVAL_RESULT_DIR"

# ── GPU 调度参数 ─────────────────────────────────────────────────────
TOTAL_GPUS=8
INIT_GPUS=4              # 训练阶段 1：1 vLLM + 3 训练 = 4 卡
OOM_GPUS=8               # 训练阶段 2：OOM 升级到 1 vLLM + 7 训练 = 8 卡
EVAL_GPUS=1              # 评估：单卡足够
FREE_GPUS=""

# ── vLLM server 参数 ─────────────────────────────────────────────────
VLLM_PORT_BASE=8000      # 实际端口 = BASE + 任务索引，避免多任务冲突
VLLM_GPU_MEM_UTIL=0.85   # vLLM 卡显存利用率（防 OOM 保守值）
VLLM_MAX_MODEL_LEN=5120  # prompt(768) + completion(4096) + 余量;不限会用模型 config 里 131072 → KV 塞不下
VLLM_DTYPE=bfloat16      # bf16 匹配训练侧精度
# GRPO 一个 prompt 生成 num_generations=8 条 completion,prefix caching 可大幅加速
VLLM_ENABLE_PREFIX_CACHING=True
VLLM_STARTUP_TIMEOUT=300 # server 启动最长等待（秒）

# ── 训练资源路径（POMO 路径需要自己填） ─────────────────────────
# 测试阶段: MODEL_BASE 直接指向 bak 目录里的 SFT 产物,免去 mv / 软链折腾。
# sort -r | head -1 自动取最新的一个 bak (按时间戳字典序,与 date +%Y%m%d_%H%M%S 一致)。
# 正式训练阶段: 把 output_sft_r1_v2 物理 mv 回 UniCOP/UniCOP-Distill/, 然后改为
#   MODEL_BASE="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/output_sft_r1_v2/merged_model"
MODEL_BASE=$(ls -d /Data04/yangzhihan/lzj/UniCOP-Distill.bak_*/output_sft_r1_v2/merged_model 2>/dev/null | sort -r | head -1)
if [ -z "$MODEL_BASE" ]; then
    echo "❌ 找不到 /Data04/yangzhihan/lzj/UniCOP-Distill.bak_*/output_sft_r1_v2/merged_model"
    echo "   请确认 bak 目录存在,或把 SFT 产物 mv 回 monorepo 后改回此处路径。"
    exit 1
fi
echo "[MODEL_BASE] $MODEL_BASE"
POMO_CKPT_DIR="/Data04/yangzhihan/lzj/POMO-Baseline/result"
POMO_BASELINE_DIR="/Data04/yangzhihan/lzj/POMO-Baseline"
# PIP-D (NeurIPS 2024) for TSPTW,和 POMO 目录并存
PIPD_CKPT_DIR="/Data04/yangzhihan/lzj/PIP-D baseline/POMO+PIP/pretrained/TSPTW"
PIPD_DIR="/Data04/yangzhihan/lzj/PIP-D baseline/POMO+PIP"

# ── TRL CLI 二进制 ────────────────────────────────────────────────────
# 用当前 conda/virtualenv 里的 trl binary,而不是 ~/.local/bin/trl
# (后者 shebang 可能指向错 Python 导致 ModuleNotFoundError)
# 也不能用 `python -m trl.cli` —— trl.cli 是 package 无 __main__.py
TRL_BIN="$(dirname "$(which python)")/trl"
if [ ! -x "$TRL_BIN" ]; then
    echo "❌ TRL binary 未找到: $TRL_BIN"
    echo "   请在当前 env 安装: pip install 'trl[vllm]==1.1.0'"
    exit 1
fi
echo "TRL CLI: $TRL_BIN"

# ── 手机通知 (Server 酱 → 微信) ──────────────────────────────────────
# 通知在: 启动、全部完成、异常退出 时触发
# 允许环境变量 SCKEY 覆盖 (避免把 key 暴露在脚本里如果你要开源这个)
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"

notify() {
    # $1 = title, $2 = desp (正文,可留空)
    local title="$1"
    local desp="${2:-}"
    if [ -n "$SCKEY" ]; then
        curl -s --max-time 10 "https://sctapi.ftqq.com/${SCKEY}.send" \
            --data-urlencode "title=${title}" \
            --data-urlencode "desp=${desp}" > /dev/null 2>&1 || true
    fi
}

# ── 训练矩阵 ─────────────────────────────────────────────────────────
# 当前仅跑 TSP n=20 一个任务,用于验证 grad_norm bug 修复后的训练收敛性。
# 完整矩阵 (TSP/VRPTW × 20/50/100) 待首轮验证通过后再恢复。
PROBLEMS=("tsp")
SIZES=(20)
ZERO_STAGE=3
OUTPUT_DIR_BASE="$WORK_DIR/output"

# ── 评估参数 ─────────────────────────────────────────────────────────
EVAL_NUM_TEST=10
EVAL_MAX_COMPLETION=10000
EVAL_BATCH_SIZE=4

# ── 全局日志 ─────────────────────────────────────────────────────────
exec > >(tee -a "$LOG_DIR/auto_train_$(date '+%Y%m%d_%H%M%S').log") 2>&1

# ── 检查 N 张空闲卡 ──────────────────────────────────────────────────
# "空闲"定义: 无 compute process AND 显存占用 < GPU_FREE_MEM_THRESHOLD_MB
# 单查 compute-apps 不够: 其他用户进程/nvidia-smi 权限限制/僵尸 context
# 都可能让 compute-apps 为空但显存其实被占了
GPU_FREE_MEM_THRESHOLD_MB=500   # 小于这个就当空闲 (系统本身可能占几十 MB)

check_gpus_free() {
    local need=$1
    FREE_GPUS=""
    local free=0
    for gpu_id in $(seq 0 $((TOTAL_GPUS - 1))); do
        # 检查 1: compute process 数
        local procs
        procs=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
                | grep -v -i "xorg\|gnome\|kde\|wayland\|Xwayland" \
                | grep -c '[0-9]')
        # 检查 2: 显存占用 MB
        local mem_used
        mem_used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                   | tr -d '[:space:]')
        # 两个条件都满足才算空闲
        if [ "$procs" -eq 0 ] && [ -n "$mem_used" ] && [ "$mem_used" -lt "$GPU_FREE_MEM_THRESHOLD_MB" ]; then
            if [ -z "$FREE_GPUS" ]; then
                FREE_GPUS="$gpu_id"
            else
                FREE_GPUS="$FREE_GPUS,$gpu_id"
            fi
            free=$((free + 1))
        fi
        if [ "$free" -ge "$need" ]; then
            return 0
        fi
    done
    return 1
}

wait_for_gpus() {
    local need=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待 ${need} 张空闲 GPU..."
    while ! check_gpus_free "$need"; do
        sleep 30
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $FREE_GPUS 已空闲"
}

# ── 拆分 FREE_GPUS：首张给 vLLM，其余给训练 ──────────────────────────
# 输入全局 FREE_GPUS（CSV），输出全局 VLLM_GPU / TRAIN_GPUS
split_gpus() {
    VLLM_GPU="${FREE_GPUS%%,*}"                          # 第一张
    TRAIN_GPUS="${FREE_GPUS#*,}"                         # 剩余所有
}

# ── 启动 vLLM server（后台，返回 PID 到 VLLM_PID） ────────────────────
start_vllm_server() {
    local vllm_gpu=$1
    local port=$2
    local log_file=$3

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 vLLM server | GPU=$vllm_gpu | port=$port"
    # 注意：TRL server 模式把 LoRA merge 进 base 后推完整权重，不走 vLLM 的
    # LoRA adapter 接口，所以不加 --enable-lora（加了反而触发 vLLM LoRA 模块 bug）
    # $TRL_BIN 指向当前 env 的 trl binary（在脚本顶部定义）
    # FLASHINFER_DISABLE_VERSION_CHECK=1: 绕过 flashinfer/flashinfer-cubin 版本不一致检查
    #
    # --logits-processors 关键参数:
    #   vLLM V1 废弃 SamplingParams(logits_processors=...),必须在 server 启动
    #   时通过 CLI 注册 AdapterLogitsProcessor 子类。我们的实现在
    #   utils/vllm_ngram_processor.py:NoRepeatNgramAdapterLP,等价于 HF 的
    #   no_repeat_ngram_size。训练端通过 SamplingParams.extra_args 传 n=6 开关。
    #   参考: https://docs.vllm.ai/en/stable/features/custom_logitsprocs/
    #
    # PYTHONPATH="$WORK_DIR": 让 vLLM 子进程能 import 到 utils.vllm_ngram_processor
    #   (本项目非 pip install 方式,靠 PYTHONPATH 暴露 Python 包路径)
    PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$vllm_gpu" \
    CUDA_HOME=/Data04/yangzhihan/envs/analog_env/targets/x86_64-linux \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
        "$TRL_BIN" vllm-serve \
        --model "$MODEL_BASE" \
        --tensor_parallel_size 1 \
        --port "$port" \
        --gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --dtype "$VLLM_DTYPE" \
        --enable_prefix_caching "$VLLM_ENABLE_PREFIX_CACHING" \
        --trust_remote_code True \
        --logits-processors utils.vllm_ngram_processor:NoRepeatNgramAdapterLP \
        > "$log_file" 2>&1 &
    VLLM_PID=$!

    # 等健康检查通过
    local waited=0
    while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动失败，详见 $log_file"
            return 1
        fi
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ vLLM server 就绪 (pid=$VLLM_PID, 用时 ${waited}s)"
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动超时 (${VLLM_STARTUP_TIMEOUT}s)"
    kill "$VLLM_PID" 2>/dev/null || true
    return 1
}

# ── 关闭 vLLM server ─────────────────────────────────────────────────
stop_vllm_server() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 关闭 vLLM server (pid=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}

# 脚本异常退出时自动清理 vLLM + 手机推送
# TRAINING_COMPLETED 在脚本末尾正常结束时置 1,此时 EXIT trap 里的推送会判断跳过
# (避免正常结束时重复推送两条)
TRAINING_COMPLETED=0
on_exit() {
    local exit_code=$?
    stop_vllm_server
    if [ "$TRAINING_COMPLETED" != "1" ] && [ "$exit_code" != "0" ]; then
        notify "❌ UniCOP GRPO 异常退出" \
"退出码: $exit_code
时间: $(date '+%Y-%m-%d %H:%M:%S')
最后一个日志:
$(ls -t $LOG_DIR/train_*.log 2>/dev/null | head -1 | xargs tail -n 20 2>/dev/null || echo '(无日志)')"
    fi
}
trap 'on_exit' EXIT INT TERM

# ── 单次训练（返回 0=成功 / 99=OOM / 1=其他错误） ───────────────
# 用法：run_train <problem> <size> <total_gpus> <free_gpus_csv> <task_idx>
# vLLM server mode: 首张卡跑 vllm-serve (含 NoRepeatNgramAdapterLP),
# 其余卡跑 accelerate launch 训练,通过 HTTP 同步权重 + 接收 completions。
run_train() {
    local problem=$1
    local size=$2
    local total_gpus=$3
    local free_gpus=$4
    local task_idx=$5

    # 拆分: 首张卡 → vllm-serve,其余卡 → 训练
    FREE_GPUS="$free_gpus"
    split_gpus   # 设置 VLLM_GPU / TRAIN_GPUS
    local train_proc=$((total_gpus - 1))   # 训练进程数 = 总卡数 - 1 (vLLM 占 1 张)
    local port=$((VLLM_PORT_BASE + task_idx))
    local label="${problem}_n${size}_g${train_proc}"
    local ts=$(date '+%Y%m%d_%H%M%S')
    local vllm_log="$LOG_DIR/vllm_${label}_${ts}.log"
    local train_log="$LOG_DIR/train_${label}_${ts}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task | problem=$problem | size=$size"
    echo "  vLLM GPU:  $VLLM_GPU    port=$port    log=$vllm_log"
    echo "  训练 GPUs: $TRAIN_GPUS ($train_proc 进程)    log=$train_log"

    # 启动 vLLM server (后台,阻塞到健康检查通过)
    if ! start_vllm_server "$VLLM_GPU" "$port" "$vllm_log"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ vLLM server 启动失败,终止本任务"
        return 1
    fi

    # 启动训练 (剩余 train_proc 张卡)
    # PYTHONPATH 给训练侧,让 utils.vllm_ngram_processor 和其它 utils 能被 import
    # (server 侧已在 start_vllm_server 内部设置过)
    PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_HOME=/Data04/yangzhihan/envs/analog_env/targets/x86_64-linux \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
        python -m accelerate.commands.launch --num_processes "$train_proc" "$WORK_DIR/train.py" \
        --problem "$problem" \
        --problem_size "$size" \
        --num_train 2000 \
        --model "$MODEL_BASE" \
        --num_gpus "$train_proc" \
        --zero_stage "$ZERO_STAGE" \
        --gradient_checkpointing \
        --output_dir "$OUTPUT_DIR_BASE" \
        --pomo_ckpt_dir "$POMO_CKPT_DIR" \
        --pomo_baseline_dir "$POMO_BASELINE_DIR" \
        --pipd_ckpt_dir "$PIPD_CKPT_DIR" \
        --pipd_dir "$PIPD_DIR" \
        --vllm_server_host "localhost" \
        --vllm_server_port "$port" \
        2>&1 | tee "$train_log"

    local exit_code=${PIPESTATUS[0]}

    # 无论成功失败都先停掉 vLLM server,释放显存/端口给下一任务
    stop_vllm_server

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 训练完成 ($label)"
        return 0
    fi

    if grep -qiE "out of memory|OOM|CUDA out of memory|OutOfMemoryError" "$train_log"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ OOM ($label, train_proc=$train_proc)"
        return 99
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 非 OOM 错误 (exit=$exit_code, $label)"
    return 1
}

# ── 单次评估（默认 1 张卡） ─────────────────────────────────────
run_eval() {
    local problem=$1
    local size=$2
    local model_path=$3
    local gpus=$4

    local label="${problem}_n${size}"
    local log_file="$LOG_DIR/eval_${label}_$(date '+%Y%m%d_%H%M%S').log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating | model=$model_path | GPU=$gpus"
    echo "  log: $log_file"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_HOME=/Data04/yangzhihan/envs/analog_env/targets/x86_64-linux \
    CUDA_VISIBLE_DEVICES="$gpus" python "$WORK_DIR/evaluate.py" \
        --model_path "$model_path" \
        --problem "$problem" \
        --problem_size "$size" \
        --num_test "$EVAL_NUM_TEST" \
        --max_completion_length "$EVAL_MAX_COMPLETION" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --prompt_mode think \
        --save_dir "$EVAL_RESULT_DIR" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 评估完成 ($label)"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 评估失败 ($label, exit=$exit_code)"
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════
# 主流程：遍历 (problem, size) 所有组合
# ══════════════════════════════════════════════════════════════════════
cd "$WORK_DIR" || exit 1

if [ ! -d "$MODEL_BASE" ]; then
    echo "❌ 模型路径不存在: $MODEL_BASE"
    exit 1
fi

if [ ! -d "$POMO_BASELINE_DIR" ]; then
    echo "❌ POMO-Baseline 目录不存在: $POMO_BASELINE_DIR"
    exit 1
fi
if [ ! -d "$POMO_CKPT_DIR" ]; then
    echo "❌ POMO checkpoint 目录不存在: $POMO_CKPT_DIR"
    exit 1
fi

TOTAL_TASKS=$((${#PROBLEMS[@]} * ${#SIZES[@]}))
DONE_COUNT=0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 共 ${TOTAL_TASKS} 个任务 (训练+评估) =========="
echo "  Problems:  ${PROBLEMS[@]}"
echo "  Sizes:     ${SIZES[@]}"
echo "  ZeRO:      stage $ZERO_STAGE | gradient_checkpointing 已开启"
echo "  显存:      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  GPU:       首轮固定 GPU 4,5,6,7 直接开跑 (不等空闲)"
echo "             OOM 才等 $INIT_GPUS 张空闲卡重试 / 非 OOM 直接退出 / 评估 $EVAL_GPUS"
echo "  生成模式:  vLLM server (--logits-processors NoRepeatNgramAdapterLP)"
echo ""

# 手机推送: 训练启动
notify "🚀 UniCOP GRPO 开始训练" \
"任务数: ${TOTAL_TASKS} (${PROBLEMS[@]} × ${SIZES[@]})
启动时间: $(date '+%Y-%m-%d %H:%M:%S')
期间异常会另外通知"

for problem in "${PROBLEMS[@]}"; do
    for size in "${SIZES[@]}"; do
        DONE_COUNT=$((DONE_COUNT + 1))
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== [${DONE_COUNT}/${TOTAL_TASKS}] $problem n=$size ====="

        # ── 阶段 1: 不等空闲卡,直接在 GPU 4,5,6,7 上开跑 ────────
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 直接在 GPU 4,5,6,7 上启动训练 (不等空闲)"
        run_train "$problem" "$size" 4 "4,5,6,7" "$DONE_COUNT"
        ec=$?

        # ── 阶段 2: 只在 OOM 时才等空闲卡重试 ──────────────────
        # 非 OOM (ec=1) 直接退出,避免无意义重试浪费资源
        if [ $ec -eq 1 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 非 OOM 错误 (ec=1),直接退出脚本"
            notify "❌ UniCOP 非 OOM 错误退出" \
"任务: $problem n=$size
exit code: $ec
时间: $(date '+%Y-%m-%d %H:%M:%S')
详见: $LOG_DIR/train_*.log"
            exit 1
        fi

        if [ $ec -eq 99 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ OOM,等待 ${INIT_GPUS} 张空闲卡后重试"
            wait_for_gpus "$INIT_GPUS"
            run_train "$problem" "$size" "$INIT_GPUS" "$FREE_GPUS" "$DONE_COUNT"
            ec=$?

            # 二次重试若是非 OOM 错误,也直接退出
            if [ $ec -eq 1 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ OOM 重试后遇非 OOM 错误,退出"
                exit 1
            fi
        fi

        # ── 阶段 3：评估（仅在非 OOM 失败时触发） ────────────────
        if [ $ec -eq 99 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ${problem} n=${size} OOM 重试仍失败，跳过 eval"
            continue
        fi

        # 到这里只可能 ec == 0 (非 OOM 错误已在上面 exit 1)
        MODEL_OUT="$OUTPUT_DIR_BASE/${problem}_n${size}/final_model"
        if [ ! -d "$MODEL_OUT" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏭️ 模型文件不存在 $MODEL_OUT，跳过 eval"
            continue
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 模型文件就绪，准备 eval: $MODEL_OUT"
        wait_for_gpus "$EVAL_GPUS"
        run_eval "$problem" "$size" "$MODEL_OUT" "$FREE_GPUS"
    done
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 全部任务结束 =========="
echo "  训练日志:  $LOG_DIR/train_*.log"
echo "  评估日志:  $LOG_DIR/eval_*.log"
echo "  评估结果:  $EVAL_RESULT_DIR/"
echo "  模型输出:  $OUTPUT_DIR_BASE/{problem}_n{size}/final_model"

# 手机推送: 全部任务完成
notify "✅ UniCOP GRPO 全部完成" \
"完成 ${TOTAL_TASKS} 个任务
结束时间: $(date '+%Y-%m-%d %H:%M:%S')
Problems: ${PROBLEMS[@]}
Sizes: ${SIZES[@]}
评估结果目录: ${EVAL_RESULT_DIR}"

# 正常退出,清除 EXIT trap 的错误推送 (见下方 trap 逻辑)
TRAINING_COMPLETED=1
