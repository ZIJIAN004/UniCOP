# preflight_gpu.sh — GPU 占用预检 (source 使用, 不要 chmod 直接跑)
#
# 场景: SLURM 分到节点后, 偶发该节点的卡仍被别的进程占着 (上个 job 的 zombie /
#   共享节点争用 / flaky 节点)。此时本 job 一 CUDA init 就 OOM 或 NCCL 崩。
#   本预检在 job 起步、还没碰 CUDA 之前查一遍分到的卡:
#     - 干净 → 返回, 正常往下跑;
#     - 被占 → 把本节点加进 exclude, sbatch 重投本 job (去别的节点排队), 本 job 退出。
#   重投次数封顶 (PREFLIGHT_MAX_RESUBMIT, 默认 8), 防整片集群繁忙时无限重投。
#
# 调用方在 source 前需先 cd 到脚本目录, 并导出:
#   SUBMIT_SCRIPT  — 要重投的 submit 绝对路径 (如 "$(pwd)/submit_grpo_cvrp20_v6.sh")
#   BASE_EXCLUDE   — 该 submit #SBATCH --exclude 的基线节点 (重投用 CLI --exclude 会覆盖
#                    #SBATCH, 故必须带上, 否则基线 exclude 丢失)
# 可选 env:
#   PREFLIGHT_GPU=0        — 整体关闭预检 (预检本身误判时的逃生开关)
#   GPU_BUSY_MIB=500       — 单卡已用显存超过此值 (MiB) 视为被占
#   PREFLIGHT_MAX_RESUBMIT=8
#   EXTRA_EXCLUDE          — 已累积的坏节点 (逗号分隔), 由重投链自动透传, 勿手动设
#   PREFLIGHT_RESUBMIT_COUNT — 重投计数, 自动透传, 勿手动设

preflight_gpu_or_resubmit() {
    if [ "${PREFLIGHT_GPU:-1}" = "0" ]; then
        echo "[preflight] PREFLIGHT_GPU=0, 跳过 GPU 占用预检。"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[preflight] 无 nvidia-smi, 跳过预检 (不阻断)。"
        return 0
    fi

    local thresh="${GPU_BUSY_MIB:-500}"
    local node="${SLURMD_NODENAME:-$(hostname -s)}"
    local n_resub="${PREFLIGHT_RESUBMIT_COUNT:-0}"
    local max_resub="${PREFLIGHT_MAX_RESUBMIT:-8}"

    # 本 job 此刻尚未起 CUDA → 分到的卡上任何 >thresh 的已用显存都是"别人占的"。
    # (SLURM GPU cgroup 下 nvidia-smi 只列分到的卡; memory.used 是物理卡属性, 即便
    #  占用进程不在本 cgroup 也照样可见。)
    local busy_lines
    busy_lines=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
                 | awk -v t="$thresh" 'NF>=2 && $NF+0 > t {print}')

    if [ -z "$busy_lines" ]; then
        echo "[preflight] 节点 $node: 分配的 GPU 均空闲 (阈值 ${thresh}MiB), 继续。"
        return 0
    fi

    echo "[preflight] ⚠️ 节点 $node 上分配的 GPU 被占用 (>${thresh}MiB):"
    echo "$busy_lines" | sed 's/^/[preflight]   idx,used(MiB)= /'

    if [ "$n_resub" -ge "$max_resub" ]; then
        echo "[preflight] ❌ 重投已达上限 $max_resub 次, 放弃 (整片集群繁忙? 或 GPU_BUSY_MIB 过严)。本 job 仍在坏节点上, 直接退出。"
        exit 1
    fi
    if [ -z "${SUBMIT_SCRIPT:-}" ]; then
        echo "[preflight] ❌ 未设 SUBMIT_SCRIPT, 无法重投; 直接退出避免在坏节点上跑崩。"
        exit 1
    fi

    # 重投: 累积 exclude = 基线 + 已知坏节点 + 本坏节点。--export=ALL 透传当前实验 env
    # (PROC_ALPHA_V6/LR/... 都在本 job 环境里), 再叠加 bookkeeping。
    local seed="${EXTRA_EXCLUDE:-${BASE_EXCLUDE:-}}"
    local new_exclude="${seed:+$seed,}$node"
    echo "[preflight] → exclude=$new_exclude, 重投 submit (第 $((n_resub+1))/$max_resub 次): $SUBMIT_SCRIPT"
    sbatch --exclude="$new_exclude" \
           --export="ALL,EXTRA_EXCLUDE=${new_exclude},PREFLIGHT_RESUBMIT_COUNT=$((n_resub+1))" \
           "$SUBMIT_SCRIPT" \
        && echo "[preflight] 重投已提交, 本 (坏节点) job 退出。" \
        || { echo "[preflight] ❌ sbatch 重投失败; 本 job 退出。"; exit 1; }
    exit 0
}
