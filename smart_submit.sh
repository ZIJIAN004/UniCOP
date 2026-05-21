#!/bin/bash
# smart_submit.sh — sbatch 包装: 提交前扫节点 GPU 占用, 自动 --nodelist 选最干净的
#
# 用法:
#   ./smart_submit.sh UniCOP-Distill/submit_sft_qwen3_full.sh
#   ./smart_submit.sh --gpus_needed=4 UniCOP-Reason-Mask/submit_grpo_cvrp20_v5_mask.sh
#
# 工作原理:
#   1. 跑 `scontrol show node canele{0..3}` 看每节点 SLURM 视角的 GresUsed
#      (这只能看到通过 SLURM 申请的占用, 看不到 luk 那种绕开 SLURM 直接占的)
#   2. 跑 `srun --immediate=5 --nodelist=<node> --gpus=0 nvidia-smi` 抓每节点真实空闲
#      (这能看到所有占用, 包括外部进程)
#   3. 按 "实际空闲 GPU 数 >= 需求" 过滤, 选最干净的 K 个节点放进 --nodelist
#   4. 若所有节点都不够干净: 用默认调度 + 依赖 sbatch 自身的 auto-resubmit 兜底

set -euo pipefail

GPUS_NEEDED=4
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus_needed=*) GPUS_NEEDED="${1#--gpus_needed=}"; shift ;;
        --gpus_needed)   GPUS_NEEDED="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

if [ ${#ARGS[@]} -lt 1 ]; then
    echo "Usage: $0 [--gpus_needed=N] <sbatch_script> [extra sbatch args]"
    echo "  默认 --gpus_needed=4 (跟 SFT 配置一致)"
    exit 1
fi

NODES=(canele0 canele1 canele2 canele3)
GOOD_NODES=()

echo "============================================================"
echo "  smart_submit: 扫描 canele 节点 GPU 占用 (need ${GPUS_NEEDED} clean GPUs)"
echo "============================================================"

for node in "${NODES[@]}"; do
    # SLURM 视角 (能看到自己调度过的, 看不到外部)
    slurm_info=$(scontrol show node "$node" 2>/dev/null || echo "")
    slurm_used=$(echo "$slurm_info" | grep -oP 'GresUsed=gpu:a5000:\K\d+' | head -1 || echo "?")
    slurm_total=$(echo "$slurm_info" | grep -oP 'Gres=gpu:a5000:\K\d+' | head -1 || echo "8")
    slurm_state=$(echo "$slurm_info" | grep -oP 'State=\K\S+' | head -1 || echo "?")

    # nvidia-smi 实际视角 (跳到节点上, 5s 超时)
    # 用 --gpus=0 不占卡, 只是借节点跑 nvidia-smi
    nvsmi_free_min="?"
    if [ "$slurm_state" != "DOWN" ] && [ "$slurm_state" != "DRAIN" ]; then
        nv_out=$(srun --immediate=5 --nodelist="$node" --gpus=0 --time=00:00:30 \
                 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "")
        if [ -n "$nv_out" ]; then
            # 真实空闲 = free > 20 GiB 的 GPU 数量
            nvsmi_free_count=$(echo "$nv_out" | awk '$1 > 20000' | wc -l)
            nvsmi_free_min=$(echo "$nv_out" | sort -n | head -1)
        else
            nvsmi_free_count="?"
        fi
    else
        nvsmi_free_count="-"
    fi

    printf "  %-9s  state=%-12s  slurm=%s/%s  nvsmi_clean_gpus=%s  min_free=%s MiB\n" \
        "$node" "$slurm_state" "$slurm_used" "$slurm_total" "$nvsmi_free_count" "$nvsmi_free_min"

    # 加入候选: SLURM 没占, 且 nvidia-smi 看到 >= GPUS_NEEDED 张干净卡
    if [ "$slurm_state" != "DOWN" ] && [ "$slurm_state" != "DRAIN" ] \
       && [ "$nvsmi_free_count" != "?" ] && [ "$nvsmi_free_count" != "-" ] \
       && [ "$nvsmi_free_count" -ge "$GPUS_NEEDED" ]; then
        GOOD_NODES+=("$node")
    fi
done

echo "============================================================"

if [ ${#GOOD_NODES[@]} -eq 0 ]; then
    echo "⚠️  没有节点同时满足 (SLURM 空闲 + nvidia-smi 真实空闲 >= ${GPUS_NEEDED})"
    echo "    退化用默认调度 (依赖 sbatch 自身的 auto-resubmit 兜底)"
    sbatch "${ARGS[@]}"
else
    NODELIST=$(IFS=,; echo "${GOOD_NODES[*]}")
    echo "✓ 候选节点 (实际 free GPU >= ${GPUS_NEEDED}): $NODELIST"
    echo "  提交: sbatch --nodelist=$NODELIST ${ARGS[*]}"
    sbatch --nodelist="$NODELIST" "${ARGS[@]}"
fi
