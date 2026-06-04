#!/bin/bash
# sweep_proc_alpha_v6.sh — 扫 v6 主轴 proc_alpha_v6, 全自动流水线: train → merge+BO1
#
#   分批推进 (每批 BATCH=2 个 pa, 对集群其他用户友好):
#     [pa_A 训练(7卡) ∥ pa_B 训练(7卡)]
#         └─ 两个都结束 → 自动投 merge+BO1 eval job (4卡, submit_sweep_eval_bo1_v6.sh,
#            DIRS 限定本批两个目录; 4 pa 并行版, 本批 2 个各占 1 卡)
#                └─ eval 结束 → 下一批 [pa_C ∥ pa_D] 才开始 (afterany 依赖链)
#   不再需要训练完手动投 eval。
#
#   每个 pa = 一个独立 sbatch job, 复用 submit_grpo_cvrp20_v6.sh (train-only),
#   PROC_ALPHA_V6 经 --export 注入 → train.py:180 覆盖 config.proc_alpha_v6
#   → grpo_prm_trainer.py:1636 读取。输出目录隔离: output_v6_lr<LR>_ep<EP>_pa<PA>_nt<NT>。
#
#   ⚠️ 已知边角: 训练 job 的 preflight 预检若发现卡被占会"自杀重投", afterany 会让本批
#   eval 提前启动 → eval 找不到 adapter 标 FAILED 并通知 (不会写坏数据)。收到该通知后
#   等重投的训练完成, 手动补一次: sbatch submit_sweep_eval_bo1_v6.sh (幂等, 只补缺的)。
#
#   用法 (集群登录节点, git pull 之后):
#       bash sweep_proc_alpha_v6.sh                       # 默认本轮: 800 1200 1700 2200
#       PA_LIST="2700 4000" bash sweep_proc_alpha_v6.sh   # 自定义
#   可覆盖: LR=2e-5 EPOCHS=1 NUM_TRAIN=1000 BATCH=2 EVAL_NUM_TEST=1000

set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1}"
NUM_TRAIN="${NUM_TRAIN:-1000}"          # 训练用全量 1000 (训练给足信号, 不砍)
PA_LIST="${PA_LIST:-800 1200 1700 2200}"  # 本轮: 接 600 的几何序列, 上探 PRM 主导区
BATCH="${BATCH:-2}"                     # 每批并行的 pa 数 (2×7卡 训练 + 批后 4卡 eval)
EVAL_NUM_TEST="${EVAL_NUM_TEST:-1000}"  # 与 optimal 冻结集对齐, 勿改小

PA_ARR=($PA_LIST)
echo "############## sweep proc_alpha_v6 (train → merge+BO1 流水线) ##############"
echo "  LR=$LR  EPOCHS=$EPOCHS  NUM_TRAIN=$NUM_TRAIN  PA_LIST=[$PA_LIST]  BATCH=$BATCH"
echo "  每批: $BATCH 个 pa 并行训练 → 双双结束自动 merge+BO1 → eval 完才放行下一批"
echo

prev_eval=""    # 上一批 eval job id: 下一批训练依赖它
i=0
while [ "$i" -lt "${#PA_ARR[@]}" ]; do
    batch=("${PA_ARR[@]:$i:$BATCH}")
    train_ids=(); dirs=""
    for PA in "${batch[@]}"; do
        OUT="output_v6_lr${LR}_ep${EPOCHS}_pa${PA}_nt${NUM_TRAIN}"
        dep=""
        if [ -n "$prev_eval" ]; then
            dep="--dependency=afterany:${prev_eval}"
            echo ">>> 提交训练 pa=$PA → $OUT/cvrp_n20  (等上一批 eval job $prev_eval 结束)"
        else
            echo ">>> 提交训练 pa=$PA → $OUT/cvrp_n20  (立即排队)"
        fi
        out=$(sbatch $dep \
               --job-name="zijia_v6_pa${PA}" \
               --export="ALL,PROC_ALPHA_V6=${PA},LR=${LR},EPOCHS=${EPOCHS},NUM_TRAIN=${NUM_TRAIN}" \
               submit_grpo_cvrp20_v6.sh)
        echo "    $out"
        train_ids+=("$(echo "$out" | awk '{print $NF}')")
        dirs="$dirs $OUT"
    done

    # 本批 merge+BO1: 依赖本批全部训练结束 (afterany: 失败也放行, eval 会自行校验 adapter)
    dep_list=$(IFS=:; echo "${train_ids[*]}")
    out=$(sbatch --dependency="afterany:${dep_list}" \
           --job-name="zijia_v6_swpeval" \
           --export="ALL,DIRS=${dirs# },EVAL_NUM_TEST=${EVAL_NUM_TEST}" \
           submit_sweep_eval_bo1_v6.sh)
    echo ">>> 提交本批 merge+BO1 (DIRS=${dirs# }, 等训练 job ${dep_list})"
    echo "    $out"
    prev_eval=$(echo "$out" | awk '{print $NF}')
    i=$((i + BATCH))
done

echo
echo "全部已提交: 训练与 eval 以批为单位链式推进。查看队列: squeue -u \$USER"
echo "结果将落在 eval_results_matrix/v6_lr${LR}_ep${EPOCHS}_pa<PA>_nt${NUM_TRAIN}_r1_RL_BO1.json"
squeue -u "${USER:-$(whoami)}" 2>/dev/null || true
