#!/bin/bash
# run_grpo_cvrp20_v6_instruct.sh
# ── yangzhihan(A*STAR-Zhihan, 直连 SSH 无 SLURM) 专用 v6 RL 启动器 ──────────────
#   基座 = Qwen3-4B-Instruct-2507 的 SFT 产物(非 thinking, 对齐 FOARL instruct 范式):
#     $DISTILL_DIR/output_sft_qwen3_instruct_template_cvrp20/final_model
#     (zhihan 实际路径: /Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/output_sft_qwen3_instruct_template_cvrp20/final_model)
#
#   本脚本是 run_grpo_cvrp20_v6.sh(纯净 v6 wrapper)的薄封装, 只额外:
#     - 把 RL 起点切到 instruct SFT 产物 (BASE_MODEL_TYPE=qwen3_instruct);
#       paths.sh 据此自动设采样参数 T=0.7/top_p=0.8/top_k=20 (Qwen3-Instruct-2507 官方 & FOARL 受控对比口径);
#     - 设温和超参 (LR/EPOCHS/SAVE_STEPS/NUM_TRAIN/PROC_ALPHA_V6), 与 zhuoyi submit_grpo_cvrp20_v6.sh 同口径,
#       便于与 thinking 版 v6 直接对照;
#     - 输出目录带 instruct 标识 + 超参标注, 与 thinking v6 (output_v6_*) 物理隔离, 避免误 resume。
#   reward 信号: REWARD_SCHEME=v6 (PRM 批级截尾标准化 + sigmoid, proc_alpha_v6) 由 run_grpo_cvrp20_v6.sh 设;
#               A_out (A_feas+A_outcome) 完全复用 v5; use_mask=False(纯净)。
#
#   GPU(zhihan 8×3090 24G): run_grpo_cvrp20_v5.sh 在 astar-zhihan 上自动挑空闲卡 (1 vLLM + ≤6 训练),
#     NCCL 默认开 P2P/SHM 提速 (单机), vLLM gpu_memory_utilization=0.80 正是 24G 卡 + 4B 模型的甜点值。
#
#   ⚠️ zhihan 无 SLURM, 直连 SSH——必须在 tmux 里跑, 否则 SSH 一断 SIGHUP 连带杀光 torchrun/vLLM:
#       tmux new -s v6_instruct
#       cd /Data04/yangzhihan/lzj/UniCOP/UniCOP-Reason-Mask
#       bash run_grpo_cvrp20_v6_instruct.sh
#     (submit_grpo_cvrp20_v6.sh 是 zhuoyi 专用 SLURM 脚本, zhihan 不能用。)
#
#   常用覆盖(扫参/省算力): LR=1e-5 EPOCHS=2 PROC_ALPHA_V6=400 NUM_TRAIN=500 bash run_grpo_cvrp20_v6_instruct.sh

set -euo pipefail
_D="$(cd "$(dirname "$0")" && pwd)"

# ── tmux 提醒 (非强制): 检测到非 tmux/screen 的交互式会话时告警, 不阻断 ─────────────
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ -t 1 ]; then
    echo "⚠️  [警告] 当前不在 tmux/screen 会话中! zhihan 直连 SSH, 会话一断会 SIGHUP 杀光训练+vLLM。"
    echo "    强烈建议先: tmux new -s v6_instruct  再跑本脚本。 (5s 后继续, Ctrl-C 中止)"
    sleep 5
fi

# ── RL 起点: instruct SFT 产物 ──────────────────────────────────────────────
export BASE_MODEL_TYPE=qwen3_instruct          # paths.sh + run_v5 case 据此切 SFT 产物路径与采样参数

# ── 温和超参 (env 可覆盖; 与 zhuoyi submit_grpo_cvrp20_v6.sh 同口径) ─────────────
export REWARD_SCHEME="${REWARD_SCHEME:-v6}"     # PRM 批级截尾标准化+sigmoid (run_grpo_cvrp20_v6.sh 也会兜底设)
export LR="${LR:-2e-5}"                          # 对齐 v5 (1e-6 训练不足); train.py 经 env 覆盖 config
export EPOCHS="${EPOCHS:-1}"                     # 单 epoch
export SAVE_STEPS="${SAVE_STEPS:-20}"           # 每 20 step 存档 (短跑存 step20/40 + final)
export NUM_TRAIN="${NUM_TRAIN:-1000}"           # 一个 epoch 的训练实例数; 扫参可降到 500 提速
export PROC_ALPHA_V6="${PROC_ALPHA_V6:-1000}"   # v6 PRM 段注入权重 (扫参主轴); 默认 1000 (用户决定 2026-06-15, 同 config.py 默认)

# ── completion 长度: instruct think 比 thinking 长, config 默认 3584(为 thinking 调)截断 ──
#   症状: 训练 log stats/truncation_rate 高 + completion/p95≈max≈3584 顶格 → parse 腰斩 (2026-06-15 实测)。
#   MAX_COMPLETION 经 train.py env 覆盖 config.max_completion_length (不动全局默认, 保 thinking 基线)。
#   ⚠️ 三旋钮耦合, 调长度必须连带改 VLLM_MAX_MODEL_LEN, 并守住并发约束 (踩坑 #32):
#     VLLM_MAX_MODEL_LEN ≥ max_prompt(1280) + MAX_COMPLETION + overhead(~256)
#     vLLM "Maximum concurrency" 必须 ≥ num_generations(8), 否则 prefix-cache 重算损坏 → parse 再次腰斩。
#   并发上限 (util=0.80, 当前 KV 预算 ~46.5k tok): max_model_len ≤ ~5800 → MAX_COMPLETION ≤ ~4200。
#   4096 是"既扩长度又保并发≥8"的安全甜点 (config.py:44 也建议 4096); 6144→5632(1280+4096+256), 并发≈8.3x。
#   若 diag_parse_rate.py 实测 6144 下 p95 仍 >4200: 必须靠 enforce_eager(省 ~4.68G capture→升 util 夺回 KV)
#     或关 --enable_prefix_caching(重算退回正确 full-prefill) 才能再扩, 否则跌破并发触发踩坑 #32。
export MAX_COMPLETION="${MAX_COMPLETION:-4096}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-5632}"   # = 1280 + 4096 + 256; 并发≈8.3x ≥ num_gen=8

# ── A_feas 权重对齐 FOARL 设计 (capacity 主导), 保 v5 总量 5.5 → PRM/A_outcome 标定无需变 ──
#   FOARL CVRP R_f 比例 parse:depot:cov:cap = 0.2:0.1:0.1:0.6 (你的 FOARL/foarl_reward_cvrp.py:75-79)。
#   Mask 无独立 depot 分量、但有 format → 把 depot 的 0.1 位给 format; 按 5.5 总量缩放:
#     parse 0.2→1.1  coverage 0.1→0.55  capacity(=constraint) 0.6→3.3  format 0.1→0.55。
#   train.py 经 W_*_V5 env 覆盖 config.w_*_v5 (v6 复用 v5 A_feas: _compute_a_out_v5)。
export W_P_V5="${W_P_V5:-1.1}"                   # parse
export W_COV_V5="${W_COV_V5:-0.55}"             # coverage (FOARL 让位给 capacity, 从 3.5 降到 0.55)
export W_CONS_V5="${W_CONS_V5:-3.3}"           # constraint = CVRP 容量满足率 (FOARL 主导, 从 1.0 升到 3.3)
export W_F_V5="${W_F_V5:-0.55}"                 # format

# 输出目录带 instruct + FOARL 权重(fw) 标识 + 超参标注 → 与 thinking v6 / 默认权重 v6 互不覆盖, 避免误 resume
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$_D/output_v6_instruct_fw_lr${LR}_ep${EPOCHS}_pa${PROC_ALPHA_V6}_nt${NUM_TRAIN}}"

echo "[v6-instruct] BASE_MODEL_TYPE=$BASE_MODEL_TYPE REWARD_SCHEME=$REWARD_SCHEME LR=$LR EPOCHS=$EPOCHS PROC_ALPHA_V6=$PROC_ALPHA_V6 NUM_TRAIN=$NUM_TRAIN MAX_COMPLETION=$MAX_COMPLETION VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN"
echo "[v6-instruct] A_feas(FOARL对齐) parse=$W_P_V5 cov=$W_COV_V5 cap/cons=$W_CONS_V5 format=$W_F_V5 (总量 $(awk "BEGIN{print $W_P_V5+$W_COV_V5+$W_CONS_V5+$W_F_V5}"))"
echo "[v6-instruct] OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE"

# ── eval 阶段参数 (可在运行前 export 覆盖) ───────────────────────────────────
RUN_EVAL="${RUN_EVAL:-1}"                 # 0 = 只训练不 eval
EVAL_NUM_TEST="${EVAL_NUM_TEST:-1000}"   # BO1 eval 实例数; 1000=与 optimal/PyVRP 冻结集 seed=9999 逐一对齐 (FOARL 受控对比口径); smoke 可 100
EVAL_TP="${EVAL_TP:-1}"                   # eval tensor parallel 卡数 (BO1 贪心很轻, 1 张足够)
EVAL_GPU="${EVAL_GPU:-}"                  # 留空 → 训练完自动挑空闲卡; 也可手动 "0" / "0,1"
EVAL_GPU_MEM="${EVAL_GPU_MEM:-0.8}"      # vLLM 显存比例 (留余量给 CUDA graph)
SCKEY="${SCKEY:-SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp}"
_notify() { curl -s --max-time 10 "https://sctapi.ftqq.com/$SCKEY.send" \
    --data-urlencode "title=${1:0:100}" --data-urlencode "desp=${2:0:500}" >/dev/null 2>&1 || true; }

# ════════════════════════════════════════════════════════════════════════════
echo "[v6-instruct] ========== [1/3] 训练 ==========  $(date '+%F %T')"
# run_grpo_cvrp20_v6.sh: 兜底设 REWARD_SCHEME=v6 + vLLM 端口 8004, 再 exec run_grpo_cvrp20_v5.sh
# 注意: 不用 exec, 训练返回后接 merge + BO1 eval。set -e 下用 set +e 包住以捕获退出码。
set +e
bash "$_D/run_grpo_cvrp20_v6.sh"
TRAIN_EC=$?
set -e

if [ "$TRAIN_EC" -ne 0 ]; then
    echo "[v6-instruct][FATAL] 训练失败 (exit=$TRAIN_EC), 跳过 merge+eval"
    exit "$TRAIN_EC"
fi
if [ "$RUN_EVAL" != "1" ]; then
    echo "[v6-instruct] RUN_EVAL=$RUN_EVAL → 训练已完成, 跳过 merge+BO1 eval"
    exit 0
fi

# OUTPUT_DIR_BASE 是绝对路径 (本脚本上面已 export); run tag 目录 = cvrp_n20 (train.py)
OUT_DIR="$OUTPUT_DIR_BASE/cvrp_n20"
ADAPTER="$OUT_DIR/final_model"          # GRPO LoRA adapter
MERGED="$OUT_DIR/merged_model"          # 合并后全量权重 (eval 吃这个)
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "[v6-instruct][FATAL] 训练号称成功但缺 LoRA adapter: $ADAPTER/adapter_config.json"
    _notify "❌ v6-instruct: 缺 adapter" "$ADAPTER"
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════
echo "[v6-instruct] ========== [2/3] merge LoRA → merged_model ==========  $(date '+%F %T')"
# merge_lora.py: 独立脚本 (base 从 adapter_config 读 = instruct SFT 产物, 自动适配),
# low_cpu_mem_usage 降 CPU 峰值防 OOM kill, 内置权重非空校验 (CLAUDE.md 代码自审)。
# ⚠️ 历史 bug: 旧内联 heredoc 不落盘 → zhihan 无 SLURM, tmux 滚屏/断连后"merge 失败且无 log";
#    且 CPU OOM 时 SIGKILL(exit 137 无 traceback)是哑崩。现全部输出 tee 到 merge.log 可复盘。
MERGE_LOG="$OUT_DIR/merge.log"
if [ -d "$MERGED" ] && [ -f "$MERGED/config.json" ]; then
    echo "[v6-instruct] merged_model 已存在, 跳过合并: $MERGED"
else
    python "$_D/merge_lora.py" --adapter "$ADAPTER" --merged "$MERGED" 2>&1 | tee "$MERGE_LOG"
    MERGE_EC=${PIPESTATUS[0]}
    if [ "$MERGE_EC" -ne 0 ]; then
        echo "[v6-instruct][FATAL] merge 失败 (exit=$MERGE_EC, 137=CPU OOM被kill), 详见 $MERGE_LOG"
        tail -n 20 "$MERGE_LOG" 2>/dev/null || true
        _notify "❌ v6-instruct: merge 失败 exit=$MERGE_EC" "log: $MERGE_LOG"
        exit 1
    fi
fi
echo "[v6-instruct] ✓ merge 完成 (校验见 $MERGE_LOG): $MERGED"

# ════════════════════════════════════════════════════════════════════════════
echo "[v6-instruct] ========== [3/3] BO1 eval (greedy, model_type=reasoning prompt_mode=think) ==========  $(date '+%F %T')"
# 自动挑空闲卡 (训练刚释放); 留空时扫 nvidia-smi, 最多重试 ~30s 等 vLLM 完全退显存
if [ -z "$EVAL_GPU" ]; then
    for _try in 1 2 3 4 5 6; do
        _free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
            | awk -F',' '{gsub(/ /,"",$1);gsub(/ /,"",$2)} $2+0>=22528 {print $1}')
        _arr=($_free)
        [ "${#_arr[@]}" -ge "$EVAL_TP" ] && break
        echo "[v6-instruct] 等空闲卡 ($_try/6): 当前 free≥22.5G 仅 ${#_arr[@]} 张, 需 $EVAL_TP ..."
        sleep 5
    done
    if [ "${#_arr[@]}" -lt "$EVAL_TP" ]; then
        echo "[v6-instruct][FATAL] eval 需 $EVAL_TP 张空闲卡, 仅 ${#_arr[@]} 张; 手动指定 EVAL_GPU=... 重跑 (训练已存档, 不必重训)"
        exit 1
    fi
    EVAL_GPU="$(IFS=,; echo "${_arr[*]:0:$EVAL_TP}")"
fi
echo "[v6-instruct] eval GPU=$EVAL_GPU TP=$EVAL_TP NUM_TEST=$EVAL_NUM_TEST"

# run_eval_matrix.sh: ONLY=RL + DO_BO1=1/DO_BO8WAVE=0 → 仅对 RL merged 跑 BO1
#   它内部硬编码 --prompt_mode think --model_type reasoning (与 SFT bo1 一致, instruct 正确口径),
#   --num_samples 1 = greedy。BO1-only 不需要 POMO ckpt。结果 → eval_results_matrix/v6_instruct_fw_RL_BO1.json
set +e
RL_MODEL="$MERGED" \
ONLY=RL DO_BO1=1 DO_BO8WAVE=0 \
NUM_TEST="$EVAL_NUM_TEST" \
GPU="$EVAL_GPU" TP="$EVAL_TP" GPU_MEM="$EVAL_GPU_MEM" \
MAXLEN_RL=6144 \
RUN_PREFIX="${RUN_PREFIX:-v6_instruct_fw_}" \
    bash "$_D/run_eval_matrix.sh"
EVAL_EC=$?
set -e

_RP="${RUN_PREFIX:-v6_instruct_fw_}"
if [ "$EVAL_EC" -eq 0 ]; then
    echo "[v6-instruct] ✅ 全流程完成 (train→merge→BO1)  $(date '+%F %T')"
    echo "[v6-instruct] 结果: eval_results_matrix/${_RP}RL_BO1.json  日志: eval_logs_matrix/${_RP}RL_BO1.log"
    _notify "✅ v6-instruct train+merge+BO1 完成" "结果 eval_results_matrix/${_RP}RL_BO1.json"
else
    echo "[v6-instruct] ⚠️ BO1 eval 非零退出 (exit=$EVAL_EC), 详见 eval_logs_matrix/${_RP}RL_BO1.log"
    _notify "⚠️ v6-instruct BO1 非零退出" "exit=$EVAL_EC"
fi
exit "$EVAL_EC"
