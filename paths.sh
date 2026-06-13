#!/bin/bash
# paths.sh — 统一路径配置，按主机自动切换
# 所有自动化脚本开头 source 此文件，不再硬编码路径
#
# 检测逻辑：通过工作目录存在性判断当前主机
#   /home/ntu/lzj/          → NTU 主机
#   /Data04/yangzhihan/lzj/ → A*STAR 主机
#
# 基座模型选择（通过 BASE_MODEL_TYPE 环境变量）：
#   r1_distill (默认)  → DeepSeek-R1-Distill-Qwen-7B
#   qwen3_thinking     → Qwen3-4B-Thinking-2507
#   qwen3_instruct     → Qwen3-4B-Instruct-2507 (非 thinking, 对齐 FOARL instruct 范式)
# 用法：
#   BASE_MODEL_TYPE=qwen3_thinking bash auto_self_rationalize.sh

BASE_MODEL_TYPE="${BASE_MODEL_TYPE:-r1_distill}"

if [ -d "/home/ntu/lzj" ]; then
    HOST_ID="ntu"
    # ── 项目 ──
    UNICOP_ROOT="/home/ntu/lzj/UniCOP"
    POMO_BASELINE_DIR="/home/ntu/lzj/POMO-Baseline"
    POMO_CKPT_DIR="$POMO_BASELINE_DIR/result"
    PIPD_DIR="/home/ntu/lzj/PIP-D baseline/POMO+PIP"
    PIPD_CKPT_DIR="$PIPD_DIR/pretrained/TSPTW"
    # ── 环境 ──
    CUDA_HOME="/home/ntu/anaconda3/envs/unicop"
    LKH_BIN="/home/ntu/LKH/LKH"
    # ── 模型路径（两套候选） ──
    BASE_MODEL_R1="/home/ntu/lzj/Model/model/DeepSeek-R1-Distill-Qwen-7B"
    BASE_MODEL_QWEN3="/home/ntu/lzj/Model/model/Qwen3-4B-Thinking-2507"
    BASE_MODEL_QWEN3_INSTRUCT="/home/ntu/lzj/Model/model/Qwen3-4B-Instruct-2507"

elif [ -d "/Data04/yangzhihan/lzj" ]; then
    HOST_ID="astar-zhihan"
    # ── 项目 ──
    UNICOP_ROOT="/Data04/yangzhihan/lzj/UniCOP"
    POMO_BASELINE_DIR="/Data04/yangzhihan/lzj/POMO-Baseline"
    POMO_CKPT_DIR="$POMO_BASELINE_DIR/result"
    PIPD_DIR="/Data04/yangzhihan/lzj/PIP-D baseline"
    PIPD_CKPT_DIR=""  # 待补充
    # ── 环境 ──
    CUDA_HOME="/Data04/yangzhihan/envs/unicop"
    LKH_BIN="/Data04/yangzhihan/lzj/LKH-3.0.9/LKH"
    # ── 模型路径（两套候选） ──
    BASE_MODEL_R1="/Data04/yangzhihan/lzj/UniCOP-Reason.bak_/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    BASE_MODEL_QWEN3="/Data04/yangzhihan/lzj/model/Qwen3-4B-Thinking-2507"
    BASE_MODEL_QWEN3_INSTRUCT="/Data04/yangzhihan/lzj/model/Qwen3-4B-Instruct-2507"

elif [ -d "/homes/zhuoyi/zijianliu" ]; then
    HOST_ID="astar-zhuoyi"
    # ── 项目 ──
    UNICOP_ROOT="/homes/zhuoyi/zijianliu/UniCOP"
    POMO_BASELINE_DIR="/homes/zhuoyi/zijianliu/POMO-Baseline"
    POMO_CKPT_DIR="$POMO_BASELINE_DIR/result"
    PIPD_DIR="/homes/zhuoyi/zijianliu/PIP-D baseline"
    PIPD_CKPT_DIR=""  # TODO
    # ── 环境 ──
    CUDA_HOME="/homes/zhuoyi/miniforge3/envs/unicop"
    LKH_BIN="/homes/zhuoyi/zijianliu/LKH-3.0.13/LKH"
    # ── 模型路径（两套候选） ──
    BASE_MODEL_R1="/homes/zhuoyi/zijianliu/models/DeepSeek-R1-Distill-Qwen-7B"
    BASE_MODEL_QWEN3="/homes/zhuoyi/zijianliu/models/Qwen3-4B-Thinking-2507"
    BASE_MODEL_QWEN3_INSTRUCT="/homes/zhuoyi/zijianliu/models/Qwen3-4B-Instruct-2507"

else
    echo "❌ 无法识别当前主机（/home/ntu/lzj、/Data04/yangzhihan/lzj、/homes/zhuoyi/zijianliu 均不存在）"
    exit 1
fi

# ── 根据 BASE_MODEL_TYPE 派生基座模型路径和推理配置 ──
# vLLM 0.6+ 默认会加载仓库 generation_config.json 并以其值覆盖客户端传的采样参数,
# 加 --generation-config vllm 让客户端传值生效(两个模型都加,以确保 paths.sh 控制权)。
_VLLM_COMMON_FLAGS="--generation-config vllm"

case "$BASE_MODEL_TYPE" in
    r1_distill)
        BASE_MODEL="$BASE_MODEL_R1"
        # R1-Distill 无需 reasoning-parser(沿用原代码行为, content 含完整 <think>...</think>)
        VLLM_REASONING_FLAGS="$_VLLM_COMMON_FLAGS"
        # 采样参数(R1-Distill 用 t=1.0 是当前代码现状, 保留)
        GEN_TEMPERATURE="1.0"
        GEN_TOP_P="1.0"
        GEN_TOP_K="-1"
        # Stage 1 训练: 沿用原逻辑(剥 <think>, completion=solution)
        STAGE1_KEEP_THINK="false"
        ;;
    qwen3_thinking)
        BASE_MODEL="$BASE_MODEL_QWEN3"
        # Qwen3-4B-Thinking-2507 官方推荐 --reasoning-parser deepseek_r1
        # (model card 原文: `--enable-reasoning --reasoning-parser deepseek_r1`)。
        # 原因: 2507 系列的 chat_template 自动 prepend <think>, 模型只输出
        # </think>...content, 跟 R1 行为一致, qwen3 parser 是给老 Qwen3 base
        # (有 <think> open tag 自己输出) 用的, 2507 用 qwen3 parser 会解析失败。
        # 启用后 thinking 段进 message.reasoning_content, 最终答案进 message.content,
        # rationalize_solutions.py 的 call_vllm 已适配。
        VLLM_REASONING_FLAGS="$_VLLM_COMMON_FLAGS --reasoning-parser deepseek_r1"
        # Qwen3-4B-Thinking-2507 官方推荐采样参数
        GEN_TEMPERATURE="0.6"
        GEN_TOP_P="0.95"
        GEN_TOP_K="20"
        # Stage 1 训练: Qwen3-Thinking 的 chat_template 强制 prepend <think>\n,
        # 训练时若剥 <think>, 推理时模型会遇到 OOD 前缀分布(assistant\n<think>\n + ?)。
        # 改为不剥 <think>, completion 以 </think>\n\n 开头, 形成空 think 占位:
        # 完整序列 = ...assistant\n<think>\n  +  </think>\n\n{solution}<eos>
        STAGE1_KEEP_THINK="true"
        ;;
    qwen3_instruct)
        BASE_MODEL="$BASE_MODEL_QWEN3_INSTRUCT"
        # Qwen3-4B-Instruct-2507: 非 thinking instruct 模型, chat_template 末尾不带 <think>。
        # 我们方法的 Stage2 SFT (train_sft_stage2.py) 已有 probe_ends_with_think=False 分支:
        # 自动手动补 <think>\n, 教模型输出结构化路径构造链, 推理时模型自行输出
        # <think>...</think>route。无需改训练代码。
        # 注: 本 SFT+eval 流程的 eval 走 HF local backend (evaluate.py --backend local),
        #     不经 vLLM serve, 故此处 reasoning-parser 暂不设。若后续 RL 阶段要 vLLM serve,
        #     parser 需单独验证 (instruct 自行输出 open <think>, 行为接近老 Qwen3 base,
        #     可能需 --reasoning-parser qwen3 而非 deepseek_r1)。
        VLLM_REASONING_FLAGS="$_VLLM_COMMON_FLAGS"
        # Qwen3-4B-Instruct-2507 官方推荐采样参数, 同时对齐 FOARL 受控对比采样(0.7/0.8/20)
        GEN_TEMPERATURE="0.7"
        GEN_TOP_P="0.8"
        GEN_TOP_K="20"
        # 我们方法走 Stage2; instruct chat_template 无强制 <think>, STAGE1_KEEP_THINK 置 false
        STAGE1_KEEP_THINK="false"
        ;;
    *)
        echo "❌ 未知 BASE_MODEL_TYPE='$BASE_MODEL_TYPE'，应为 r1_distill / qwen3_thinking / qwen3_instruct"
        exit 1
        ;;
esac

# ── 派生路径（两台主机共用逻辑） ──
REASON_DIR="$UNICOP_ROOT/UniCOP-Reason"
DISTILL_DIR="$UNICOP_ROOT/UniCOP-Distill"
MASK_DIR="$UNICOP_ROOT/UniCOP-Reason-Mask"   # constrained-decoding 实验项目

export HOST_ID UNICOP_ROOT REASON_DIR DISTILL_DIR MASK_DIR
export POMO_BASELINE_DIR POMO_CKPT_DIR PIPD_DIR PIPD_CKPT_DIR
export CUDA_HOME LKH_BIN
export BASE_MODEL_TYPE BASE_MODEL BASE_MODEL_R1 BASE_MODEL_QWEN3 BASE_MODEL_QWEN3_INSTRUCT
export VLLM_REASONING_FLAGS GEN_TEMPERATURE GEN_TOP_P GEN_TOP_K STAGE1_KEEP_THINK
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# ── CUDA dev 头/库永久补全(让 DeepSpeed/torch JIT 编译 CUDA 扩展能找到头) ──────────
# conda env 把 CUDA 头/库放在 $CUDA_HOME/targets/x86_64-linux/{include,lib}, 而 torch 默认
# 只看 $CUDA_HOME/include → FusedAdam 等编译时报 "cuda_runtime.h: No such file or directory"。
# 这里把真目录永久加进 编译器(CPATH)/链接器(LIBRARY_PATH)/运行时(LD_LIBRARY_PATH) 搜索路径。
# 用 [ -d ] 守卫: 目录存在才加, 没有 targets 布局的主机/env 上是无害 no-op。不需换 analog_env。
_CUDA_TARGETS="$CUDA_HOME/targets/x86_64-linux"
if [ -d "$_CUDA_TARGETS/include" ]; then
    export CPATH="$_CUDA_TARGETS/include:${CPATH:-}"
    export LIBRARY_PATH="$_CUDA_TARGETS/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$_CUDA_TARGETS/lib:$LD_LIBRARY_PATH"
fi
