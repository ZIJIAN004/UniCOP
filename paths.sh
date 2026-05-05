#!/bin/bash
# paths.sh — 统一路径配置，按主机自动切换
# 所有自动化脚本开头 source 此文件，不再硬编码路径
#
# 检测逻辑：通过工作目录存在性判断当前主机
#   /home/ntu/lzj/          → NTU 主机
#   /Data04/yangzhihan/lzj/ → A*STAR 主机

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
    # ── 模型 ──
    BASE_MODEL="/home/ntu/lzj/Model/model/DeepSeek-R1-Distill-Qwen-7B"

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
    # ── 模型 ──
    BASE_MODEL="/Data04/yangzhihan/lzj/UniCOP-Reason.bak_/model/deepseek-reasoning/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

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
    LKH_BIN="/homes/zhuoyi/zijianliu/LKH-3.0.9/LKH"
    # ── 模型 ──
    BASE_MODEL="/homes/zhuoyi/zijianliu/models/DeepSeek-R1-Distill-Qwen-7B"

else
    echo "❌ 无法识别当前主机（/home/ntu/lzj、/Data04/yangzhihan/lzj、/homes/zhuoyi/zijianliu 均不存在）"
    exit 1
fi

# ── 派生路径（两台主机共用逻辑） ──
REASON_DIR="$UNICOP_ROOT/UniCOP-Reason"
DISTILL_DIR="$UNICOP_ROOT/UniCOP-Distill"

export HOST_ID UNICOP_ROOT REASON_DIR DISTILL_DIR
export POMO_BASELINE_DIR POMO_CKPT_DIR PIPD_DIR PIPD_CKPT_DIR
export CUDA_HOME LKH_BIN BASE_MODEL
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
