#!/bin/bash
# preflight.sh — UniCOP-Reason GRPO 训练前全面 sanity check
#
# 用法:   bash preflight.sh
# 退出码:  0 = 全部通过, 1 = 有 FAIL
#
# 7 级检查:
#   Level 1: 路径 & 目录
#   Level 2: CUDA_HOME / nvcc
#   Level 3: TRL CLI binary
#   Level 4: POMO ckpts (TSP/CVRP/VRPTW × 当前矩阵的 size)
#   Level 5: PIP-D TSPTW ckpts (n=50, 100 硬查; n=20 warn)
#   Level 6: Python 包 (trl/vllm/transformers/... 全导入)
#   Level 7: 实例化 POMOPRM + PIP-D 跑一次 foresight + rollout

set +e  # 单项失败不立刻退出,累积报告

FAIL=0
WARN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/paths.sh"

WORK_DIR="$REASON_DIR"
MODEL_BASE=$(ls -d "$DISTILL_DIR/output/*/final_model" 2>/dev/null | sort -r | head -1)
if [ -z "$MODEL_BASE" ]; then
    MODEL_BASE="$DISTILL_DIR/output/NOT_FOUND/final_model"
fi
echo "[MODEL_BASE] $MODEL_BASE"
CUDA_HOME_PATH="$CUDA_HOME"

# 当前训练矩阵: 自动从 auto_train.sh 读取,避免两边手动对齐
AUTO_TRAIN_SH="$(dirname "$0")/auto_train.sh"
if [ -f "$AUTO_TRAIN_SH" ]; then
    # 用 grep + eval 提取 PROBLEMS=(...) 和 SIZES=(...) 两行
    eval "$(grep -E '^PROBLEMS=\(' "$AUTO_TRAIN_SH" | head -1)"
    eval "$(grep -E '^SIZES=\('    "$AUTO_TRAIN_SH" | head -1)"
    echo "[INFO] 从 auto_train.sh 读取训练矩阵: PROBLEMS=(${PROBLEMS[*]}) SIZES=(${SIZES[*]})"
else
    echo "[WARN] $AUTO_TRAIN_SH 不存在,回退 ('tsp' / 20)"
    PROBLEMS=("tsp")
    SIZES=(20)
fi

check_dir() {
    if [ -d "$1" ]; then
        echo "  [OK  ] $2"
    else
        echo "  [FAIL] $2 不存在: $1"
        FAIL=$((FAIL+1))
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK  ] $2"
    else
        echo "  [FAIL] $2 不存在: $1"
        FAIL=$((FAIL+1))
    fi
}

check_file_warn() {
    if [ -f "$1" ]; then
        echo "  [OK  ] $2"
    else
        echo "  [WARN] $2 不存在: $1"
        WARN=$((WARN+1))
    fi
}

# MODEL_BASE 这种训练产物在 .gitignore 里被排除,git clone 后不存在是预期的。
# 自动尝试从 UniCOP-Distill/output/*/final_model 建立软链,
# 让路径在 monorepo 里一致,不需要手动 ln -s。
check_sft_product_with_bak() {
    local target="$1"   # 例: .../UniCOP/UniCOP-Distill/output_sft_r1_v2/final_model
    local desc="$2"

    if [ -d "$target" ]; then
        echo "  [OK  ] $desc"
        return 0
    fi

    # 推导路径:
    #   link_point = .../UniCOP/UniCOP-Distill/output_sft_r1_v2  (软链要建在这里)
    #   jlz_dir    = .../lzj                                      (UniCOP-Distill.bak_* 在这)
    local link_point
    link_point=$(dirname "$target")
    local monorepo_distill
    monorepo_distill=$(dirname "$link_point")
    local jlz_dir
    jlz_dir=$(dirname "$(dirname "$monorepo_distill")")

    local bak_candidate
    bak_candidate=$(ls -d "${jlz_dir}/UniCOP-Distill.bak_"*/output_sft_r1_v2 2>/dev/null | head -1)

    if [ -z "$bak_candidate" ] || [ ! -d "$bak_candidate" ]; then
        echo "  [FAIL] $desc 不存在: $target"
        echo "         (且 ${jlz_dir}/UniCOP-Distill.bak_*/output_sft_r1_v2 也找不到)"
        FAIL=$((FAIL+1))
        return 1
    fi

    # link_point 不存在 → 建软链
    if [ ! -e "$link_point" ]; then
        ln -s "$bak_candidate" "$link_point"
        echo "  [AUTO] 建立软链: $link_point -> $bak_candidate"
    elif [ -L "$link_point" ]; then
        # 已是软链但指向有问题
        echo "  [INFO] $link_point 已是软链但 target 不可达,跳过重建"
    else
        # 是真实目录但里面没 target (final_model / merged_model)
        echo "  [WARN] $link_point 存在但缺 $(basename "$target") 子目录"
    fi

    # 再检一次
    if [ -d "$target" ]; then
        echo "  [OK  ] $desc (通过 bak 软链恢复)"
        return 0
    else
        echo "  [FAIL] $desc 建立软链后仍找不到: $target"
        FAIL=$((FAIL+1))
        return 1
    fi
}

echo "==================================================="
echo "Level 1: 核心路径"
echo "==================================================="
check_dir "$WORK_DIR" "WORK_DIR"
check_sft_product_with_bak "$MODEL_BASE" "MODEL_BASE (SFT 产物)"
check_dir "$POMO_BASELINE_DIR" "POMO_BASELINE_DIR"
check_dir "$POMO_CKPT_DIR" "POMO_CKPT_DIR"
check_dir "$PIPD_DIR" "PIPD_DIR"
check_dir "$PIPD_CKPT_DIR" "PIPD_CKPT_DIR"

echo ""
echo "==================================================="
echo "Level 2: CUDA_HOME / nvcc"
echo "==================================================="
check_file "$CUDA_HOME_PATH/bin/nvcc" "nvcc (CUDA_HOME)"

echo ""
echo "==================================================="
echo "Level 3: TRL CLI binary (当前 env)"
echo "==================================================="
TRL_BIN="$(dirname "$(which python)")/trl"
if [ -x "$TRL_BIN" ]; then
    echo "  [OK  ] TRL binary: $TRL_BIN"
    "$TRL_BIN" vllm-serve --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  [OK  ] trl vllm-serve --help 能跑"
    else
        echo "  [FAIL] trl vllm-serve --help 报错"
        FAIL=$((FAIL+1))
    fi
else
    echo "  [FAIL] TRL binary 不存在: $TRL_BIN"
    echo "         请在当前 env 装: pip install 'trl[vllm]==1.1.0'"
    FAIL=$((FAIL+1))
fi

echo ""
echo "==================================================="
echo "Level 4: POMO ckpts (TSP/CVRP/VRPTW)"
echo "==================================================="
for problem in "${PROBLEMS[@]}"; do
    if [ "$problem" == "tsptw" ]; then
        continue   # TSPTW 走 PIP-D,Level 5 单独查
    fi
    for size in "${SIZES[@]}"; do
        problem_upper=$(echo "$problem" | tr '[:lower:]' '[:upper:]')
        candidates=$(ls -d ${POMO_CKPT_DIR}/*POMO_${problem_upper}_n${size} 2>/dev/null | head -1)
        if [ -z "$candidates" ]; then
            echo "  [FAIL] ${problem_upper} n=${size}: 无匹配 *POMO_${problem_upper}_n${size} 目录"
            FAIL=$((FAIL+1))
        else
            if [ -f "$candidates/MODEL_FINAL.pt" ]; then
                echo "  [OK  ] ${problem_upper} n=${size}"
            else
                echo "  [FAIL] ${problem_upper} n=${size}: 目录存在但缺 MODEL_FINAL.pt"
                FAIL=$((FAIL+1))
            fi
        fi
    done
done

echo ""
echo "==================================================="
echo "Level 5: PIP-D TSPTW ckpts"
echo "==================================================="
# 本训练矩阵里要用 TSPTW 吗?
if [[ " ${PROBLEMS[*]} " =~ " tsptw " ]]; then
    for size in "${SIZES[@]}"; do
        n_total=$((size + 1))
        ckpt="${PIPD_CKPT_DIR}/tsptw${n_total}_easy/POMO_star_PIP-D/epoch-10000.pt"
        if [ "$size" == "20" ]; then
            # n=20 ckpt 还在训练,允许 warn
            check_file_warn "$ckpt" "PIP-D TSPTW n=20 (ckpt 名: tsptw21_easy)"
        else
            check_file "$ckpt" "PIP-D TSPTW n=${size} (ckpt 名: tsptw${n_total}_easy)"
        fi
    done
else
    echo "  [SKIP] 训练矩阵中无 tsptw,跳过 PIP-D 检查"
fi

echo ""
echo "==================================================="
echo "Level 6: Python 包"
echo "==================================================="
export CUDA_HOME="$CUDA_HOME_PATH"
python - <<'EOF'
import sys
packages = ['trl', 'vllm', 'transformers', 'accelerate', 'deepspeed', 'peft', 'torch', 'numpy']
fail = 0
for p in packages:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', '?')
        print(f"  [OK  ] {p:14s} {ver}")
    except Exception as e:
        print(f"  [FAIL] {p}: {e}")
        fail += 1
sys.exit(fail)
EOF
if [ $? -ne 0 ]; then
    FAIL=$((FAIL+1))
fi

echo ""
echo "==================================================="
echo "Level 7a: UniCOP-Reason 核心模块 import + POMOPRM 实例化"
echo "==================================================="
cd "$WORK_DIR" || exit 1
export CUDA_HOME="$CUDA_HOME_PATH"
python - <<EOF
import sys
sys.path.insert(0, ".")
try:
    from pomo_prm import POMOPRM
    from grpo_prm_trainer import GRPOPRMTrainer
    from terminal_reward import compute_terminal_reward
    from utils.pipd_wrapper import PIPDWrapper, _resolve_pipd_dir
    from utils.vllm_ngram_processor import NoRepeatNgramAdapterLP
    from config import config
    print("  [OK  ] 核心模块全部可 import (含 vllm_ngram_processor)")
except Exception as e:
    print(f"  [FAIL] 模块 import 失败: {e}")
    sys.exit(1)

# 验证 vLLM logits_processors entry point 已注册
# (走 pyproject.toml + pip install -e . 路径,vLLM 启动时自动 discover)
try:
    from importlib.metadata import entry_points
    eps = list(entry_points(group="vllm.logits_processors"))
    names = [ep.name for ep in eps]
    values = [ep.value for ep in eps]
    if "no_repeat_ngram" in names:
        idx = names.index("no_repeat_ngram")
        print(f"  [OK  ] vllm.logits_processors entry point 已注册: {names[idx]} -> {values[idx]}")
    else:
        print(f"  [FAIL] vllm.logits_processors entry point 未注册 (当前: {names or '空'})")
        print(f"         修复: cd $(pwd) && pip install -e .")
        sys.exit(1)
except Exception as e:
    print(f"  [FAIL] entry point 检查异常: {e}")
    sys.exit(1)

# 实例化 POMOPRM (不会触发 ckpt 实际加载,懒加载在 _get_model 里)
try:
    pomo_prm = POMOPRM(
        pomo_ckpt_dir="${POMO_CKPT_DIR}",
        pomo_baseline_dir="${POMO_BASELINE_DIR}",
        device="cuda",
        pipd_ckpt_dir="${PIPD_CKPT_DIR}",
        pipd_dir="${PIPD_DIR}",
    )
    print("  [OK  ] POMOPRM 实例化")
except Exception as e:
    print(f"  [FAIL] POMOPRM 实例化失败: {e}")
    sys.exit(1)
EOF
if [ $? -ne 0 ]; then
    FAIL=$((FAIL+1))
fi

echo ""
echo "==================================================="
echo "Level 7b: PIP-D TSPTW 端到端测试"
echo "==================================================="
if [[ " ${PROBLEMS[*]} " =~ " tsptw " ]]; then
    python - <<EOF
import sys
sys.path.insert(0, ".")
from pomo_prm import POMOPRM

pomo_prm = POMOPRM(
    pomo_ckpt_dir="${POMO_CKPT_DIR}",
    pomo_baseline_dir="${POMO_BASELINE_DIR}",
    device="cuda",
    pipd_ckpt_dir="${PIPD_CKPT_DIR}",
    pipd_dir="${PIPD_DIR}",
)

try:
    import numpy as np
    import torch
    wrapper = pomo_prm._get_model("tsptw", 50)
    print("  [OK  ] PIP-D TSPTW n=50 模型加载 (aux decoder 权重检查通过)")

    # 随机实例
    rng = np.random.default_rng(42)
    n = 50
    tn = 26.6
    coords = rng.uniform(0, 1, size=(n+1, 2))
    tw = np.zeros((n+1, 2))
    tw[0] = [0, 1e9]
    l = rng.uniform(0, tn, n)
    width = (0.5 + 0.25 * rng.uniform(size=n)) * tn
    tw[1:, 0] = l
    tw[1:, 1] = l + width
    instance = {"n": n, "coords": coords, "time_windows": tw}

    prefix = [0, 3, 7, 1, 5, 10, 20, 30, 40, 2]
    fs_idx = wrapper.foresight_check(instance, prefix)
    print(f"  [OK  ] foresight_check 跑通, 返回 fs_idx={fs_idx}")

    values = wrapper.batch_rollout(instance, valid_prefix=prefix, prefix_lengths=[3, 6, 10])
    print(f"  [OK  ] batch_rollout 跑通, values={[round(v,3) for v in values]}")
    if all(v < 0 for v in values):
        print("  [OK  ] rollout values 全为负数 (正常)")
    else:
        print(f"  [WARN] rollout values 有非负值,可疑: {values}")

except Exception as e:
    import traceback
    print(f"  [FAIL] PIP-D 端到端测试失败:\n{traceback.format_exc()}")
    sys.exit(1)
EOF
    if [ $? -ne 0 ]; then
        FAIL=$((FAIL+1))
    fi
else
    echo "  [SKIP] 训练矩阵无 tsptw (PROBLEMS=${PROBLEMS[*]}),跳过 PIP-D 端到端测试"
fi

echo ""
echo "==================================================="
echo "总结"
echo "==================================================="
echo "  FAIL: $FAIL"
echo "  WARN: $WARN"
echo ""
if [ $FAIL -eq 0 ]; then
    echo "✅ 所有硬性检查通过,可以启动训练"
    if [ $WARN -gt 0 ]; then
        echo "⚠️  注意 $WARN 项 WARN (通常是 n=20 PIP-D ckpt 还没训完,TSPTW n=20 任务会跑不了)"
    fi
    exit 0
else
    echo "❌ 有 $FAIL 项硬性失败,必须先修"
    exit 1
fi
