#!/bin/bash
# fix_tokenizer.sh — 修 merged_model 的 tokenizer_config.json
#
# 做三件事:
#   1. 备份 tokenizer_config.json → tokenizer_config.json.bak
#   2. 删掉非标准 tokenizer_class="TokenizersBackend" + auto_map 里相关条目
#   3. 用 Python 加载一次确认 tokenizer 能正常用
#
# 用法: bash fix_tokenizer.sh
# 如果已经修过也可以重跑,不会重复备份覆盖旧 .bak

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/paths.sh"

MERGED="$DISTILL_DIR/output/merged_model"
CFG="$MERGED/tokenizer_config.json"
BAK="$MERGED/tokenizer_config.json.bak"

echo "=================================================="
echo "Step 1: 检查文件存在"
echo "=================================================="
if [ ! -f "$CFG" ]; then
    echo "❌ $CFG 不存在"
    exit 1
fi
echo "  [OK] $CFG"

echo ""
echo "=================================================="
echo "Step 2: 备份 (只首次备份, 避免覆盖原始)"
echo "=================================================="
if [ ! -f "$BAK" ]; then
    cp "$CFG" "$BAK"
    echo "  [OK] 备份到 $BAK"
else
    echo "  [SKIP] $BAK 已存在 (首次备份保留)"
fi

echo ""
echo "=================================================="
echo "Step 3: 清理 tokenizer_class + auto_map"
echo "=================================================="
python <<PY
import json, sys
path = "$CFG"
with open(path) as f:
    cfg = json.load(f)

before_class = cfg.get('tokenizer_class')
before_auto_map = cfg.get('auto_map')
print(f"  原 tokenizer_class: {before_class}")
print(f"  原 auto_map: {before_auto_map}")

# 清 tokenizer_class
cfg.pop('tokenizer_class', None)

# 清 auto_map 里指向 TokenizersBackend 的条目
if isinstance(before_auto_map, dict):
    cleaned = {k: v for k, v in before_auto_map.items()
               if 'TokenizersBackend' not in str(v)}
    if cleaned:
        cfg['auto_map'] = cleaned
    else:
        cfg.pop('auto_map', None)

with open(path, 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print(f"  新 tokenizer_class: {cfg.get('tokenizer_class', '(deleted)')}")
print(f"  新 auto_map: {cfg.get('auto_map', '(deleted)')}")
print("  [OK] tokenizer_config.json 清理完成")
PY

echo ""
echo "=================================================="
echo "Step 4: 加载 tokenizer 验证"
echo "=================================================="
python <<PY
from transformers import AutoTokenizer
import sys

path = "$MERGED"
try:
    tok = AutoTokenizer.from_pretrained(path)
    print(f"  [OK] 加载成功")
    print(f"       class: {type(tok).__name__}")
    print(f"       vocab_size: {tok.vocab_size}")
    print(f"       chat_template: {'有' if tok.chat_template else '无'}")
    # 简单编码测试
    test = tok("<think>Hello</think>")
    print(f"       编码测试: {len(test['input_ids'])} tokens")
except Exception as e:
    print(f"  [FAIL] tokenizer 加载失败: {e}")
    sys.exit(1)
PY

echo ""
echo "=================================================="
echo "✅ 修复完成"
echo "=================================================="
echo ""
echo "下一步:"
echo "  1. pkill -9 -f vllm; pkill -9 -f trl; sleep 3   # 杀掉旧进程"
echo "  2. nvidia-smi    # 确认显存释放"
echo "  3. bash auto_train.sh   # 重启训练"
