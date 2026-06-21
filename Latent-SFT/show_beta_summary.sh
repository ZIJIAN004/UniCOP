#!/bin/bash
# 汇总 HLR β sweep 结果 (含 β=1.0)
#
# 用法 (所有 job 跑完后, 在 zhuoyi 登录节点):
#   bash Latent-SFT/show_beta_summary.sh
#
# 默认从 output_hlr_beta{0.5,1.0,1.5,2.0}/ 读 compare.json

BETAS="${BETAS:-0.5 1.0 1.5 2.0}"
BASE_DIR="${BASE_DIR:-/homes/zhuoyi/zijianliu/UniCOP/Latent-SFT}"

echo "============================================================"
echo "  HLR β Sweep 汇总"
echo "============================================================"
echo ""
printf "  %-6s | %8s | %8s | %10s | %12s | %8s | %8s\n" \
    "β" "parse" "feas" "avg_dist" "CoT_tokens" "HLR_wall" "base_wall"
printf "  %-6s-|-%8s-|-%8s-|-%10s-|-%12s-|-%8s-|-%8s\n" \
    "------" "--------" "--------" "----------" "------------" "--------" "--------"

for BETA in $BETAS; do
    COMPARE_JSON="${BASE_DIR}/output_hlr_beta${BETA}/checkpoint-final/compare_eval/compare.json"
    if [ -f "$COMPARE_JSON" ]; then
        python -c "
import json
with open('$COMPARE_JSON') as f:
    d = json.load(f)
hlr = d.get('hlr', [{}])
b   = d.get('baseline', [{}])
if isinstance(hlr, list) and hlr: hlr = hlr[0]
if isinstance(b,   list) and b:   b   = b[0]
pr  = hlr.get('parse_rate', 0)
fr  = hlr.get('global_feas_rate', 0)
ad  = hlr.get('avg_instance_distance', 0)
cot_h = hlr.get('total_completion_tokens', 0)
cot_b = b.get('total_completion_tokens', 0)
wl_h = d.get('hlr_wall_seconds', 0)
wl_b = d.get('baseline_wall_seconds', 0)
print(f'  ${BETA:>5} | {pr:7.3f} | {fr:7.3f} | {ad:9.2f} | {cot_h:>6}/{cot_b:<6} | {wl_h:7.0f} | {wl_b:7.0f}')
"
    else
        echo "  ${BETA:>5} | ❌ 未完成 (无 compare.json)"
    fi
done

echo ""
echo "  全部 compare.json:"
for BETA in $BETAS; do
    COMP="${BASE_DIR}/output_hlr_beta${BETA}/checkpoint-final/compare_eval/compare.json"
    if [ -f "$COMP" ]; then
        echo "    ✓  β=${BETA}  $COMP"
    else
        echo "    ❌ β=${BETA}  (无文件)"
    fi
done
echo "============================================================"
