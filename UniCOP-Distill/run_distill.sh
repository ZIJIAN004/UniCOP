#!/bin/bash
set -e

WORK_DIR="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill"
CREDENTIALS="/Data04/yangzhihan/lzj/UniCOP/UniCOP-Distill/advance-subject-493905-h9-020e2dc30ae7.json"
PROJECT="advance-subject-493905-h9"
LKH_BIN="/Data04/yangzhihan/lzj/LKH-3.0.9/LKH"
SEED=2027
RAW_OUTPUT="data/chains_v4.jsonl"
CLEAN_OUTPUT="data/chains_v4_clean.jsonl"

cd "$WORK_DIR"

echo "===== [$(date '+%F %T')] Step 1/2: generate chains ====="
python generate_chains.py \
    --credentials "$CREDENTIALS" \
    --project "$PROJECT" \
    --lkh_bin "$LKH_BIN" \
    --problems tsp cvrp tsptw vrptw \
    --sizes 20 50 100 \
    --num_samples 400 \
    --seed "$SEED" \
    --concurrency 4 \
    --output "$RAW_OUTPUT"

echo "===== [$(date '+%F %T')] Step 2/2: clean chains ====="
python clean_single.py \
    --input "$RAW_OUTPUT" \
    --output "$CLEAN_OUTPUT"

echo "===== [$(date '+%F %T')] All done. Output: $CLEAN_OUTPUT ====="
