#!/bin/bash
# Iteration-3 stage A: reconstruction-weight sweep for the hybrid objective.
set -e
cd "$(dirname "$0")/.."
PY=.pixi/envs/default/bin/python
LOGS=results/benchmarks/logs

for w in 0.3 1.0 3.0; do
    tag=$(echo "$w" | tr -d '.')
    echo "=== hybrid w=$w ==="
    $PY benchmarks/train_tokenizers.py --model hybrid --data aoj --epochs 15 \
        --recon-weight "$w" --run-name "hybrid_w${tag}" \
        > "$LOGS/train_hybrid_w${tag}.log" 2>&1
    echo "hybrid w=$w OK"
done
echo "=== STAGE A DONE ==="
