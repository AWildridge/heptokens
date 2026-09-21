#!/bin/bash
# Iteration-3 stage B at w*=2.0: interpolated weight, tighter dpT kernel, dim study.
set -e
cd "$(dirname "$0")/.."
PY=.pixi/envs/default/bin/python
LOGS=results/benchmarks/logs

run() {
    name=$1; shift
    echo "=== $name ==="
    $PY benchmarks/train_tokenizers.py --model hybrid --data aoj --epochs 15 \
        --recon-weight 2.0 --run-name "$name" "$@" > "$LOGS/train_${name}.log" 2>&1
    echo "$name OK"
}

run hybrid_w20
run hybrid_w20_dpt05 --dpt-frac 0.05
run hybrid_w20_d8 --latent-dim 8
run hybrid_w20_d4 --latent-dim 4
echo "=== STAGE B DONE ==="
