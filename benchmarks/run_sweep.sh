#!/bin/bash
# Fair-comparison seed sweep: library seeds 43,44 + vqvae_particle seeds 42-44
# (15 epochs each on AOJ), then the jets5c library consumer probe.
set -e
cd "$(dirname "$0")/.."
PY=.pixi/envs/default/bin/python
LOGS=results/benchmarks/logs
mkdir -p "$LOGS"

rm -rf results/benchmarks/tokenizers/vqp_smoke

for spec in "library 43" "library 44" "vqvae_particle 42" "vqvae_particle 43" "vqvae_particle 44"; do
    set -- $spec
    m=$1; s=$2
    echo "=== $m seed $s ==="
    $PY benchmarks/train_tokenizers.py --model "$m" --data aoj --seed "$s" --epochs 15 \
        > "$LOGS/train_${m}_aoj_s${s}.log" 2>&1
    echo "$m seed $s OK"
done

echo "=== jets5c library probe ==="
$PY benchmarks/train_probes.py --arms library --epochs 15 > "$LOGS/probe_library.log" 2>&1
echo "probe OK"
echo "=== SWEEP DONE ==="
