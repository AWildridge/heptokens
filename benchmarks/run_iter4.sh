#!/bin/bash
# Iteration 4 — compressive tokens: criterion-shaped VQ, low-dim continuous,
# encoder-capacity ablation. All AOJ, 15 epochs.
#
# Idempotent + fault-tolerant: a run whose checkpoint already exists is skipped
# (safe to re-run to resume), and one failing run does not abort the rest.
# The single 8192 codebook needs a smaller batch — its one-hot is 8192 wide
# (~10 GB at batch 1024 on the 20 GB MIG); RVQ 512x3 and the MLP arms are fine
# at 1024.
cd "$(dirname "$0")/.."
PY=.pixi/envs/default/bin/python
LOGS=results/benchmarks/logs
FAILED=""

run() {
    name=$1; shift
    if [ -f "results/benchmarks/tokenizers/$name/checkpoints/last.ckpt" ]; then
        echo "=== $name (skip: checkpoint exists) ==="; return 0
    fi
    echo "=== $name ==="
    if $PY benchmarks/train_tokenizers.py --data aoj --epochs 15 --run-name "$name" "$@" \
        > "$LOGS/train_${name}.log" 2>&1; then
        echo "$name OK"
    else
        echo "$name FAILED (see $LOGS/train_${name}.log)"; FAILED="$FAILED $name"
    fi
}

# Track A: contrastive-VQ (kernel-shaped codebook, 512x3 = 27 bits) + prod anchor.
run contrastive_vqvae_aoj --model contrastive_vqvae
run vqvae_particle_k8192  --model vqvae_particle --codebook-size 8192 --num-quantizers 1 --batch-size 256

# Track B: low-dim continuous kernel-only library (fp16).
run library_aoj_d8 --model library --latent-dim 8
run library_aoj_d6 --model library --latent-dim 6
run library_aoj_d4 --model library --latent-dim 4

# Track C: encoder-capacity ablation (dim 8, kernel-only; h128 == library_aoj_d8).
run library_aoj_h256 --model library --latent-dim 8 --hidden-dims 256,256,128
run library_aoj_h64  --model library --latent-dim 8 --hidden-dims 64,64,32
run library_aoj_h32  --model library --latent-dim 8 --hidden-dims 32,32,16

if [ -n "$FAILED" ]; then echo "=== ITER 4 DONE (with failures:$FAILED) ==="; else echo "=== ITER 4 DONE ==="; fi
