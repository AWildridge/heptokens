#!/usr/bin/env bash
# Campaign 5 — continuous(VAE) vs discrete(VQ), +-physics, context-free vs -aware.
#
# The 2x2x2 study on AOJ (real CMS data):
#   bottleneck : VAE (continuous)          | VQ-VAE RVQ 3x1024 = 30 bits
#   objective  : reconstruction-only       | + physics-contrastive term on latent
#   encoder    : mlp (context-free)         | attention (context-aware)
# plus a VAE latent-dim sweep {2,4,6}, a supcon ("hard CLR") variant, and the
# already-trained 8192x1 plain-VQ anchor (vqvae_particle_k8192, reused).
#
# Idempotent (skips a run whose last.ckpt exists), fault-tolerant (no `set -e`;
# each failure is recorded, the sweep continues), sequential (one GPU job at a
# time on the shared MIG slice — do NOT run eval concurrently).
#
# Usage: benchmarks/run_campaign5.sh [a|b|all]      (default: all)

cd "$(dirname "$0")/.." || exit 1
PY=.pixi/envs/default/bin/python
TOK=results/benchmarks/tokenizers
STAGE="${1:-all}"
FAILED=""

run () {
  local name="$1"; shift
  if [ -f "$TOK/$name/checkpoints/last.ckpt" ]; then
    echo "[skip] $name (last.ckpt exists)"
    return 0
  fi
  echo "[run ] $name :: $*"
  if $PY benchmarks/train_tokenizers.py --run-name "$name" --data aoj --epochs 20 "$@"; then
    echo "[ok  ] $name"
  else
    echo "[FAIL] $name"
    FAILED="$FAILED $name"
  fi
}

if [ "$STAGE" = "a" ] || [ "$STAGE" = "all" ]; then
  echo "===== Stage A: context-free (mlp) ====="
  # Continuous VAE, reconstruction-only — latent-dim sweep (rate-distortion).
  run c5_vae_aoj    --model vae --encoder mlp --latent-dim 4 --batch-size 1024
  run c5_vae_aoj_d2 --model vae --encoder mlp --latent-dim 2 --batch-size 1024
  run c5_vae_aoj_d6 --model vae --encoder mlp --latent-dim 6 --batch-size 1024
  # Continuous VAE + physics term on mu (graded kernel, then hard-positive supcon).
  run c5_vaeK_aoj        --model vae --encoder mlp --latent-dim 4 --contrastive --loss kernel --batch-size 1024
  run c5_vaeK_aoj_supcon --model vae --encoder mlp --latent-dim 4 --contrastive --loss supcon --batch-size 1024
  # Discrete VQ, RVQ 3x1024 = 30 bits: plain, then contrastive-VQ (kernel on z_e).
  run c5_vqp_aoj --model vqvae_particle   --encoder mlp --codebook-size 1024 --num-quantizers 3 --batch-size 1024
  run c5_cvq_aoj --model contrastive_vqvae --encoder mlp --codebook-size 1024 --num-quantizers 3 --kernel-on z_e --batch-size 1024
  # 8192x1 plain-VQ anchor: reuse the existing vqvae_particle_k8192 run.
fi

if [ "$STAGE" = "b" ] || [ "$STAGE" = "all" ]; then
  echo "===== Stage B: context-aware (attention) ====="
  run c5_vae_aoj_attn  --model vae --encoder attention --latent-dim 4 --batch-size 512
  run c5_vaeK_aoj_attn --model vae --encoder attention --latent-dim 4 --contrastive --loss kernel --batch-size 512
  run c5_vqp_aoj_attn  --model vqvae_particle    --encoder attention --codebook-size 1024 --num-quantizers 3 --batch-size 512
  run c5_cvq_aoj_attn  --model contrastive_vqvae --encoder attention --codebook-size 1024 --num-quantizers 3 --kernel-on z_e --batch-size 512
fi

echo "===================================================="
if [ -n "$FAILED" ]; then
  echo "FAILED runs:$FAILED"
else
  echo "all campaign-5 runs ok"
fi
