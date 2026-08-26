#!/usr/bin/env bash
# Campaign 6 — alternate schemes beyond the standard VQ-VAE (see
# docs/research/campaign6_alternate_schemes.md for the design + pre-registered
# outcomes).
#
#   6a : rate-matched contrastive-VAE frontier — beta x dim x free-bits sweep
#        (the sanctioned beta-sweep; makes VAE-vs-VQ rate-fair for the first time)
#   6c : FSQ-bottleneck library tokenizer — codebook-free discrete token, kernel
#        on the smooth pre-rounding latent (z_q ablation + recon-only control)
#   6b : physics-grid untrained discrete reference — eval-only, computed inside
#        eval_library.py (stage "eval")
#
# Idempotent (skips a run whose last.ckpt exists), fault-tolerant (no `set -e`),
# sequential (one GPU job at a time on the shared MIG slice — eval only runs
# after training, never concurrently).
#
# Usage: benchmarks/run_campaign6.sh [a|b|eval|all]      (default: all)

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
  echo "===== Stage 6a: contrastive-VAE beta/dim/free-bits frontier (MLP, kernel on mu) ====="
  # dim-6 beta ladder (the C4 sweet-spot dimension).
  run c6_vaeK_aoj_b003_d6 --model vae --encoder mlp --latent-dim 6 --contrastive --loss kernel --beta 0.03 --batch-size 1024
  run c6_vaeK_aoj_b01_d6  --model vae --encoder mlp --latent-dim 6 --contrastive --loss kernel --beta 0.1  --batch-size 1024
  run c6_vaeK_aoj_b03_d6  --model vae --encoder mlp --latent-dim 6 --contrastive --loss kernel --beta 0.3  --batch-size 1024
  run c6_vaeK_aoj_b1_d6   --model vae --encoder mlp --latent-dim 6 --contrastive --loss kernel --beta 1.0  --batch-size 1024
  run c6_vaeK_aoj_b3_d6   --model vae --encoder mlp --latent-dim 6 --contrastive --loss kernel --beta 3.0  --batch-size 1024
  # dim-4 / dim-8 arms at the interesting betas (c5_vaeK_aoj is the dim-4 beta=1 point).
  run c6_vaeK_aoj_b01_d4  --model vae --encoder mlp --latent-dim 4 --contrastive --loss kernel --beta 0.1  --batch-size 1024
  run c6_vaeK_aoj_b03_d4  --model vae --encoder mlp --latent-dim 4 --contrastive --loss kernel --beta 0.3  --batch-size 1024
  run c6_vaeK_aoj_b01_d8  --model vae --encoder mlp --latent-dim 8 --contrastive --loss kernel --beta 0.1  --batch-size 1024
  run c6_vaeK_aoj_b03_d8  --model vae --encoder mlp --latent-dim 8 --contrastive --loss kernel --beta 0.3  --batch-size 1024
  # Posterior-collapse insurance at high beta.
  run c6_vaeK_aoj_b3_d6_fb --model vae --encoder mlp --latent-dim 6 --contrastive --loss kernel --beta 3.0 --free-bits 0.05 --batch-size 1024
  # Recon-only low-beta control: does it approach the deterministic dim-6 library?
  run c6_vae_aoj_b01_d6   --model vae --encoder mlp --latent-dim 6 --beta 0.1 --batch-size 1024
fi

if [ "$STAGE" = "b" ] || [ "$STAGE" = "all" ]; then
  echo "===== Stage 6c: FSQ-bottleneck library tokenizer (codebook-free discrete) ====="
  # Kernel on z_e (default), rate ladder: ~10 / ~15.5 / 18 bits.
  run c6_fsqK_aoj_d4_10b --model fsq --encoder mlp --fsq-levels 8,5,5,5       --loss kernel --batch-size 1024
  run c6_fsqK_aoj_d6_15b --model fsq --encoder mlp --fsq-levels 6,6,6,6,6,6   --loss kernel --batch-size 1024
  run c6_fsqK_aoj_d6_18b --model fsq --encoder mlp --fsq-levels 8,8,8,8,8,8   --loss kernel --batch-size 1024
  # Hybrid: kernel + full-particle reconstruction through the grid token.
  run c6_fsqK_aoj_d6_15b_w1 --model fsq --encoder mlp --fsq-levels 6,6,6,6,6,6 --loss kernel --recon-weight 1.0 --batch-size 1024
  # Ablation: kernel on the grid token itself (through the rounding STE).
  run c6_fsqK_aoj_d6_15b_zq --model fsq --encoder mlp --fsq-levels 6,6,6,6,6,6 --loss kernel --fsq-kernel-on z_q --batch-size 1024
  # Recon-only FSQ control (plain-VQ analog at this bottleneck).
  run c6_fsq_aoj_d6_15b  --model fsq --encoder mlp --fsq-levels 6,6,6,6,6,6   --contrastive-weight 0 --recon-weight 1.0 --batch-size 1024
fi

if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
  echo "===== Eval: all providers incl. 6b physics grid (GPU must be free of training) ====="
  if $PY benchmarks/eval_library.py; then
    echo "[ok  ] eval_library"
  else
    echo "[FAIL] eval_library"
    FAILED="$FAILED eval_library"
  fi
fi

echo "===================================================="
if [ -n "$FAILED" ]; then
  echo "FAILED runs:$FAILED"
  exit 1
else
  echo "all campaign-6 runs ok"
fi
