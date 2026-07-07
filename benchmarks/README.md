# Tokenizer validation benchmark — continuous contrastive vs VQ-VAE

GPU validation campaign (A100, torch 2.5.1) on a **620k-jet, 5-class** dataset
(≤150 constituents × (pT, Δη, Δφ), zero-padded). Matched budgets: identical
4-layer transformer encoders; contrastive `latent_dim=16` vs VQ `codebook 512 ×
dim 16, 3 quantizers`; same preprocessing (log-quantile), split (70/15/15,
seed 42), and 20-epoch schedule.

**Consumer-view evaluation**: downstream probes see *only the exported token
release* (`latents.npz` / `indices+codebooks.npz`) — simulating the AI-ready-
dataset consumer with no constituent access. All arms use the same probe
(2-layer transformer `VectorClassifier` / `FeatureClassifier`, 250k jets).

## Results

| interface | probe acc | probe macro AUC | kNN class purity | anti-collapse |
|---|---|---|---|---|
| raw features (reference ceiling) | 0.743 | 0.931 | 0.249 | — |
| **contrastive tokens** | 0.701 | 0.917 | **0.486** | eff. rank 8.3/16, all dims active |
| **VQ tokens (baseline)** | **0.748** | **0.934** | 0.243 | quantizer-0 collapsed to <10% of codes |

Tokenizer-level diagnostics (validation):

| metric | contrastive | VQ-VAE |
|---|---|---|
| alignment ↓ (σ=0.05 views) | **0.0006** | 0.0030 |
| uniformity ↓ | **−2.79** | −1.77 |
| distance–ΔR correlation | 0.73 | 0.72 |
| distance–ΔpT correlation | 0.31 | **0.61** |
| cos-sim under train-σ smearing | 0.9997 | 0.9985 |
| scaled reconstruction MAE | — | 0.0137 |

## Findings

1. **No identity collapse, by construction.** The VICReg tokenizer keeps every
   latent dimension active (effective rank 8.3, per-dim std ≈ 2.9) with the best
   alignment *and* uniformity — the anti-collapse machinery works as designed
   (`collapse_diagnostics.png`, `alignment_uniformity.png`).
2. **The embeddings organize by different things.** PCA maps
   (`embedding_maps.png`): contrastive tokens cluster constituents by **jet
   class** (kNN purity 0.49 ≈ 2× raw features) — semantic structure emerged
   without labels; VQ tokens organize by **pT** with no visible class structure
   (its job is reconstruction).
3. **The probe gap is a diagnosed design knob, not a mystery.** The v1
   augmentation smeared *all three* scaled features — including pT — so the
   contrastive metric traded away pT sensitivity (distance–ΔpT correlation 0.31
   vs 0.61; flat curve in `physics_faithfulness.png`), costing ~1.5 AUC points
   downstream. The invariance list must *preserve hard-pT structure* (e.g.
   physics-SupCon positives with same-PID/ΔpT/ΔR, angular-only smearing).
4. **The audit's "strong null" prediction is confirmed.** With only 3 input
   features, information-preserving interfaces (VQ z_q ≈ compressed features)
   match the raw-feature ceiling — discriminative gains must come from *metric
   shaping*, which is exactly what contrastive tokenization controls.
5. **VQ pathology observed live.** Quantizer 0 collapsed to <10% codebook
   utilization (~50/512 codes; `codebook_vqvae.png`) — the documented VQ failure
   mode, now measured in our own baseline.

## Next steps

- Physics-informed positives: angular-only smearing + same-PID/(ΔpT,ΔR) SupCon
  (loss already implemented: `supcon_physics_loss`), then re-run this benchmark.
- Add masked-continuous and JEPA arms on the same harness.
- Aspen Open Jets (real CMS, PID/charge/d0/dz) as the next dataset.

## Reproduce

```bash
PY=.pixi/envs/default/bin/python
$PY benchmarks/prepare_data.py                                   # schema + scalers
$PY benchmarks/train_tokenizers.py --model contrastive --epochs 20
$PY benchmarks/train_tokenizers.py --model vqvae       --epochs 20
$PY benchmarks/train_probes.py    --epochs 15                    # exports + 3 probes
$PY benchmarks/make_plots.py                                     # figures/ + summary.json
```

Heavy artifacts land under `results/benchmarks/` (gitignored); figures and
`summary.json` are committed in `benchmarks/figures/`.
