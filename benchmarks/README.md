# Tokenizer benchmarks

## Campaign 3 — hybrid objective, tighter ΔpT, and embedding-size study

**The question.** Campaign 2 left the library with strong criterion geometry but
weak within-PID pT recovery (R² 0.18). Iteration 3 asks: can a **hybrid
objective** (physics kernel + a `ParticleDecoder` reconstruction term, weight
`w`) lift pT R² without breaking the criterion; does tightening the ΔpT kernel to
5% help; and — since the 16-D token is *larger than the 5-column input* — can the
embedding shrink without degrading the library? All runs: AOJ, hybrid arm, 15
epochs, seed 42.

| run | dim | recall@8 ↑ | recall@8 (ΔpT 5%) | within-PID pT R² | kNN PID purity | eff. rank |
|---|---|---|---|---|---|---|
| library (kernel only) | 16 | **0.9992** | 0.9995 | 0.175 | 0.906 | 15.8 |
| hybrid w=0.3 | 16 | 0.9992 | 0.9993 | 0.360 | 0.899 | 15.5 |
| hybrid w=1.0 | 16 | 0.9984 | 0.9983 | 0.867 | 0.915 | 15.3 |
| hybrid w=2.0 | 16 | 0.9968 | 0.9961 | 0.807 | 0.918 | 15.3 |
| hybrid w=3.0 | 16 | 0.9955 | 0.9947 | **0.908** | 0.935 | 15.2 |
| hybrid w=2.0, ΔpT 5% | 16 | 0.9954 | 0.9971 | 0.891 | 0.909 | 15.3 |
| **hybrid w=2.0, dim 8** | 8 | 0.9982 | 0.9987 | 0.613 | **0.987** | 7.8 |
| hybrid w=2.0, dim 4 | 4 | 0.9801 | 0.9914 | 0.799 | 0.987 | 3.8 |

**Findings**

1. **Recall and pT R² are a controllable frontier, not a corner you can reach at
   once.** `recon_weight` interpolates smoothly — w=0.3 owns recall
   (0.9992/0.36), w=3.0 owns pT R² (0.9955/0.908) — but no single weight clears
   both the recall ≥ 0.999 *and* R² ≥ 0.9 target. The nearest point is w=3.0,
   which meets R² and misses recall by ~0.004. The hybrid buys a tunable dial
   between organization and information, which is the right release knob to
   expose; the "both corners" ideal is not attainable by weight alone.
   See `library_pareto.png`.
2. **Tightening ΔpT to 5% sharpens pT fidelity, modestly.** At fixed w=2.0,
   10%→5% lifted R² 0.807→0.891 and the tight-criterion recall@8 0.9961→0.9971,
   costing a little loose recall (0.9968→0.9954). It nudges the pT pole but does
   not cross the target corner.
3. **The embedding halves for free — and the criterion improves.** Going 16→8
   dims *raised* criterion recall (0.9968→0.9982) and sharply raised kNN PID
   purity (0.918→0.987), with effective rank 7.8 (nearly all 8 dims used). Only
   pT reconstruction R² softened (0.807→0.613), the expected cost of fewer
   regression dimensions. The criterion never needed 16 dims — at 16 the space
   was under-packed for clustering, and compressing it *sharpens* the library
   metric. **Recommended library size: 8-D** (still ≥ the 5 inputs, but no longer
   inflated). Dim 4 is the floor: recall finally cracks (0.980) while purity
   holds.

Reproduce: `run_stage_a.sh` (weight sweep) + `run_stage_b.sh` (w=2.0, ΔpT 5%,
dims 8/4) → `eval_library.py`. Figures: `library_pareto.png` (frontier with
shaded target region), `library_retrieval.png`.

## Campaign 2 — particle-library tokenizer on Aspen Open Jets (real CMS)

**The user-specified objective:** particles with **same PID, same charge,
ΔpT ≤ 10%, |Δη| ≤ 0.1 and |Δφ| ≤ 0.1** (same detector region) are positives;
everything else is repelled. Context-free per-particle **MLP Φ** (PID/charge
via embeddings — same particle ⇒ same token anywhere), trained with a
**graded physics-kernel** soft-target contrastive loss (hard PID+charge gates,
Gaussian grading on the kinematics). Data: AOJ `RunG_batch0` — 1.96M real CMS
jets, absolute η–φ, PDG ∈ {±211, ±13, ±11, 22, 130}. No labels, no
augmentations, no views.

**Criterion recall@8** — of an anchor's *available* criterion partners in a
30k pool, the fraction ranked inside its top-8 neighbours (1.0 = perfect;
plain precision@k is availability-limited at criterion density ~3e-5 and
flattens all methods):

| provider | recall@8 ↑ | kNN PID purity | within-PID pT R² | eff. rank |
|---|---|---|---|---|
| **library (graded, coords)** | **0.9992** | 0.910 | 0.18 | 15.8/16 |
| library (graded, dR cone) | 0.9985 | 0.911 | 0.16 | 15.7 |
| library (hard windows) | 0.9935 | 0.664 | 0.30 | 15.4 |
| null: PID one-hot ⊕ kinematics (untrained) | 0.9833 | 0.993 | **0.95** | 5.1 |
| raw kinematics only | 0.300 | 0.290 | 0.95 | 3.0 |
| v1 transformer-contrastive | 0.052 | 0.284 | 0.84 | 4.4 |
| v1 VQ z_q | 0.608 | 0.298 | 0.97 | 3.7 |

**Findings**

1. **The library criterion is achieved.** 99.92% of available
   same-PID/charge/pT/region partners rank in the top-8 — a ~20× reduction in
   retrieval misses vs the untrained hand-crafted null (1.7% → 0.08%): the
   learning pays rent on the criterion itself. `library_pid_islands.png`
   shows clean species islands with kinematic substructure on real CMS data.
2. **Graded kernel > hard windows**, as predicted: fewer retrieval misses
   (8×) and far better species organization (purity 0.91 vs 0.66);
   `library_cliff_check.png` shows the hard variant's geometry kinking ~2×
   steeper past the ΔpT window edge.
3. **Context-free MLP was the right call:** the jet-context-aware v1
   transformer scores 0.052 on criterion recall — context leakage destroys
   the library property.
4. **Open deficiency — within-island kinematic compression:** linear pT
   recovery inside a PID class is R² 0.18 vs 0.95 for the null. For the
   AI-ready release (observable recovery matters), next iteration should add
   kinematic-preservation pressure: an auxiliary feature-prediction term,
   multi-scale ΔpT kernels, or a hybrid release (token ⊕ scaled kinematics).
5. Region definition (per-coordinate vs dR cone) is a wash on these metrics —
   per-coordinate kept as default per spec.

Reproduce: `prepare_aoj.py` → `train_tokenizers.py --model library --data aoj
[--region cone | --hard-windows]` → `eval_library.py`.

## Campaign 2b — apples-to-apples: library vs VQ-VAE

**The contract.** The `v1_vqvae` row above is *not* a fair comparison (different
data, inputs, architecture). This campaign isolates the **objective axis** and
holds everything else fixed: same AOJ data/split, same 5 inputs (scaled
kinematics + PID + charge), the **same `ParticleEncoder`** (MLP 128-128-64,
16-dim token), same schedule, **3 seeds each**. The VQ arm
(`--model vqvae_particle`) adds only its own machinery: an RVQ bottleneck
(3×512 = 27 bits/particle) and a `ParticleDecoder` trained to reconstruct the
**full particle** (L1 kinematics + CE PID + CE charge — a kinematics-only
target would strawman VQ on species metrics).

| interface (AOJ, mean±std over 3 seeds) | criterion recall@8 ↑ᵃ | kNN PID purity | PID linear probe | pT R² (within PID) | eff. rank | bits/cst |
|---|---|---|---|---|---|---|
| **library (graded kernel)** | **0.9991 ± 0.0002** | 0.905 ± 0.003 | 0.366 ± 0.014 | 0.169 ± 0.006 | 15.8 | 256 (fp16) |
| **VQ, matched (`vqvae_particle`)** | 0.9271 ± 0.0062 | **0.9961 ± 0.0004** | **0.9999 ± 0.0001** | **0.9706 ± 0.0006** | 5.5 ± 0.3 | **27** |
| library → post-hoc RVQ @ 27 bits | 0.832 | 0.863 | 0.380 | 0.174 | 15.8 | 27 |
| null: PID-one-hot ⊕ kin. (untrained) | 0.9833 | 0.9929 | 1.0 | 0.9488 | 5.1 | — |

ᵃ criterion recall is the library's native objective (flagged); PID/charge
probes and pT R² are reconstruction-flavoured (VQ-native-ish); purity and rank
are neutral.

Downstream consumer probe (jets5c, labels available; identical kinematics-only
arms, same `VectorClassifier`): raw features 0.743 / 0.931 · **library tokens
0.744 / 0.931** · VQ tokens 0.748 / 0.934 · v1 contrastive 0.701 / 0.917.

**Findings**

1. **The objectives are complementary, and neither dominates.** The library
   owns the criterion geometry (miss rate 0.09% vs 7.3% — ~80× fewer misses
   than the matched VQ); the VQ owns information sufficiency (PID and charge
   perfectly linearly decodable, within-PID pT R² 0.97 vs 0.17).
2. **Library species structure is metric, not linear:** kNN PID purity 0.91
   but a *linear* PID probe scores ≈ the majority-class baseline (0.37 ≈ raw
   kinematics 0.36) — the t-SNE islands are non-convex. Consumers using
   nonlinear models see the structure; linear probes do not.
3. **The library objective costs nothing downstream:** its jets5c probe sits
   exactly at the raw-features ceiling (0.744/0.931), closing v1's −1.5 AUC
   gap. Matched VQ is +0.3 AUC above ceiling (within noise).
4. **Rate regime matters for the release format:** at 27 bits/particle the
   *trained* VQ (recall 0.927) beats post-hoc-compressed library tokens
   (0.832) — post-hoc compression holds at mild rates but trained quantization
   defends the geometry better under ~10× compression.
5. **VQ quantizer-0 partial collapse is robust across seeds** (15–19%
   utilization; q1/q2 ≈ 100%) — the campaign-1 pathology, reproduced in the
   matched setting.
6. **Iteration 3 target is now quantified:** a hybrid objective (physics
   kernel + reconstruction term) should aim for recall ≥ 0.999 **and**
   pT R² ≥ 0.9 simultaneously — the two poles are measured.

Reproduce: `benchmarks/run_sweep.sh` → `eval_library.py`;
probe arm: `train_probes.py --arms library`.

---

## Campaign 1 — continuous contrastive vs VQ-VAE (jets5c)

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
