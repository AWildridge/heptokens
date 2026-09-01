# Campaign 7 — the downstream consumer probe: QCD vs top on AOJ MC

*Method of record and results, 2026-09-01. Design pre-registered 2026-08-27
(commit `151517d`); harness fixes `cf77c26`, `969e691`, `c521350`; results
commit `7c04ca3`. Summary table in `benchmarks/README.md` §Campaign 7; this
document is the full protocol, the systematics record, and the interpretation.*

---

## 1. Motivation and question

Campaigns 1–6 evaluated tokens with producer-side metrics (criterion recall,
reconstruction R², linear probes). The mission's actual success metric — what a
foundation-model builder gets from the released token files — had never been
measured for any Campaign 3–6 winner. Campaign 7 measures it on the first
labelled task available to the program: **binary QCD-vs-top classification on
the Aspen Open Jets MC dataset** (fdr.uni-hamburg.de record 18610), with every
token arm consuming the frozen export of a tokenizer **trained on real AOJ RunG
collision data** and applied unmodified to MC. That transfer is deliberate: it
is exactly how a released tokenizer would be consumed in practice (frozen
weights, frozen preprocessing, someone else's sample).

Question: *ranked by downstream value to an identical supervised consumer,
which token interface wins — and does the producer-side complementarity
(criterion geometry vs information sufficiency) resolve?*

## 2. Data

**Source (record 18610).** QCD jets in four generator-pT bins
(`QCD_Pt{300,470,600,800}_batch0.h5`), each file carrying a top-level per-jet
`weight` dataset — the cross-section weights required to combine bins (the
record ships `pt_weights.py` documenting this contract). Top jets from
ZPrime→tt̄ at several mass points; the top files carry no `weight` dataset
(unit weights). Schema identical to real AOJ: `PFCands [N,150,11]`,
`jet_kinematics [N,4]`.

**Preparation (`benchmarks/prepare_aoj_mc.py`).** Sequential
download → window → subsample → delete-raw (23 GB disk ceiling):

- Common jet-pT window **[300, 1200] GeV** on both classes, so neither class is
  identifiable by phase space alone.
- 62,500 QCD jets per bin (250 k total), uniformly subsampled with kept weights
  scaled by 1/keep_frac so per-bin Σw — the physics — is preserved. Resulting
  per-bin Σw: 3.02×10⁶ / 3.13×10⁵ / 4.14×10⁴ / 1.55×10⁴ (steeply falling, as
  the cross sections demand).
- 250 k top jets pooled from M1000/M1200/M2500 batch0 (in-window yields 22.7 k /
  73.5 k / 153.9 k; M4000 not needed), then one weight-preserving subsample.
- Constituent derivation identical to `prepare_aoj.py` (valid = E>0,
  pt = hypot(px,py), η = asinh(pz/pt), φ = atan2(py,px)).
- **No scaler fitting.** The frozen real-AOJ scalers are reused — the release
  contract. Verified adequate on MC: scaled kinematics mean/std ≈
  (0.09, 0.00, 0.00) / (1.04, 0.90, 1.00); constituent-pT saturation 0.019 %.
- PID vocabulary check: MC contains the encoder's ten NanoAOD codes; codes
  1/2 (HF hadrons, absent in real RunG) are 0.0008 % of constituents — the
  feared PID domain shift is negligible.
- Output `aojmc_structured.h5`: 500 k jets,
  `jets/{pt,eta,phi,m,nconst,label,weight,origin}` (label 0=QCD, 1=top; origin
  = source file), `csts/{pt,eta,phi,pid,charge,valid}`.

Effective sample sizes (Σw)²/Σw²: QCD 78 k of 250 k (four per-bin weight
deltas), top 250 k (unit weights).

## 3. Protocol

**Consumer view, weights end-to-end.** The weight chain (all additions
backward-compatible): `jets/weight` → `StructuredArrayHDF5Dataset(weight_key)`
→ collation passthrough → tokenizer `predict_step` → export npz → probe
weighted cross-entropy → sklearn `sample_weight` metrics. The export
datamodule iterates in `rng(SEED).permutation(n)` order, so weights travel
*inside* each npz; label-equality and weight-allclose assertions guard the
permutation join used to attach jet pT for per-bin metrics.

**Split contract.** All arms use contiguous 70/15/15 slices of the same
permuted order — identical train/val/**held-out test** membership everywhere
(the jets5c-era harness evaluated on val; Campaign 7 reports test only).

**pT-spectrum protocol (user decision: *match + report both*).** The classes
have different pT spectra and the arms differ enormously in how much pT they
encode (VQ within-PID pT R² 0.97 vs library 0.17), so an unmatched spectrum
would reward pT bookkeeping rather than substructure. Training weights =
cross-section weights × histogram-ratio reweighting of top to the weighted QCD
spectrum (24 log bins, train slice only) × per-class normalization. Test
metrics are reported under both matched and physical weighting, plus
per-pT-bin AUC (edges 300/470/600/800/1200) as the residual-shortcut check.

**Metrics.** Weighted ROC AUC (both schemes); weighted balanced accuracy;
background rejection 1/FPR at 30 % and 50 % signal efficiency (the
HEP-standard working points — AUC compresses exactly the high-purity region a
tagger lives in); per-pT-bin weighted AUC.

**Probe (identical for every arm).** `VectorClassifier`: Linear(D→128) →
4-layer transformer (d_model 128, 8 heads, ff 512) → masked mean-pool → linear
head; ~0.83 M parameters — the repo's standard `transformer_partial()` scale,
chosen so the raw-feature arm is not capacity-limited. 3-epoch warmup-cosine
schedule, min 12 / max 25 epochs, early-stop patience 8 on val loss, gradient
clip 1.0, bf16-mixed, batch 512, Adam 1e-3. Seeds 42/43/44 per arm.

## 4. Arms

| arm | source | input dim | note |
|---|---|---|---|
| raw features | scaled kin ⊕ PID-one-hot ⊕ charge | 14 | full-information ceiling |
| raw kinematics | scaled (pt, η, φ) | 3 | PID-blind reference |
| physics grid, double | C6b binning, offsets 0/0.5, ⊕ PID ⊕ charge | 17 | untrained, ≈26.3 b |
| library dim 6 | `library_aoj_d6` | 6 | kernel-only, recall 0.9996 |
| hybrid w=3 | `hybrid_w30` | 16 | kernel+recon, R² 0.908 |
| contrastive VAE | `c6_vaeK_aoj_b003_d6` | 6 | β=0.03, recall 0.9984 |
| recon-only VAE | `c6_vae_aoj_b01_d6` | 6 | β=0.1, R² 0.920 |
| **VQ 3×512** | `vqvae_particle_aoj` | 16 | **heptokens baseline**, 27 b |
| VQ 3×1024 | `c5_vqp_aoj` | 16 | 30 b |

Latent exports are fp16 (`export_latents.py` gained `latent_dtype` and a
`model_class="vae"` path — closing the known LitVae export gap); VQ arms export
indices+codebooks and the probe reconstructs z_q.

## 5. Systematics found and fixed (methodological record)

Three harness pathologies were caught during validation; each is a reusable
lesson.

1. **Forked-dataloader deadlock (twice).** On this machine (A100 MIG slice),
   forked DataLoader workers intermittently deadlock — once at
   `trainer.predict` (17 h idle), once in fit's sanity-check loop (3 days
   idle; py-spy: main thread blocked on the worker queue, children idle). The
   probe data is fully in-RAM, so workers only paid fork overhead:
   **num_workers=0 everywhere** in this harness, and the runner wraps the
   probes stage in a 24 h `timeout` so any future hang fails loudly.
2. **Raw-PID encoding pathology.** Feeding raw PDG codes (±211, 130, …) as a
   linear-input column swamps the O(1) scaled kinematics after the first
   Linear+LayerNorm: the "ceiling" scored AUC 0.58 and effectively saw only
   species composition. One-hot encoding fixed it (val AUC 0.90 by epoch 6).
   Any consumer-facing schema should never ship raw categorical codes as
   numeric columns.
3. **Training-plateau / early-stopping artifact — the run-1 invalidation.**
   Probes on this task sit at AUC ≈ 0.5 for ~5 epochs before escaping; with
   patience-4 early stopping, runs were executed mid-plateau at epochs 5–6.
   Same arm, three seeds: raw kinematics 0.50 / 0.98 / 0.51; raw features
   0.95 / 0.53 / 0.84. The resulting "ranking" (archived:
   `results/benchmarks/probes/mc_run1_unfair_stopping/`) conflated token
   quality with plateau luck — and it *systematically favoured* the smooth,
   decorrelated learned-token interfaces, which escape the plateau fastest.
   Fix: warmup schedule + min_epochs 12 + patience 8. Under the fair budget
   every arm converges (24–25 epochs) and seed std collapses to ≤0.0015.
   Lesson: **under tight training budgets, "which interface trains easily"
   masquerades as "which interface is better" — a consumer-relevant effect,
   but one that must be measured separately from asymptotic value.**

## 6. Results (fair budget; test split; 3 seeds, mean ± std)

| interface | matched AUC | physical AUC | rej@30 % | rej@50 % | bal. acc |
|---|---|---|---|---|---|
| physics grid, double (untrained) | **0.9829 ± .0002** | 0.9734 | **1084** | **305** | 0.937 |
| raw features (ceiling) | 0.9822 ± .0002 | 0.9691 | 1034 | 301 | 0.938 |
| raw kinematics | 0.9819 ± .0003 | 0.9722 | 823 | 291 | 0.936 |
| recon-only VAE β=0.1 d6 | **0.9781 ± .0004** | 0.9711 | 737 | 207 | 0.929 |
| VQ 3×1024 | 0.9777 ± .0010 | 0.9643 | 710 | 213 | 0.928 |
| VQ 3×512 (baseline) | 0.9770 ± .0006 | 0.9687 | 761 | 213 | 0.926 |
| contrastive VAE β=0.03 d6 | 0.9743 ± .0010 | 0.9638 | 408 | 138 | 0.924 |
| hybrid w=3 | 0.9715 ± .0015 | 0.9591 | 362 | 131 | 0.916 |
| library d6 | 0.9696 ± .0004 | 0.9588 | 276 | 101 | 0.917 |

Per-pT-bin AUC (physical weights) degrades uniformly with pT for every arm
(≈0.98 in 300–470 → 0.86–0.91 in 800–1200) with the ordering broadly
preserved (adjacent arms swap within seed bands in the last bin) — no arm's
advantage is a residual pT shortcut.

### Findings

1. **The untrained physics grid is the best consumer interface in the
   program — at ≈26.3 bits.** It statistically matches/exceeds the raw
   ceiling (0.9829 vs 0.9822; rejection 305 vs 301): it *is* the raw
   information, re-encoded in a well-conditioned, criterion-aligned form with
   double-grid smoothing. Continuing the pattern of Campaigns 2b and 6, an
   untrained reference beats the learned tokens on the metric that matters.
2. **Downstream value tracks information retention and is the exact reverse
   of criterion recall.** Learned-token order: recon-VAE ≥ VQ(30 b) ≈
   VQ(27 b) > contrastive-VAE > hybrid > library — precisely the within-PID
   pT-R²/linear-probe order, and the mirror image of the criterion-recall
   order (library, recall 0.9996, finishes last). The producer-side
   complementarity of C2b–C6 resolves in favour of information for supervised
   consumers.
3. **AUC flatters; rejection tells the truth.** Worst-to-ceiling gap is
   0.013 AUC but **3× in background rejection at 50 % signal efficiency**
   (301 → 101). Criterion-shaped forgetting has a real physics price.
4. **The release-realistic transfer works.** Real-data-trained tokenizers +
   frozen scalers on MC: 0.97+ AUC everywhere; scaler saturation and PID
   domain shift negligible (§2).
5. **PID contributes almost nothing to this task** (kinematics-only 0.9819 vs
   full 0.9822): top tagging at these pT is a substructure task. The PID
   channel's value is task-dependent — flavour-type consumers will weigh it
   very differently; this probe does not license dropping it from the release.
6. **The two VQ rates are downstream-equivalent** (0.9777 vs 0.9770,
   overlapping seed bands): the extra 3 bits of the 1024-codebook buy nothing
   a supervised consumer can use.

## 7. Implications for the release, and revised program conclusions

- **For supervised/tagging consumers the tier ordering flips:** the primary
  recommendation becomes the criterion-aligned grid (26.3 b, zero parameters,
  computable on any detector) or an information-first learned token
  (recon-VAE dim 6 at 96 fp16 b, or the VQ baseline at 27 b). The
  kernel-organized tokens (library/contrastive-VAE/hybrid) are demoted for
  this consumer class.
- **The kernel tier is not retired** — it remains the only interface that
  delivers the retrieval/library property (criterion recall 0.999 vs the
  grid's 0.966 and VQ's 0.945, C2–C6), which this supervised probe does not
  exercise. The release stays two-tier, with the roles now measured on both
  sides: *information tier for training consumers, kernel tier for retrieval
  consumers.*
- **C6's "learning a discrete token buys almost nothing over binning"
  extends to the consumer side** — and strengthens: downstream, the grid
  doesn't merely match the trained VQ, it beats it (0.9829 vs 0.9770,
  rejection 305 vs 213).
- The C2b claim "the library objective costs nothing downstream" (jets5c,
  kinematics-only, 3 features) does **not** transfer to the 5-feature MC task:
  with real headroom, kernel organization costs 0.9822 → 0.9696 AUC and 3× in
  rejection. The earlier null result was a no-headroom artifact, as the C6
  audit suspected.

## 8. Limitations and next steps

- One probe family and size, one supervised task. Not yet measured: label-free
  consumers (masked/JEPA-style FM pretraining on the exports — the original
  Campaign-7 concept, still open), retrieval-at-scale, anomaly detection (the
  protocol's mandated third probe, still unrun), and low-label efficiency.
- The task barely uses PID; a flavour-sensitive task would stress the
  categorical channel (and the d0/dz gap — impact parameters are still absent
  from the release, C6 audit).
- Top statistics under pT matching have ESS ≈ 15 k (M2500 dominance vs the
  falling QCD spectrum); a broader Z′ mass scan would improve this.
- Frontier unchanged otherwise: REPA distillation and the factorized
  symbols ⊕ latent release for the open corner; cross-experiment invariance
  remains data-blocked.

## 9. Reproduce

```bash
benchmarks/run_campaign7.sh prepare   # download → window → subsample → assemble
benchmarks/run_campaign7.sh probes    # exports (idempotent) + 9 arms × 3 seeds
.pixi/envs/default/bin/python benchmarks/train_probes_mc.py --figures-only
```

Artifacts: `results/benchmarks/probes/mc/{<arm>_s<seed>/metrics.json,
summary.json}`, figures `benchmarks/figures/c7_{auc_summary,roc}.png`;
invalidated first run preserved under `probes/mc_run1_unfair_stopping/`.
