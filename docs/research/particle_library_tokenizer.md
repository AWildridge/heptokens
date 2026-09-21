# The Particle-Library Tokenizer

> **Status:** implemented and validated on real CMS data (July 2026).
> **Method of record** for the heptokens tokenizer stage — supersedes the
> exploratory smearing-VICReg design (v1) whose audit and survey live in
> [`continuous_tokenization.md`](continuous_tokenization.md).
> **Mission context:** AI-ready token datasets — experiments publish
> per-constituent embeddings as the shareable artifact; FM builders train on
> tokens without PF-candidate access.

---

## 1. The objective (specification)

Among all particles in all jets, build a **library of particles**: an
embedding in which

> particles with the **same PID**, the **same charge**, similar transverse
> momentum (**ΔpT ≤ 10 %**, relative) and heading to the **same detector
> region** (**|Δη| ≤ 0.1 and |Δφ| ≤ 0.1**) have similar embeddings
> (positives), and everything else is pushed away.

PID and charge are *inputs* to the token (embedded, not one-hot physics
codes), and the criterion organizes the space around them. The design is
deliberately **detector-based**: the token encodes *what was measured where*,
so that (later) CMS and ATLAS particles with the same properties and detector
region cluster together — cross-experiment alignment falls out of mixed
batches, with **no paired events required**, because the criterion is
computable on any experiment's particles.

This replaces an *implicit* metric (whatever an augmentation list happens to
preserve) with an **explicit, auditable, versionable** one — the property a
data release needs.

## 2. Design decisions and why

| Decision | Choice | Rationale / evidence |
|---|---|---|
| Encoder | **context-free per-particle MLP (Φ)** — no attention | The criterion is a function of single-particle properties; jet context would make the same particle embed differently in different jets, destroying the library property. Measured: the context-aware v1 transformer scores **0.052** criterion recall vs **0.9992** for the MLP. |
| Positive structure | **graded Gaussian kernel** on (ΔpT %, Δη, Δφ) with **hard PID and charge gates** | Binary windows create cliffs at arbitrary edges; the graded kernel keeps the criterion as the organizing principle with smooth geometry. Measured: hard windows lose species purity (0.66 vs 0.91) and kink ~2× steeper past the ΔpT edge. |
| Region definition | per-coordinate windows (\|Δη\|, wrapped \|Δφ\|) default; dR cone as config switch | Ablated on AOJ: statistically a wash; per-coordinate kept per specification (detector-cell semantics). |
| Charge gate | explicit, independent of PID | Robust to PID conventions that do or don't encode sign; matters cross-experiment. |
| Negatives | softmax denominator over the batch (soft-target contrastive) | "The rest should be away" without hand-picked negatives; physically identical particles across jets are positives *by construction* — no false-negative problem. |
| Augmentations | **none** | Positives come from the physics criterion across all jets in the batch, not from synthetic views. |
| Projector | none (loss on the token itself) | The released token is the trained object; no train/release mismatch. |

## 3. The loss

For valid constituents across the batch (subsampled to `max_points`), with
physical-unit features (pT, η, φ, PID, q):

```
s_ij = [pid_i = pid_j] · [q_i = q_j]
       · exp(−½ (ΔpT%_ij / 0.1)²) · exp(−½ [(Δη_ij/0.1)² + (Δφ_ij/0.1)²])
```

(`Δφ` wrapped; the cone variant replaces the bracket with `(ΔR/dr_max)²`.)
Each kernel row (self excluded, normalized) is the **soft target** for the
softmax over embedding cosine similarities at temperature τ:

```
L = − mean over anchors i with kernel mass:  Σ_j  q_ij · log softmax_j(z_i·z_j / τ)
```

Anchors with no kernel mass in the sample are dropped (`positive_frac`
logged — 0.95 on AOJ at batch 1024/`max_points` 4096). Collapse is prevented
by the softmax denominator; optional VICReg variance/covariance terms exist as
insurance (off by default). Implementation:
`physics_kernel_loss` in `src/heptokens/models/losses.py`; the hard-window
ablation is `supcon_physics_loss` (same gates, binary windows).

## 4. Architecture

`ParticleEncoder` (`src/heptokens/models/particle_encoder.py`):

- continuous kinematics (scaled pT, η, φ) ⊕ `nn.Embedding` of PID (PDG-code
  lookup table, unknown → "other" bucket) ⊕ `nn.Embedding` of charge
  ({−1,0,+1});
- fused vector → MLP (`CoderModel`, 128–128–64) → **16-dim token**;
- masked positions zeroed; unit-tested that the same particle produces the
  same token wherever it appears (the context-free property).

Wrapped by `LitContrastiveTokenizer` in the single-view
`loss_inputs="features"` mode: physical-unit features ride along under
`csts_raw` (`keep_raw_batch` collate transform ordered before preprocessing),
and PID/charge survive scaling via `PassthroughColumnsTransformer`
(`src/heptokens/data/transforms.py`). Config:
`src/heptokens/conf/model/particle_library.yaml`.

## 5. Data

**Aspen Open Jets** `RunG_batch0.h5` (record 16505): 1.96M real CMS jets,
`PFCands [N,150,11]` = (px, py, pz, E, d0, d0Err, dz, dzErr, charge, pdgId,
puppi). Prep (`benchmarks/prepare_aoj.py`): 4-momenta → **absolute** (pT, η,
φ); PID ∈ {±211, ±13, ±11, 22, 130}; charge; validity from E > 0. Real
collision data ⇒ no truth labels ⇒ evaluation is label-free (library
metrics). A kinematics-only mechanics run (PID gate off, jet-relative
coordinates) validates the machinery on the 620k-jet 5-class set first.

## 6. Evaluation protocol (label-free, criterion-referenced)

- **Criterion recall@k** (primary): of an anchor's *available* criterion
  partners in a 30k pool, the fraction ranked inside its top-k neighbours.
  (Plain precision@k saturates: the criterion's pair density is ~3·10⁻⁵, so
  every method hits the availability ceiling — recall against that ceiling is
  the informative quantity.) `compute_retrieval_precision` +
  `criterion_recall_at_k` in `benchmarks/eval_library.py`.
- **kNN PID purity**, **effective rank / collapse diagnostics**
  (`models/utils.py`).
- **Within-PID kinematic recovery** (linear R² of log pT inside a species) —
  the over-compression / PID-shortcut check.
- **Cliff check**: median embedding distance vs ΔpT for same-PID/charge/region
  pairs across the window edge (graded vs hard geometry).
- **Nulls**: untrained PID-one-hot ⊕ scaled kinematics (the "is learning
  paying rent" reference), raw kinematics, and the v1 arms.

## 7. Results (AOJ, real CMS)

| provider | recall@8 ↑ | PID purity | within-PID pT R² | eff. rank |
|---|---|---|---|---|
| **library (graded, coords)** | **0.9992** | 0.910 | 0.18 | 15.8/16 |
| library (graded, cone) | 0.9985 | 0.911 | 0.16 | 15.7 |
| library (hard windows) | 0.9935 | 0.664 | 0.30 | 15.4 |
| null (untrained one-hot ⊕ kin.) | 0.9833 | 0.993 | **0.95** | 5.1 |
| raw kinematics | 0.300 | 0.290 | 0.95 | 3.0 |
| v1 transformer-contrastive | 0.052 | 0.284 | 0.84 | 4.4 |
| v1 VQ z_q | 0.608 | 0.298 | 0.97 | 3.7 |

- The criterion is **achieved**: 20× fewer retrieval misses than the
  hand-crafted null; t-SNE shows clean species islands with kinematic
  substructure (`benchmarks/figures/library_pid_islands.png`).
- **Known limitation:** within-species kinematics are over-compressed
  (R² 0.18 vs 0.95) — the criterion doesn't reward pT *ordering* beyond the
  10 % scale, so the network discards it. For a released token that must
  support observable recovery, iteration 2 adds kinematic-preservation
  pressure: an auxiliary feature-prediction term, multi-scale ΔpT kernels, or
  a hybrid release (token ⊕ scaled kinematics).

Full tables, figures, and reproduction commands: `benchmarks/README.md`.

## 7b. Fair benchmark vs VQ-VAE (matched everything, objective isolated)

Same data, split, 5 inputs, the same `ParticleEncoder`, same schedule, 3 seeds;
the VQ arm differs only in its objective — an RVQ bottleneck (3×512 = 27
bits/particle) with a `ParticleDecoder` reconstructing the full particle
(L1 kinematics + CE PID + CE charge). Outcome (details in
`benchmarks/README.md` §Campaign 2b):

- **Complementarity, not dominance.** Library owns the criterion (recall@8
  0.9991 ± 0.0002 vs 0.9271 ± 0.0062 — ~80× fewer retrieval misses); matched
  VQ owns information sufficiency (PID/charge linearly decodable at ~100%,
  within-PID pT R² 0.97 vs 0.17).
- Library species structure is **metric, not linear** (kNN purity 0.91;
  linear PID probe ≈ majority baseline) — visible as non-convex t-SNE islands.
- **No downstream cost:** library tokens probe at the raw-features ceiling on
  jets5c (0.744 acc / 0.931 AUC), closing v1's gap.
- **Rate regime:** at 27 bits, trained VQ (0.927) beats post-hoc RVQ-compressed
  library tokens (0.832) — post-hoc compression is fine at mild rates, but
  trained quantization defends the geometry under aggressive (~10×) rates.
  This refines §1d of the survey doc.
- VQ quantizer-0 partial collapse reproduces across all seeds (15–19% util).

## 7c. Hybrid objective + embedding-size study (iteration 3)

The hybrid arm adds a `ParticleDecoder` reconstruction term (weight `w`) to the
kernel, reconstructing scaled kinematics + PID + charge from the token. Sweep and
dim study (AOJ, 15 epochs, seed 42; full table in `benchmarks/README.md`
§Campaign 3):

- **Organization vs information is a controllable frontier.** `w` interpolates
  smoothly between the kernel pole (w=0.3: recall@8 0.9992, pT R² 0.36) and the
  reconstruction pole (w=3.0: recall@8 0.9955, R² 0.908). No single `w` clears
  **both** recall ≥ 0.999 and R² ≥ 0.9 — the target corner is not reachable by
  weight alone; the nearest point (w=3.0) meets R² and misses recall by ~0.004.
  The practical outcome is a tunable release dial, not a dominated optimum
  (`benchmarks/figures/library_pareto.png`).
- **Tighter ΔpT (5%) sharpens the pT pole modestly:** at w=2.0, R² 0.807→0.891
  and tight-criterion recall 0.9961→0.9971, at a small cost to loose recall.
- **The token halves for free.** The 16-D token was *larger than the 5-column
  input*; compressing to **8-D raised** criterion recall (0.9968→0.9982) and kNN
  purity (0.918→0.987), softening only pT R² (0.807→0.613). The criterion never
  needed 16 dims — 8-D is the recommended library size. Dim 4 is the floor
  (recall drops to 0.980; purity holds).

## 7d. Compression: criterion-shaped VQ vs low-dim continuous (iteration 4)

Can the token be genuinely *compressed* (the continuous token is anti-compressive
— 96–256 bits vs a ~50-bit raw particle)? Two routes tested against plain VQ, all
AOJ, 15 epochs (full table in `benchmarks/README.md` §Campaign 4):

- **Criterion-shaping a VQ codebook fails — a robust negative.** Adding the
  physics kernel to a residual-VQ (`LitContrastiveVqVae`) *underperforms* plain
  reconstruction-only VQ at the same 27 bits, whether the kernel organizes the
  quantized code (`z_q`: recall 0.708) or the continuous pre-quantization
  embedding (`z_e`: 0.772) vs plain VQ's 0.927 ± 0.006. The kernel spreads the
  representation to all 16 dims (eff. rank 15+ vs plain VQ's 5.6) but quantization
  cannot preserve that geometry and disrupts the clean recon-induced clustering.
  Injecting the criterion *into the codebook* does not yield a compact
  criterion-preserving token.
- **Compression works on the continuous side.** The continuous library compresses
  256→96 bits (dim 16→6) with the criterion *sharper*: recall 0.9992→0.9996,
  purity 0.906→0.980, pT R² *rising* 0.175→0.743. **dim-6 dominates every VQ
  point** (`library_rate.png`); dim-4 (64 bits) is the floor.
- **Granularity beats bit-count** for VQ: RVQ 512×3 (0.927) ≫ a single 8192
  codebook that collapses to 1.7% util (0.028). **The encoder is overbuilt** —
  MLP width 64 matches 256.
- **Honest tradeoff:** sub-raw compression (VQ, 13–27 bits) costs criterion
  recall; criterion-preserving compression (continuous dim-6, 96 bits) sits at
  ~2× the raw particle. Nothing tried is both sub-raw *and* criterion-perfect.

## 7e. VAE vs VQ, ± physics, context-free vs -aware (campaign 5)

A 2×2×2 on one bits axis — bottleneck {continuous **VAE**, discrete **VQ** 3×1024},
objective {reconstruction-only, **+ physics term on the latent**}, encoder
{context-free **MLP**, context-aware **attention**}; matched coders, AOJ, 20 epochs.
VAE rate = KL/ln2 bits (full table in `benchmarks/README.md` §Campaign 5):

- **Placement is decisive.** The physics kernel on the VAE's smooth posterior mean
  lifts criterion recall **0.383 → 0.916**; the *same* kernel on the VQ codebook
  (straight-through + EMA) **drops** it 0.945 → 0.827 (pT R² 0.970 → 0.331). The
  continuous latent is the clean-gradient home for the criterion; the codebook is
  not — the iteration-4 negative, reconfirmed at 30 bits.
- **Reconstruction buys species, not the criterion.** Recon-only VAE: 0.996 kNN PID
  purity and 1.00 PID/charge probe, yet 0.383 recall — the criterion also needs
  ΔpT/region, which reconstruction never imposes. kNN purity is a misleading proxy;
  recall@8 is the test, and only the contrastive term installs it.
- **Rate–distortion.** Plain VQ (30 bits) is the all-around champion (recall 0.945,
  pT R² 0.970, near-perfect recon); the contrastive VAE reaches recall **0.916 at
  ~1.4 bits** (~20× fewer) but lossy (pT R² 0.665). Faithful → VQ; extreme
  compression + criterion geometry → contrastive VAE. This is the first token that
  is *both* sub-raw and criterion-strong (answering §7d's open question), at the
  cost of reconstruction — no decoupling channel needed.
- **Context is an asymmetric tradeoff.** Attention slightly *helps* the high-bit VQ
  (0.945 → 0.967) but *wrecks* the low-bit recon-only VAE (0.383 → **0.051**): a
  context-aware token folds jet environment into the per-particle code and, at low
  rate, crowds out the criterion — the universal-vocabulary property breaking. Only
  the high-rate discrete token affords both context and criterion.

## 8. Roadmap

1. **Iteration 3 — done (frontier, not corner).** The hybrid objective delivers a
   *controllable* recall/pT-R² trade-off but no single weight reaches recall
   ≥ 0.999 **and** R² ≥ 0.9 at once; expose `recon_weight` as a release knob.
2. **Iteration 4 — done (compress continuous, not discrete).** Criterion-shaped
   VQ is falsified (both `z_q`/`z_e` < plain VQ); the compression path is the
   **continuous dim-6 token** (96 bits, recall 0.9996, pT R² 0.743), and the
   encoder can be halved to width 64. Open question — a token that is *both*
   sub-raw *and* criterion-perfect: pursue decoupling (dim-6 library ⊕ an explicit
   scaled-kinematics channel) rather than a VQ bottleneck, or accept the tradeoff.
3. **Campaign 5 — done (placement + rate, not architecture).** The physics term
   belongs on the **continuous** latent (VAE recall 0.38 → 0.92), not the VQ
   codebook (0.95 → 0.83); reconstruction yields species but not the criterion;
   plain VQ (30 bits) is the faithful all-rounder, while the **contrastive VAE
   reaches criterion parity at ~1.4 bits** — the sub-raw *and* criterion-strong
   token §7d asked for (lossy, no decoupling needed). Context (attention) buys jet
   information but breaks the low-rate per-particle criterion (VAE 0.38 → 0.05).
2. **Release contract:** version tokens (dim, preprocessing hash, checkpoint
   id) via `export_latents.py`; post-hoc compression (fp16 / PQ).
3. **Cross-experiment:** mixed CMS + second-detector batches under the same
   criterion (no paired events needed); measure cross-sample mixing in the
   library metrics.
4. **Downstream:** train FM backbones (masked-continuous / JEPA) on the
   released tokens; consumer-view probes per the survey's benchmark protocol.
