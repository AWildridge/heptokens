# Tokenizer benchmarks

## Campaign 6 — alternate schemes: rate-matched VAE frontier, physics grid, FSQ

**The question.** Three arms from the alternate-scheme evaluation
(`docs/research/campaign6_alternate_schemes.md`, pre-registered outcomes in §5):
**6a** — the sanctioned β-sweep: is the C5 VAE-vs-VQ verdict rate-fair, i.e. does
the contrastive VAE match plain VQ on *information* once allowed VQ-level rate?
**6b** — an untrained criterion-aligned grid (log-pT bins of ln(1.1), η/φ bins of
0.1, exact PID/charge symbols; double grid = half-bin-offset copy): what must any
learned *discrete* token beat? **6c** — FSQ (codebook-free quantization, zero
learned params): was the C4/C5 criterion-shaped-VQ negative about codebook/EMA
dynamics, or about quantization itself? All AOJ, context-free MLP, 20 epochs;
headline arms at 3 seeds.

| artifact | bits | recall@8 ↑ | dpt5 | kNN purity | PID probe | pT R² | note |
|---|---|---|---|---|---|---|---|
| *6a — contrastive VAE (kernel on μ), β sweep* | | | | | | | |
| **vaeK β=0.03 d6** (3 seeds) | **13.3 KL** | **0.9984 ± .0015** | 1.000 | 0.989 | 0.59 | 0.758 ± .037 | recall parity with library_d6 at ~7× fewer bits |
| vaeK β=0.1 d6 | 10.7 KL | 0.992 | 0.991 | 0.994 | 0.69 | 0.767 | |
| vaeK β=0.3 d6 | 0.57 KL | 0.977 | 0.981 | 0.976 | 0.63 | 0.821 | best R² on the sweep |
| vaeK β=1 d6 | 0.18 KL | 0.974 | 0.978 | 0.975 | 0.81 | 0.803 | |
| vaeK β=3 d6 | **0.001 KL** | 0.9995 | 1.000 | 0.987 | 0.56 | 0.568 | **KL-rate artifact — see finding 2** |
| vaeK β=3 d6 + free-bits .05 | 0.41 KL | 0.974 | 0.979 | 0.974 | 0.78 | 0.767 | free-bits does not repair the accounting |
| **vae β=0.1 d6, recon-only** (3 seeds) | 11.0 KL | 0.9762 ± .0041 | 0.985 | 0.995 | **1.00** | **0.920 ± .011** | revises the C5 recon-only story — finding 3 |
| *6b — physics grid (UNTRAINED discrete reference)* | | | | | | | |
| physics grid | 23.3 | 0.9553 | 0.963 | 0.987 | 1.00 | 0.948 | zero parameters |
| **physics grid, double (½-bin offset)** | 26.3 | **0.9655** | 0.973 | 0.979 | 1.00 | 0.949 | beats every learned discrete token on recall |
| *6c — FSQ (codebook-free discrete)* | | | | | | | |
| fsqK d4 [8,5,5,5] | 10.0 | 0.039 | 0.037 | 0.883 | 0.91 | 0.562 | 208/1000 grid pts used |
| fsqK d6 [6⁶] | 15.5 | 0.136 | 0.138 | 0.719 | 0.53 | 0.661 | 1 004/46 656 pts (2%) — code degeneracy |
| fsqK d6 [6⁶], kernel→z_q | 15.5 | 0.092 | 0.093 | 0.977 | 0.95 | 0.346 | STE placement no better |
| fsqK d6 [6⁶] + recon w=1 | 15.5 | 0.571 | 0.589 | 0.933 | 0.73 | 0.692 | recon partly rescues |
| fsq d6 [6⁶] recon-only | 15.5 | 0.633 | 0.676 | 0.862 | 0.76 | 0.849 | 12 968 pts (28%) used |
| fsqK d6 [8⁶] | 18.0 | 0.411 | 0.416 | 0.832 | 0.48 | 0.582 | |
| fsqK d8 [8⁸] | 24.0 | 0.530 | — | 0.728 | 0.43 | 0.388 | rate-matched: kernel still hurts |
| fsq d8 [8⁸] recon-only | 24.0 | 0.851 | — | 0.939 | 0.81 | 0.840 | < grid @ 23.3 bits |
| *references (this eval)* | | | | | | | |
| plain VQ 3×1024 (c5) | 30 | 0.945 | 0.966 | 0.996 | 1.00 | 0.970 | |
| VQ particle 3×512 (3 seeds) | 27 | 0.927 ± .006 | 0.945 | 0.996 | 1.00 | 0.971 | |
| library dim 6 (C4) | 96 | 0.9996 | 1.000 | 0.980 | 0.49 | 0.743 | |
| null (PID one-hot ⊕ kin) | 192 | 0.9833 | 0.986 | 0.991 | 1.00 | 0.949 | untrained |

**Findings**

1. **The β dial completes the rate axis, and VQ is *not* retired.** The
   contrastive VAE reaches criterion recall **0.9984 ± 0.0015 at ~13 KL-bits**
   (β=0.03, dim 6) — parity with the 96-bit deterministic dim-6 library and far
   above plain VQ (0.945 at 30 bits) — but pT R² tops out ≈ 0.82 anywhere on the
   sweep, short of the pre-registered ≥ 0.97 retirement bar. The
   organization-vs-information complementarity survives rate-matching.
2. **KL-rate is not a release rate for deterministic-μ tokens — measured.** The
   β=3 arm scores recall 0.9995 at a measured KL of **0.001 bits**: cosine
   ranking is scale-invariant, so a tiny-norm but structured μ carries the full
   criterion geometry while the posterior sits on the prior. The C5 "~1.4-bit"
   label was this artifact in milder form. Until an operational codec exists
   (entropy-coding roadmap item), continuous-token rates must be quoted as fp16
   dim·16; KL-bits are reported as diagnostics only.
3. **"Reconstruction buys species, not the criterion" was a rate artifact.** At
   β=0.1 (≈ 11 KL-bits) the *recon-only* VAE reaches recall **0.976 ± 0.004**
   with pT R² 0.920 and perfect linear PID/charge probes — C5's 0.383 was the
   β=1 bottleneck, not the objective. It still does not beat the untrained null
   (0.983 / 0.949) on either axis, so the kernel remains the only thing that
   *ever* has (0.9984 vs 0.9833).
4. **The untrained physics grid near-dominates every learned discrete token.**
   Double grid: recall 0.9655 at 26.3 bits with R² 0.949 and probes 1.0 — more
   recall than plain VQ at fewer bits; the trained VQ's entire surviving edge is
   +0.02 pT R² and +0.017 purity. Below the 0.99 auto-retirement threshold, but
   the practical verdict stands: at the 23–30-bit tier, **learning a discrete
   token currently buys almost nothing over criterion-aligned binning**, and the
   grid is trivially computable on any detector (a natural cross-experiment
   release tier).
5. **The discrete negative upgrades: quantization per se conflicts with the
   kernel geometry.** With no codebook, no EMA, no commitment loss, the kernel
   still *hurts* at every rate tested (10 → 24 bits: 0.039/0.136/0.411/0.530 vs
   recon-only 0.633/0.851), and the mechanism is now visible: the kernel's fixed
   point drives criterion partners toward coincidence, which any quantizer turns
   into code degeneracy (2% grid occupancy vs 28% for recon-only) — retrieval
   then cannot rank true partners among identical-code impostors. Together with
   C4/C5 this closes the "criterion-shaped discrete token" program: install the
   criterion on a continuous latent, and if a discrete tier is needed, bin the
   *features* (6b), not the embedding.

**The honest tradeoff.** Still no single trained token is sub-raw,
reconstruction-faithful, and criterion-perfect. The updated menu: criterion +
compact → contrastive VAE (recall 0.998 at dim 6 / 96 fp16 bits, honest rate);
faithful + criterion at a discrete tier → the untrained double grid (0.966 /
R² 0.949 / 26.3 bits) with trained VQ worth keeping only if the last +0.02 of
pT R² matters; the corner remains open for the REPA-style distillation and
factorized-release proposals (doc §4).

Reproduce: `benchmarks/run_campaign6.sh` → `eval_library.py` (grid providers are
eval-time, no training). Figures: `library_rate.png` (all points, one bits axis).

## Campaign 5 — continuous (VAE) vs discrete (VQ), ± physics, context-free vs -aware

**The question.** A clean 2×2×2 on one bits axis: bottleneck {continuous **VAE**,
discrete **VQ** RVQ 3×1024}, objective {reconstruction-only, **+ physics-contrastive
term on the latent**}, encoder {context-free **MLP**, context-aware **attention**}.
Everything else matched (same `ParticleEncoder`/`ParticleDecoder`, AOJ, 20 epochs).
VAE rate = KL/ln2 bits (the ELBO rate; naive fp16 size is dim·16); VQ rate =
num_quantizers·log₂(codebook).

| artifact | bits | recall@8 ↑ | dpt5 | kNN purity | pT R² | eff. rank | note |
|---|---|---|---|---|---|---|---|
| *context-free (MLP)* | | | | | | | |
| VAE dim2 (recon) | 2.7 | 0.033 | 0.041 | 0.996 | 0.194 | 2.0 | recon → species, not criterion |
| VAE dim4 (recon) | 2.6 | 0.383 | 0.527 | 0.996 | 0.836 | 2.0 | 0.996 purity, 0.38 recall |
| VAE dim6 (recon) | 2.5 | 0.629 | 0.770 | 0.997 | 0.491 | 2.0 | |
| **VAE dim4 + kernel** | **1.4** | **0.916** | 0.924 | 0.995 | 0.665 | 3.8 | kernel installs the criterion |
| VAE dim4 + supcon | 1.6 | 0.774 | 0.848 | 0.996 | 0.437 | 3.2 | hard-CLR variant |
| **plain VQ 3×1024** | 30 | **0.945** | 0.966 | 0.997 | **0.970** | 6.3 | all-around champion |
| contrastive-VQ (kernel→z_e) | 30 | 0.827 | 0.845 | 0.899 | 0.331 | 15.8 | kernel **hurts** VQ again |
| VQ single 8192 | 13 | 0.028 | 0.026 | 0.996 | 0.821 | 8.3 | collapsed (iter-4) |
| *context-aware (attention)* | | | | | | | |
| VAE dim4 (recon) | 2.6 | 0.051 | 0.052 | 0.995 | 0.479 | 2.0 | context **wrecks** the criterion |
| VAE dim4 + kernel | 3.0 | 0.859 | 0.868 | 0.996 | 0.809 | 3.7 | kernel partly rescues |
| plain VQ 3×1024 | 30 | **0.967** | 0.979 | 0.996 | 0.967 | 4.9 | context slightly **helps** VQ |
| contrastive-VQ (kernel→z_e) | 30 | 0.816 | 0.831 | 0.880 | 0.498 | 15.5 | |
| *references* | | | | | | | |
| raw kinematics | 48 | 0.300 | 0.401 | 0.296 | 0.949 | 3.0 | scaled pt,η,φ only |
| null (PID one-hot + kin) | 192 | 0.983 | 0.986 | 0.994 | 0.949 | 5.1 | untrained |
| library (continuous kernel, dim16) | 256 | 0.999 | 0.999 | 0.909 | 0.175 | 15.8 | Campaign 2 |

**Findings**

1. **The contrastive term belongs on the continuous latent, not the codebook —
   placement is decisive.** On the VAE (smooth mean μ) the physics kernel lifts
   criterion recall **0.383 → 0.916**; on the VQ (straight-through + EMA codebook)
   the *same* kernel **drops** it **0.945 → 0.827** and pT R² 0.970 → 0.331. Same
   term, opposite sign — exactly the gradient-path prediction, and it re-confirms
   iteration-4's VQ negative at the new 30-bit rate. The clean-gradient continuous
   quadrant is where physics-shaping works.
2. **Reconstruction buys species; the criterion needs the kernel.** The recon-only
   VAE has 0.996 kNN PID purity and 1.00 PID/charge linear probe yet only **0.383**
   criterion recall — because the criterion also demands ΔpT ≤ 10% and same region,
   which reconstruction never imposes. kNN purity is a misleading proxy; recall@8 is
   the real test, and only the contrastive term installs it.
3. **Rate–distortion: VQ (30 bits) is the all-around champion; the contrastive VAE
   reaches criterion parity at ~1.4 bits.** Plain VQ wins every axis at 30 bits
   (recall 0.945, pT R² 0.970, near-perfect recon 0.029). The contrastive VAE hits
   recall 0.916 at **1.4 bits** (~20× fewer, by the KL rate) — but is
   non-reconstructive and weaker on pT (R² 0.665). Faithful + criterion → VQ;
   extreme compression + criterion geometry → contrastive VAE.
4. **Context is a real, asymmetric tradeoff — not a free upgrade.** Attention
   slightly *helps* the high-bit VQ (recall 0.945 → 0.967) but *wrecks* the low-bit
   recon-only VAE (0.383 → **0.051**): a context-aware token folds jet environment
   into the per-particle code, and at low rate that crowds out the criterion — the
   universal-vocabulary property breaking, made vivid. The kernel partly rescues the
   attention VAE (0.859). Only the high-rate discrete token affords both.
5. **Granularity beats bit-count, again.** RVQ 3×1024 (util 0.09/1.0/0.99) is
   healthy; the single 8192 codebook still collapses (recall 0.028).

**The honest tradeoff.** No token is at once sub-raw, reconstructive, *and*
criterion-perfect. Discrete VQ (30 bits) is faithful and criterion-strong but not
the smallest; the contrastive VAE is extraordinarily compact (~1.4 bits) and
criterion-strong but lossy; context buys jet information at the cost of the
per-particle criterion. See `library_rate.png` (all points on one bits axis).

Reproduce: `run_campaign5.sh` → `eval_library.py`.

## Campaign 4 — compressive tokens: criterion-shaped VQ, low-dim, capacity

**The question.** The continuous library token is *anti-compressive*: a raw
particle packs into ~50 bits, but the token is 128 bits (dim-8 fp16) to 256
(dim-16). Genuine compression needs a discrete codebook — so does *shaping that
codebook with the physics kernel* give a compact token that keeps the criterion?
And separately: how far does the continuous token compress, and is the encoder
overbuilt? All arms AOJ, 15 epochs; VQ rate = num_quantizers·log₂(codebook).

| artifact | bits | recall@8 ↑ | kNN purity | pT R² | eff. rank | note |
|---|---|---|---|---|---|---|
| library (dim 16) | 256 | 0.9992 | 0.906 | 0.175 | 15.8 | continuous, iter-2 |
| **library dim 6** | **96** | **0.9996** | **0.980** | **0.743** | 6.0 | continuous sweet spot |
| library dim 8 | 128 | 0.9997 | 0.973 | 0.445 | 8.0 | |
| library dim 4 | 64 | 0.9807 | 0.987 | 0.684 | 4.0 | continuous floor |
| plain VQ (RVQ 512×3) | 27 | 0.927 ± .006 | 0.996 | 0.971 | 5.6 | 3 seeds |
| contrastive-VQ (kernel→z_q) | 27 | 0.708 | 0.991 | 0.719 | 15.2 | kernel **hurt** |
| contrastive-VQ (kernel→z_e) | 27 | 0.772 | 0.901 | 0.295 | 15.8 | still < plain VQ |
| VQ single 8192 codebook | 13 | 0.028 | 0.996 | 0.821 | 8.3 | collapsed, 1.7% util |

**Findings**

1. **Criterion-shaping a VQ codebook fails — the headline negative.** Adding the
   physics kernel to the VQ *underperforms* plain reconstruction-only VQ at the
   same 27 bits, whether the kernel organizes the quantized code (`z_q`: 0.708)
   or the continuous pre-quantization embedding (`z_e`: 0.772) vs plain VQ's
   0.927. The kernel and the discrete reconstruction bottleneck fight: the kernel
   spreads the representation to all 16 dims (eff. rank 15+ vs plain VQ's compact
   5.6) but quantization cannot preserve that geometry, and it disrupts the clean
   reconstruction-induced clustering. Injecting the criterion *into the codebook*
   does not yield a compact criterion-preserving token.
2. **Compression works — on the continuous side.** The continuous library
   compresses 256→96 bits (dim 16→6) with the criterion *intact and sharper*:
   recall 0.9992→0.9996, purity 0.906→0.980, and pT R² *rising* 0.175→0.743
   (fewer dims force the token to retain kinematics it otherwise discards inside
   the 10% window). **dim-6 dominates every VQ point** on the compression curve.
   dim-4 (64 bits) is the floor (recall 0.981). To keep the criterion under
   compression, shrink the continuous token; do not quantize it.
3. **Among VQ options, granularity beats bit-count.** RVQ 512×3 (0.927) crushes a
   single 8192 codebook that collapses to 1.7% utilization (recall 0.028) — the
   coarse code cannot rank true partners within a shared code. How the bits are
   spent matters far more than how many.
4. **The encoder is overbuilt.** MLP width 64 (hidden 64,64,32) matches width 256
   (recall 0.998 vs 0.9998); only width 32 dips (0.989). The context-free encoder
   can be more than halved.

**The honest tradeoff.** Genuine sub-raw compression (VQ, 13–27 bits) costs
criterion recall; criterion-preserving compression (continuous dim-6, 96 bits)
lands at ~2× the raw particle. No method tried is *both* sub-raw *and*
criterion-perfect. See `library_rate.png` (the compression curve).

Reproduce: `run_iter4.sh` + the `--kernel-on z_e` variant → `eval_library.py`.

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
