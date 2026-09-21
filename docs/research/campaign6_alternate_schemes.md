# Alternate tokenization & embedding schemes — evaluation and Campaign 6 design

*2026-08-26. Product of a full-repo evaluation (both research docs, the slide deck,
all library + benchmark code, all recorded results) plus three literature scans
(discrete, continuous, and HEP-specific tokenization, 2023–2026), synthesized and
then adversarially audited. This document records (1) what Campaigns 1–5
established about departing from the standard VQ-VAE, (2) which recorded claims
survive audit and which need caveats, (3) verified gaps, (4) the ranked frontier
of alternate schemes, and (5) the Campaign 6 experimental design now implemented
in `benchmarks/`.*

---

## 1. What Campaigns 1–5 established

The program departed from the original residual-VQ-VAE tokenizer along one axis at
a time, under a consumer-view protocol (probes see only exported tokens), matched
encoders/data/schedules, and strong untrained nulls:

- **C1 (jets5c):** generic smearing-VICReg trains collapse-free, but the
  augmentation list silently defines the metric (smearing pT cost ~1.5 AUC
  downstream) and the VQ baseline reproduced quantizer-0 codebook collapse live.
- **C2/2b (AOJ, 1.96M real CMS jets):** the explicit physics criterion (same PID ∧
  charge, ΔpT ≤ 10 %, |Δη|,|Δφ| ≤ 0.1; graded kernel; context-free MLP) achieves
  recall@8 0.9992 vs 0.9833 untrained null. Matched-budget VQ comparison:
  **complementarity, not dominance** — library owns criterion geometry
  (0.9991 ± 0.0002 vs 0.9271 ± 0.0062), VQ owns information (PID/charge linearly
  decodable, within-PID pT R² 0.97 vs 0.17).
- **C3:** recall-vs-R² is a smooth `recon_weight` frontier; the corner
  (recall ≥ 0.999 ∧ R² ≥ 0.9) is unreachable by the weight dial; 16-D → 8-D is free.
- **C4:** headline negative — criterion-shaping a learned VQ codebook fails on both
  `z_q` (0.708) and `z_e` (0.772) vs plain VQ 0.927 at equal 27 bits. Compression of
  the criterion works on the continuous side (dim-6, 96 bits, recall 0.9996).
- **C5 (2×2×2):** **placement decides** — the kernel on a VAE's smooth posterior
  mean lifts recall 0.383 → 0.916 at ~1.4 KL-bits; the identical term through the
  VQ codebook drops 0.945 → 0.827. Context is an asymmetric trade (helps 30-bit VQ
  0.945 → 0.967, wrecks the low-rate recon VAE 0.383 → 0.051). Plain RVQ 3×1024 is
  the all-round champion at 30 bits; the contrastive VAE the first sub-raw
  criterion-strong token.

**Net:** the metric must be stated explicitly in physics, not implied by
augmentations; the contrastive criterion needs a clean gradient on a continuous
latent; reconstruction and criterion are separable currencies with a controllable
exchange rate; compression of the criterion should happen on the continuous side.

## 2. Audit: solid vs settled-with-caveats

Solid and multiply confirmed: placement of the contrastive term; criterion-shaping
STE+EMA codebooks fails (27 & 30 bits, `z_q` & `z_e`, MLP & attention);
reconstruction buys species but never the criterion (kNN purity is a misleading
proxy); graded kernels > hard windows; RVQ granularity > raw bit-count.

Claims that need caveats (adversarial audit findings):

1. **"The token must be context-free"** is mission-normative, not measured: C5's
   own attention-VQ *beats* context-free on the criterion at 30 bits (0.967 vs
   0.945, single seed vs measured ±0.006 seed std). The "must" rests on the
   untested, data-blocked cross-experiment universality argument plus low-rate
   results.
2. **"Costs nothing downstream"** was only measured for kinematics-mode arms on
   jets5c; the PID/charge method-of-record token — and every C3–C5 winner — has
   **no downstream consumer number at all**.
3. **"InfoNCE unsoundness confirmed empirically"** — it wasn't. The supcon arm uses
   physics-defined positives (no false-negative problem by construction); its
   deficit shows graded > hard gating. Vanilla in-batch-negative InfoNCE never ran.
4. **The ~1.4-bit contrastive VAE is an ELBO rate, not a storage format** — no
   operational codec exists at that rate.
5. **Seed fragility:** all C3–C5 headline numbers are seed-42-only against a
   measured ±0.006 recall std; several settled deltas are within a few σ.
6. **The 8192-codebook collapse arm has a confound**: it trained at batch 256 (MIG
   memory) vs 1024 for every competitor; kmeans init / dead-code resets were never
   tried, so the "granularity beats bit-count" negative is about *this* quantizer
   configuration.

## 3. Verified repo gaps (all checked in code)

- **`prepare_aoj.py` silently drops d0, d0Err, dz, dzErr, puppiWeight** — the
  survey's own invariance table marks d0/dz "preserve, high priority". b-tagging
  consumers need them; no campaign restores them.
- **The anomaly-detection probe mandated by the benchmark protocol never ran.**
- **Two coexisting release recommendations were never adjudicated:** C3 recommends
  8-D hybrid (R² 0.613); C4 crowns dim-6 kernel-only (recall 0.9996, R² 0.743).
- **`set2vec_vqvae` exists in `src/heptokens/conf/model/` but was never run** — a
  jet-level side token would resolve C5's context-vs-universality tension.
- Kernel temperature τ = 0.1 has zero ablation; recall is measured on a single 30k
  pool from one shard; `free_bits` existed in `LitVae` but had no CLI (fixed in
  this campaign).

## 4. Ranked frontier of alternate schemes

Runnable today (order reflects the locked frontier: sanctioned steps first, then
the highest-information new family):

1. **Rate-matched contrastive-VAE frontier (β/free-bits sweep)** — *Campaign 6a.*
2. **Consumer-view masked-continuous FM probe across release formats** — *deferred
   to Campaign 7*, deliberately: it is the biggest build, and probing after 6a/6c
   lets it cover the complete release-format menu (rate-matched VAE points + any
   FSQ winner) in one pass instead of two.
3. **FSQ/BSQ codebook-free discrete tokenizer + kernel** — *Campaign 6c.* Honest
   framing: the nearest repo evidence (kernel→`z_e` before a learned codebook)
   **failed twice** (0.772@27b, 0.827@30b). FSQ is the test of *why*: it removes
   the codebook, EMA dynamics, and commitment loss — per-dim rounding of a bounded
   latent, exact designer-chosen bits. If FSQ+kernel also fails, the negative
   upgrades from "STE+EMA codebooks" to "quantization per se conflicts with the
   kernel geometry". No HEP precedent either way.
4. **Criterion-aligned physics-grid token (untrained discrete reference)** —
   *Campaign 6b.* Eval-only, hours. Every learned discrete token must beat it or
   be retired.
5. **Physics-factorized release** (exact PID/charge symbols ⊕ compact kinematic
   latent) — fixes the metric-not-linear blocker by construction. Caveat: the
   untrained null nearly *is* this artifact (0.983 recall / R² 0.95 / probes 1.0 at
   ~53 symbol-coded bits); the learned channel's rent is the last ~1.6 % of recall.
6. **REPA-style frozen-teacher alignment** (arXiv:2410.06940) — distill the frozen
   dim-6 library geometry into a reconstruction latent by regression (no live
   contrastive dynamics); a mechanism-changed attack on the unreached corner.
7. **Rotation-trick (arXiv:2410.06424) + SimVQ (arXiv:2411.02038) codebook retest +
   VQ hygiene** (kmeans init, dead-code resets, batch-size deconfound) — either
   revives criterion-shaped VQ or upgrades the negative; independently calibrates
   whether q0 collapse depressed the plain-VQ champion.
8. **Criterion-ordered invertible flow tokenizer** (JetFormer-style soft tokens) —
   R² ≡ 1 by construction, kernel on leading dims, nested release tiers, exact
   likelihoods as a free anomaly handle. Larger build.
9. **Pseudo-second-detector mixed-batch prototype** — falsification harness for the
   data-blocked north star (frozen documented response map on AOJ, not an
   augmentation-defined metric: positives still come only from the criterion).

Additional families surfaced by the audit, held for later: Gumbel-softmax/dVAE
stochastic discrete bottlenecks; prototype/cell-classification objective (installs
the criterion at O(N) and is linearly decodable by construction); learned entropy
coding to turn KL-bits into release bytes; the never-run `set2vec_vqvae` jet-level
side token; d0/dz/PUPPI feature restoration; the anomaly probe.

## 5. Campaign 6 design (implemented)

Three sub-campaigns, all AOJ, context-free MLP encoder (per locked design), 20
epochs, seed 42 (winner re-run at seeds 43–44 before any claim is promoted).

### 6a — rate-matched contrastive-VAE frontier (β / dim / free-bits)

Arms (`--model vae --contrastive --loss kernel`, batch 1024):
β ∈ {0.03, 0.1, 0.3, 1, 3} at dim 6; β ∈ {0.1, 0.3} at dims 4 and 8;
β = 3 + `--free-bits 0.05` (posterior-collapse insurance); one recon-only β = 0.1
dim-6 arm (does a low-β recon VAE approach the deterministic dim-6 library?).
`kernel_weight` held at 1.0 (the β×kernel_weight interaction is noted as un-swept).

*What each arm forgets:* exactly β-controlled — low β forgets little (→ the
deterministic dim-6 library), high β forgets everything but the criterion. The
sweep makes the forgetting curve explicit, which is the release knob the mission
wants. Every point reports measured KL-bits **and** fp16 dim·16 side by side (audit
item 4).

*Pre-registered outcomes:* (a) some (β, dim) point reaches recall ≥ 0.945 (≥ plain
VQ 30 b) **and** pT R² ≥ 0.97 at ≤ 30 KL-bits → the VQ arm is retired and the
release unifies around one continuous family; or (b) the bottleneck saturates →
the plot delimits where discrete tokens remain necessary.

### 6b — criterion-aligned physics-grid reference (untrained, eval-only)

Providers `physics_grid` / `physics_grid2` in `eval_library.py`: log-pT bins of
width ln(1.1), η/φ bins of 0.1, PID/charge exact; the double grid adds a
half-bin-offset copy (~+3 bits) to repair boundary pairs — the zero-parameter
cousin of FSQ in feature space. Bits are computed from occupied bins at eval time.

*What it forgets:* within-bin kinematic position — exactly quantified, exactly
versionable; nothing about species. Within-PID pT R² is ~1 *by construction* (bin
centers): this is reported as the cap, not a learning result.

*Pre-registered outcomes:* grid ≥ 0.99 recall at ~23–27 bits → it becomes the
sub-raw discrete reference every learned discrete token must beat (and is itself a
shippable, trivially cross-detector-computable release tier); grid ≈ null (0.983)
→ it cleanly quantifies what learning buys. **Blast-radius note (from the
audit):** if the double grid clears ≥ 0.99, several "what learning buys" claims
(e.g. trained-VQ-defends-geometry, which rests on a single 27-bit post-hoc point)
must be revisited — this is a feature of the reference, not a bug.

### 6c — FSQ-bottleneck library tokenizer (codebook-free discrete)

New `FSQ` module (`src/heptokens/models/fsq.py`, zero learned parameters; Mentzer
et al., arXiv:2309.15505) wired as an optional bottleneck in
`LitContrastiveTokenizer` (`fsq_levels`). The released token is the quantized grid
point `z_q`; the kernel is applied to the bounded pre-rounding latent `z_e`
(default) with a `z_q` ablation (`--fsq-kernel-on`), since the z_e-vs-z_q question
was contested for VQ and both modes must be measured (design rule).

Arms: kernel at levels [8,5,5,5] ≈ 10 bits (dim 4), [6×6] ≈ 15.5 bits (dim 6),
[8×6] = 18 bits (dim 6); + recon hybrid w = 1 at 15.5 bits; kernel-on-`z_q`
ablation; recon-only FSQ control (`--contrastive-weight 0 --recon-weight 1`) —
the plain-VQ analog that isolates the kernel's effect on this bottleneck.

*What each arm forgets:* same as the kernel (within-species pT ordering beyond the
10 % scale; dial back with w) **plus** sub-grid kinematic detail below the
quantization step — both explicit and auditable, unlike VQ's collapse-driven
forgetting.

*Pre-registered outcomes:* (a) FSQ+kernel ≥ 0.99 recall at ≤ 20 bits, beating
plain VQ (0.945@30 b), post-hoc RVQ (0.832@27 b), and the physics grid at matched
bits → the first sub-raw criterion-strong discrete token; (b) FSQ+kernel ≲ plain
VQ → the discrete negative upgrades: the kernel-quantization conflict is about
quantization per se, not codebook dynamics. Both outcomes are publishable; the
recall-vs-bits placement against 6b decides whether *training* the discrete token
pays rent at all.

### Outcomes (2026-08-27 — full numbers in `benchmarks/README.md` §Campaign 6)

All 23 runs + eval completed; headline arms at 3 seeds. **6a:** recall parity with
the deterministic dim-6 library at ~13 KL-bits (0.9984 ± 0.0015) but R² ≤ 0.82 —
outcome (b): VQ not retired, complementarity survives rate-matching. Bonus
findings: the KL-rate axis is *broken* for deterministic-μ releases (β=3: recall
0.9995 at 0.001 KL-bits — retro-flags C5's "1.4-bit" label), and C5's
"reconstruction buys species, not the criterion" was a β=1 rate artifact
(recon-only β=0.1: 0.976 / R² 0.920 / probes 1.0 — still below the null). **6b:**
the double grid (0.9655 @ 26.3 bits, R² 0.949, probes 1.0) near-dominates every
learned discrete token; blast-radius clause partially triggered. **6c:** outcome
(b), upgraded — the kernel hurts FSQ at every rate (10–24 bits) vs recon-only
FSQ, mechanism = kernel-induced code degeneracy (2 % grid occupancy): the
conflict is quantization per se, not codebook dynamics. The criterion-shaped
discrete program is closed; discrete tiers should bin features, not embeddings.

**Campaign 7 executed (2026-09-01)** as a *supervised* consumer probe on AOJ MC
(QCD vs top, record 18610) rather than the masked-continuous FM probe (which
remains open): see `campaign7_downstream_probe.md`. Verdict: downstream value
tracks information retention (the exact reverse of criterion recall), and the
untrained double grid tops the table at the raw-feature ceiling — settling this
doc's ranked proposal 2 on the supervised side.

### Run plan

`benchmarks/run_campaign6.sh [a|b|eval|all]` — idempotent, sequential (one GPU job
on the shared MIG slice; eval only after training, per the allocator-crash rule).
~17 training runs ≈ 4–5 GPU-hours; eval + figures at the end. Results land in
`benchmarks/figures/library_metrics.json` and `library_rate.png` (all points on
one bits axis), and are written up in `benchmarks/README.md` §Campaign 6.

## 6. References

FSQ: Mentzer et al., arXiv:2309.15505 · BSQ: arXiv:2406.07548 · rotation trick:
arXiv:2410.06424 · SimVQ: arXiv:2411.02038 · REPA: arXiv:2410.06940 · RAE:
arXiv:2510.11690 · MAR: arXiv:2406.11838 · GIVT: arXiv:2312.02116 · JetFormer:
arXiv:2411.19722 · tokenization-free MPM in HEP: arXiv:2409.12589 · OmniJet-α:
arXiv:2403.05618 · MPM: arXiv:2401.13537.
