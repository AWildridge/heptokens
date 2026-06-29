# Continuous Tokenization for HEP Constituents: A Survey and Design Study

> **Status:** design document (no implementation).
> **Scope:** CMS particle-flow constituents, starting from Aspen Open Jets, later
> CMS event-level data.
> **Audience:** `heptokens` developers planning the next-generation tokenizer.
> **Companion:** the implementation roadmap lives in the project plan; this document
> motivates and surveys the design space and defines the benchmark.

---

## 1. Motivation

`heptokens` currently tokenizes constituents with a **residual VQ-VAE**: each
particle is encoded and snapped to a learned codebook, emitting integer indices
(`indices [N, n_csts, num_quantizers]`). Discrete codes are robust — a finite
vocabulary cannot collapse to the identity map — but they are **lossy**: every
particle is rounded to one of `K` vectors per quantizer, which caps reconstruction
fidelity and the expressiveness of anything built on the tokens.

We want representations that are **more expressive and serve downstream tasks
better**, while avoiding the central failure mode of a continuous autoencoder: the
**identity map** (perfect reconstruction, useless representation). The thesis of
this document:

> Modern self-supervised objectives make continuous per-constituent tokens both
> *more expressive than VQ codes* and *collapse-free by construction*, and
> **quantization is not actually required** for our goals.

---

## 2. Do we need quantization?

The word "tokenization" conflates two separable choices. Separating them largely
dissolves the question.

| | Can be continuous? | Requires quantization? |
|---|---|---|
| **Input** to the backbone (per-particle features / embedding) | always | never |
| **Pre-training target / loss** | depends on the objective | only for *categorical* objectives |

Quantization is forced **only** when the training *target* is a class over a finite
vocabulary. Mapping objectives onto that requirement:

| Pre-training objective | Needs VQ? | Examples |
|---|---|---|
| GPT next-token (categorical) / classic masked-token (BERT-MPM) | **yes** | OmniJet-α (2403.05618), MPM (2401.13537) |
| Contrastive / self-supervised (embedding invariance) | **no** | JetCLR (2108.04253), VICReg (2105.04906), jBOT (2601.11719) |
| Masked **continuous** modeling (regress / diffuse the particle) | **no** | *Is Tokenization Needed for MPM?* (2409.12589) |
| Continuous autoregressive (per-token diffusion head) | **no** | MAR (2406.11838), jet next-token hybrid (2512.04149) |

**Empirical anchors:**

- **"Is Tokenization Needed for MPM?" (2409.12589)** replaces the VQ-token target
  with a continuous conditional-generative target and **outperforms** the tokenized
  objective on downstream tagging.
- **MAR (2406.11838, NeurIPS 2024 spotlight)** performs autoregressive/masked
  generation on *continuous* tokens via a per-token **diffusion loss** instead of a
  categorical cross-entropy — removing vector quantization from generation entirely.
- **Jet next-token hybrid (2512.04149)** feeds *continuous* feature vectors as input
  and uses token-IDs only as the generative *target*, combined with masked modeling
  — an explicit decoupling of "tokenized input" from "tokenized target."

**Recommendation.** Decouple input from target. Default to **continuous
representation learning** (contrastive or masked-continuous), which needs no
codebook and tends to win downstream. Reserve discrete tokens as an *optional
target* for a categorical generative branch, where simple softmax sampling is
convenient (and even there, MAR-style continuous generation is an alternative).

---

## 3. Background

**HEP tokenization.** OmniJet-α (2403.05618) established the VQ-VAE → GPT pipeline
for jets; Masked Particle Modeling (2401.13537) used VQ tokens as masked-prediction
targets to learn transferable set representations. Both treat the discrete token as
the model interface.

**The NLP analogy.** A token embedding can be *context-free* (word2vec — the vector
depends only on the token) or *contextual* (BERT — the vector depends on the whole
sequence). The same axis governs particle tokenization and is central to the design
(Section 4).

**Per-particle functions and IRC safety.** Energy/Particle Flow Networks
(1810.05165) learn a per-particle latent `Φ(particle)` summed over the set,
respecting infrared-and-collinear (IRC) safety by construction — the canonical
*context-free* particle representation and a natural backbone.

---

## 4. Design axes

### Axis 1 — Context dependence

| Mode | Pros | Cons |
|---|---|---|
| **Context-free** `Φ(particle)` | universal, transferable, interpretable | ignores inter-particle correlations |
| **Contextual** (transformer over the set) | captures substructure, discriminative | not portable; can entangle pileup/UE |
| **Hybrid / disentangled** | explicit, *measurable* independence knob | most complex |

For a tokenizer whose outputs feed a downstream model that can itself be
context-aware, a **context-free or lightly-contextual** tokenizer keeps the
per-particle representation clean and reusable, and lets the downstream model own
event context.

### Axis 2 — Which invariances define "the same region"

Self-supervised objectives *choose the embedding metric* through the construction of
positive pairs; **choosing positives = choosing what to forget.**

- **Forget** (be invariant to): pileup / underlying event, detector resolution, soft
  radiation (IRC), azimuthal rotation.
- **Preserve** (stay sensitive to): PID, hard `pT`, mass-relevant kinematics, impact
  parameters (d₀, d_z), charge.

This is the precise form of the driving question — *should same-PID,
similar-kinematics constituents embed to the same region, and how independent from
the event?* The answer: **yes for the intrinsic physics; independent of the
soft/instrumental parts of the event.** Independence should be a *measured* quantity
(e.g., mutual information between the token and the pileup level), not an assumption.

---

## 5. Literature survey

**Per-particle / set learning & IRC safety.** Energy/Particle Flow Networks
(1810.05165): Deep-Sets per-particle `Φ`, IRC-safe.

**Discrete tokenization & foundation models.** OmniJet-α (2403.05618, VQ + GPT,
pretrained on Aspen Open Jets); MPM (2401.13537, VQ tokens as masked targets);
*Is Tokenization Needed for MPM?* (2409.12589, continuous targets win); OmniLearned
(2510.24066, cross-domain/detector-agnostic transfer); "Foundation models for HEP"
review (2509.21434); discrete+continuous readout FMs (2505.08736).

**Contrastive / self-supervised in HEP — all per-jet (per-constituent is the gap).**
JetCLR "Symmetries, Safety, and Self-Supervision" (2108.04253) — permutation-
invariant transformer + InfoNCE with physics augmentations (rotations, translations,
soft/collinear). AnomalyCLR and DarkCLR (2312.03067) extend the idea to anomaly
detection; anomaly-preserving contrastive embeddings (2502.15926). **jBOT
(2601.11719)** is the closest prior art to per-constituent: it combines *local
particle-level* with global jet-level self-distillation (DINO-style) and reports
emergent semantic clustering. **RS3L (2403.07066)** builds positive pairs by
*re-simulating* the same event under different shower/detector settings — a
principled, physics-grounded augmentation.

**General SSL & anti-collapse machinery.** Contrastive InfoNCE/SimCLR uses negatives;
**VICReg (2105.04906)** and **Barlow Twins (2103.03230)** avoid negatives via
variance/covariance or redundancy-reduction terms that *prevent representational
collapse directly*. **Alignment & uniformity on the hypersphere (2005.10242)**
provides the evaluation lens (and a directly optimizable objective) for "good"
embeddings.

**Equivariance & symmetry.** LorentzNet (2201.08187) and PELICAN (2307.16506) build
Lorentz symmetry into the architecture — relevant if we prefer equivariance over
augmentation-based invariance.

**Benchmarks.** "An Evaluation of Representation Learning Methods in Particle Physics
Foundation Models" (2511.12829) compares contrastive vs masked vs generative
pre-training head-to-head and is the template for Section 8.

---

## 6. Invariance taxonomy (CMS-internal; cross-experiment = future)

| Invariance | Build positives via | Forget or preserve | Priority |
|---|---|---|---|
| Pileup / underlying event | PUPPI-weight perturbation, pileup overlay (PUMML 1707.08600) | forget | high |
| Detector resolution | pT / angular smearing | forget | high |
| IRC (soft / collinear) | JetCLR-style soft & collinear augmentations | forget | high |
| Azimuthal rotation / beam boost | η-φ rotation, longitudinal boost | forget (selectively) | medium |
| PID / hard kinematics / d₀,d_z / charge | — | **preserve** | — |
| Cross-detector (CMS ↔ ATLAS) | re-simulation (RS3L), domain-adversarial | forget | **future** |

---

## 7. Candidate approaches (surveyed with equal depth)

| # | Approach | Target / loss | Quantization | Primary output |
|---|---|---|---|---|
| 1 | Contrastive / SSL | embedding invariance (InfoNCE or VICReg) | none | continuous representation |
| 2 | Masked-continuous (MPM without tokens) | regress / diffuse masked particle | none | continuous representation |
| 3 | Continuous autoregressive / hybrid | per-token diffusion head (+ optional token target) | optional | representation + generation |
| B | Discrete VQ baseline | categorical token | yes | discrete tokens |

**Within #1 — positive-pair design (the heart of the physics):**

- **Neighborhood contrastive** — positives = same PID within (ΔpT%, ΔR). Yields a
  smooth physical-similarity metric; directly tests the "same region" hypothesis.
- **Pileup / resolution-invariant** — positives = same particle under overlay /
  smearing / re-simulation (RS3L).
- **Soft physics-kernel target** — replace binary positives with a target similarity
  `k(ΔR, ΔpT, PID)` (metric learning with a physics teacher).
- **Collapse prevention** — InfoNCE negatives, or VICReg/Barlow if negative-free.

**Trade-offs.** #1 is the simplest and strong for downstream / anomaly tasks; #2
adds reconstruction fidelity and a generative-flavoured target without a codebook;
#3 buys generation but adds a diffusion head and compute; B is the current baseline
and the easy categorical-generation route.

---

## 8. Benchmark protocol

- **Intrinsic representation quality:** alignment & uniformity (2005.10242);
  collapse diagnostics — embedding rank, per-dimension variance, off-diagonal
  covariance (VICReg-style); effective dimensionality.
- **Physics faithfulness:** does embedding distance track physical distance
  (ΔR, ΔpT, PID)? kNN-purity by PID; recovery of known observables by linear probe.
- **Robustness / independence:** token stability under pileup overlay and resolution
  smearing; mutual information `MI(token; pileup level)`.
- **Downstream (extrinsic):** linear-probe + fine-tune jet tagging (top-vs-QCD,
  JetClass); distance-based anomaly detection (jBOT/AnomalyCLR style); pileup
  robustness of downstream performance.
- **Baselines:** the current `heptokens` VQ-VAE; a continuous reconstruction-only
  autoencoder (to exhibit identity-collapse risk); per-jet JetCLR.
- **Datasets:** Aspen Open Jets (2412.10504) first; CMS event-level later; JetClass
  for domain-shift transfer.

---

## 9. Method comparison

To be scored in a final table on: expressiveness · anti-collapse mechanism ·
context-dependence · transferability · interpretability · pileup robustness ·
generative capability · compute. Rows: VQ baseline · contrastive · masked-continuous
· continuous-AR · self-distillation (jBOT) · equivariant (PELICAN).

---

## 10. Implementation roadmap (tie-back to heptokens)

Each approach reuses existing infrastructure:

- **Encoder/decoder contract** (`models/coders.py`) and the `Transformer` /
  `SetToVectorTransformer` backbones (`models/transformer.py`).
- **Lightning/Hydra plumbing** and `ScheduledOptimiserMixin` (`models/utils.py`,
  `train.py`).
- **Streaming data** for the 178M-jet Aspen Open Jets via
  `StructuredArrayIterableModule` (`data/structured_array.py`).
- **Augmentation hook**: `collate_and_transform` + the `transforms` kwarg; `mask_jet`
  is a ready constituent-dropout primitive (`data/collation.py`).
- **Downstream** `VectorClassifier` already consumes a continuous `z_q`
  (`models/token_classifier.py`) — continuous tokens drop in.
- **Metrics slot**: extend `compute_codebook_utilization` (`models/utils.py`) with
  alignment/uniformity, collapse, and MI diagnostics.
- **New code**: a `LitContrastiveTokenizer` (VICReg default, InfoNCE option), an
  augmentation pipeline (batch transforms), a `CategoricalEmbedEncoder` for PID/charge
  (categoricals must be embedded, not fed as raw floats), and a continuous export path
  (float latents, no codebook table).

**Recommended first build:** Approach #1, neighborhood + pileup-invariant
contrastive with a **VICReg** loss, benchmarked against the VQ baseline on Aspen
Open Jets.

---

## 11. References

All arXiv IDs verified against the source; no fabricated references.

| ID | Work |
|---|---|
| 1707.08600 | PUMML — Pileup Mitigation with Machine Learning |
| 1810.05165 | Energy Flow Networks: Deep Sets for Particle Jets |
| 2005.10242 | Alignment & Uniformity on the Hypersphere (Wang & Isola) |
| 2103.03230 | Barlow Twins |
| 2105.04906 | VICReg |
| 2108.04253 | JetCLR — Symmetries, Safety, and Self-Supervision |
| 2201.08187 | LorentzNet — efficient Lorentz-equivariant GNN |
| 2307.16506 | PELICAN |
| 2312.03067 | DarkCLR / semi-visible jets, energy-based models, self-supervision |
| 2401.13537 | Masked Particle Modeling on Sets |
| 2403.05618 | OmniJet-α |
| 2403.07066 | RS3L — Re-Simulation-based Self-Supervised Learning |
| 2406.11838 | MAR — Autoregressive Image Generation without Vector Quantization |
| 2409.12589 | Is Tokenization Needed for Masked Particle Modelling? |
| 2412.10504 | Aspen Open Jets |
| 2502.15926 | Anomaly-preserving contrastive neural embeddings |
| 2505.08736 | Foundation models for experimental readout (discrete + continuous) |
| 2509.21434 | Foundation models for high-energy physics (review) |
| 2510.24066 | OmniLearned |
| 2511.12829 | Evaluation of Representation Learning Methods in HEP FMs |
| 2512.04149 | Enhancing next-token-prediction pre-training for jet FMs |
| 2601.11719 | jBOT — semantic jet clustering from self-distillation |

AnomalyCLR: <https://github.com/bmdillon/AnomalyCLR-jets>.
