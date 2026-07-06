# Continuous Tokenization for HEP Constituents: A Survey and Design Study

> **Status:** design document, revision 2 (post red-team audit).
> **Mission:** produce **AI-ready token datasets** — experiments (CMS/ATLAS) publish
> per-constituent embeddings/tokens as the shareable artifact; foundation-model
> builders train on tokens *without* PF-candidate access or LHC-reconstruction
> knowledge.
> **Scope:** CMS particle-flow constituents, starting from Aspen Open Jets, later
> CMS event-level data.
> **Audience:** `heptokens` developers planning the next-generation tokenizer.

---

## 1. Motivation

`heptokens` currently tokenizes constituents with a **residual VQ-VAE**: each
particle is encoded and snapped to a learned codebook, emitting integer indices
(`indices [N, n_csts, num_quantizers]`). Discrete codes are robust — a finite
vocabulary cannot collapse to the identity map — but they are **lossy**: every
particle is rounded to one of `K` vectors per quantizer, which caps fidelity and
the expressiveness of anything built on the tokens.

We want a token interface that is **more expressive, collapse-free, and sufficient
for downstream tasks that are not known in advance** (tagging, anomaly detection,
observable regression, possibly generation). The thesis of this document:

> Modern self-supervised objectives make continuous per-constituent tokens both
> *more expressive than VQ codes* and *collapse-free by construction*, and
> **quantization is not required** for representation goals — its remaining
> legitimate role is post-hoc release compression.

## 1b. The mission changes the design problem

Because the **product is the token dataset itself**, the tokenizer is not an
optional ML trick — it is a *data-release interface*. That brings requirements a
pure ML experiment does not have:

| Requirement | Consequence |
|---|---|
| Information sufficiency for unknown future tasks | benchmark must probe tagging **and** observable recovery **and** anomaly detection |
| Detector abstraction | hiding reconstruction detail is a **feature** — consumers need no CMS internals |
| Frozen, versioned contract | version tokens like a data format (dim, preprocessing hash, checkpoint id) |
| Release size at 178M-jet scale | **post-hoc** compression (fp16, product quantization) preferred over baking VQ into training |

---

## 2. Do we need quantization?

The word "tokenization" conflates two separable choices. Separating them largely
dissolves the question — but a second distinction (Section 2b) matters just as much.

| | Can be continuous? | Requires quantization? |
|---|---|---|
| **Input** to the backbone (per-particle features / embedding) | always | never |
| **Pre-training target / loss** | depends on the objective | only for *categorical* objectives |

| Pre-training objective | Needs VQ? | Examples |
|---|---|---|
| GPT next-token (categorical) / classic masked-token (BERT-MPM) | **yes** | OmniJet-α (2403.05618), MPM (2401.13537) |
| Contrastive / self-supervised (embedding invariance) | **no** | JetCLR (2108.04253), VICReg (2105.04906), jBOT (2601.11719) |
| Masked **continuous** modeling (regress / diffuse the particle) | **no** | *Is Tokenization Needed for MPM?* (2409.12589) |
| Joint-embedding prediction (JEPA) | **no** | JP-JEPA (2606.14813), HEP-JEPA (2502.03933), J-JEPA (2412.05333) |
| Continuous autoregressive (per-token diffusion head) | **no** | MAR (2406.11838), jet next-token hybrid (2512.04149) |

**Empirical anchors.**

- *Is Tokenization Needed for MPM?* (2409.12589) replaces the VQ-token target with
  a continuous conditional-generative target and **outperforms** the tokenized
  objective on downstream tagging.
- MAR (2406.11838, NeurIPS 2024 spotlight) performs autoregressive/masked
  generation on *continuous* tokens via a per-token **diffusion loss** — removing
  vector quantization from generation entirely.
- The jet next-token hybrid (2512.04149) feeds *continuous* feature vectors as
  input and uses token-IDs only as the generative *target*, and adds masked
  modeling because next-token prediction alone under-delivers on classification.
- VQ carries documented pathologies: codebook collapse, **dimensional collapse
  (a hard loss floor)** and index collapse at large codebooks (2411.16550,
  2605.06870 — vision/speech evidence; `heptokens`' own
  `compute_codebook_utilization` metric is internal corroboration that collapse is
  a live concern).

**Recommendation.** Train **continuous**; keep VQ only as a benchmark baseline.
For release-size concerns, compress post-hoc (fp16 / product quantization) instead
of accepting VQ's training-time pathologies. If a *categorical* generative branch
is ever required, prefer MAR-style continuous-AR first.

## 2b. Tokenizer-level vs backbone-level objectives (a category distinction)

MPM and GPT/NTP are objectives for pretraining the **downstream FM backbone**, not
for training a tokenizer: OmniJet-α's VQ-VAE exists *only* to give the backbone a
categorical target. The two questions must not be conflated:

- **Tokenizer** (per-constituent, frozen, shared — *our product*): needs a good
  **local** embedding. Candidate objectives: contrastive with physics positives,
  VICReg, or a distilled encoder from backbone-style pretraining.
- **Backbone / FM** (event- or jet-level, context-aware — *the consumer's model*):
  pretrained with masked-continuous / JEPA / (optionally NTP) objectives **over
  the tokens**.

A consequence for evaluation: whether "MPM/GPT-style is advantageous downstream"
is a *backbone* question; whether "quantization is needed" is a *tokenizer
interface* question. Our benchmark (Section 8) tests both in the setting that
matters: which pretraining yields the best **interface**.

## 2c. The null reference: no learned tokenizer at all

A PF candidate is ~8–11 numbers. With `latent_dim ≥ input_dim`, an identity map is
achievable and *not even harmful* for a pure interface — so the marginal value of
a learned per-constituent tokenizer over **preprocessed continuous features + a
linear embedding layer** must be *demonstrated, not assumed* (that null is
effectively what 2409.12589 / 2512.04149 use, successfully). Under the AI-ready
dataset mission the tokenizer stage is mandatory anyway (interface, abstraction,
sharing), so the null is retained as a **reference ceiling**: it quantifies the
information cost of the interface relative to full PF access.

---

## 3. Background

**HEP tokenization.** OmniJet-α (2403.05618) established the VQ-VAE → GPT pipeline
for jets; Masked Particle Modeling (2401.13537) used VQ tokens as masked-prediction
targets to learn transferable set representations. Both treat the discrete token as
the model interface.

**The NLP analogy.** A token embedding can be *context-free* (word2vec) or
*contextual* (BERT). The same axis governs particle tokenization (Section 4).

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
| **Contextual** (transformer over the set) | captures substructure, discriminative | less portable; can entangle pileup/UE |
| **Hybrid / disentangled** | explicit, *measurable* independence knob | most complex |

For a released token interface consumed by context-aware FMs, a **context-free or
lightly-contextual** tokenizer keeps the per-particle representation clean and
reusable, and lets the downstream model own event context.

### Axis 2 — Which invariances define "the same region"

Self-supervised objectives *choose the embedding metric* through positive-pair
construction; **choosing positives = choosing what to forget.**

- **Forget** (be invariant to): pileup / underlying event, detector resolution,
  soft radiation (IRC), azimuthal rotation.
- **Preserve** (stay sensitive to): PID, hard `pT`, mass-relevant kinematics,
  impact parameters (d₀, d_z), charge.

**Caveat (from the audit):** some augmentations are *nullified by preprocessing* —
with η_rel/φ_rel coordinates, azimuthal rotation is already absorbed. The
augmentation set must be checked against the coordinate and preprocessing choices,
or it trains nothing.

---

## 5. Literature survey

**Per-particle / set learning & IRC safety.** Energy/Particle Flow Networks
(1810.05165).

**Discrete tokenization & foundation models.** OmniJet-α (2403.05618); MPM
(2401.13537); *Is Tokenization Needed for MPM?* (2409.12589); OmniLearned
(2510.24066); "Foundation models for HEP" review (2509.21434); discrete+continuous
readout FMs (2505.08736).

**JEPA in HEP (added in revision 2 — previously missing).** Joint-Embedding
Predictive Architectures predict *latent* representations of masked particles from
context, with an EMA target encoder preventing collapse — no tokenization, no
reconstruction, and (J-JEPA) no hand-crafted augmentations. JetParticle-JEPA
(2606.14813) reports strong JetClass results (author-reported); HEP-JEPA
(2502.03933) and J-JEPA (2412.05333) evaluate top tagging and quark/gluon
discrimination. A leading candidate for backbone pretraining, and its frozen
encoder is a candidate token producer.

**Contrastive / self-supervised in HEP — all per-jet (per-constituent is the
gap).** JetCLR (2108.04253); AnomalyCLR; DarkCLR (2312.03067); anomaly-preserving
contrastive embeddings (2502.15926); jBOT (2601.11719 — local particle-level +
global jet-level self-distillation, the nearest prior art to per-constituent);
RS3L (2403.07066 — re-simulation as principled augmentation).

**General SSL & anti-collapse machinery.** InfoNCE/SimCLR (negatives); VICReg
(2105.04906) and Barlow Twins (2103.03230) — collapse prevention without
negatives; alignment & uniformity (2005.10242) as the evaluation lens.

**Caveat (from the audit): InfoNCE is methodologically suspect at constituent
level.** Two soft pions from *different* jets are physically near-identical;
in-batch negatives wrongly repel them, at huge rate (~10⁴ constituents per batch).
VICReg (no negatives) or a physics-kernel SupCon (same-PID/ΔpT/ΔR pairs are
positives by construction) are the defensible variants.

**Equivariance & symmetry.** LorentzNet (2201.08187); PELICAN (2307.16506).

**Benchmarks.** "An Evaluation of Representation Learning Methods in Particle
Physics Foundation Models" (2511.12829): with a shared ParT-style encoder, *fully
supervised* training is best overall and **supervised contrastive** is the
strongest representation objective. **Nuance: the winner uses labels** — the paper
(as accessible to us) does not establish a ranking among purely *self-supervised*
objectives, and all of its evidence is **jet-level**, not per-constituent.
Extrapolating any SSL ranking to per-constituent tokenizers is unvalidated; it can
only be settled by our own benchmark (Section 8).

---

## 6. Invariance taxonomy (CMS-internal; cross-experiment = future)

| Invariance | Build positives via | Forget or preserve | Priority |
|---|---|---|---|
| Pileup / underlying event | PUPPI-weight perturbation, pileup overlay (PUMML 1707.08600) | forget | high |
| Detector resolution | pT / angular smearing | forget | high |
| IRC (soft / collinear) | JetCLR-style soft & collinear augmentations | forget | high |
| Azimuthal rotation / beam boost | η-φ rotation, longitudinal boost (check not absorbed by coordinates) | forget (selectively) | medium |
| PID / hard kinematics / d₀,d_z / charge | — | **preserve** | — |
| Cross-detector (CMS ↔ ATLAS) | re-simulation (RS3L), domain-adversarial | forget | **future** |

---

## 7. Candidate approaches

| # | Approach | Level | Target / loss | Quantization |
|---|---|---|---|---|
| 0 | Null reference: preprocessed features + linear embed | — | — | none |
| 1 | Contrastive tokenizer (VICReg / physics-SupCon) | tokenizer | embedding invariance | none |
| 2 | Masked-continuous pretraining, encoder as tokenizer | backbone → tokenizer | regress masked particle | none |
| 3 | JEPA pretraining, encoder as tokenizer | backbone → tokenizer | predict latent of masked particle | none |
| B | Discrete VQ (current heptokens) | tokenizer | categorical reconstruction | yes |

**Within #1 — positive-pair design (the heart of the physics):**

- **Neighborhood contrastive** — positives = same PID within (ΔpT%, ΔR); yields a
  smooth physical-similarity metric; directly tests the "same region" hypothesis.
- **Pileup / resolution-invariant** — positives = same particle under overlay /
  smearing / re-simulation.
- **Soft physics-kernel target** — target similarity `k(ΔR, ΔpT, PID)`.
- **Collapse prevention** — VICReg or physics-SupCon (vanilla InfoNCE demoted; see
  Section 5 caveat).

---

## 8. Benchmark protocol (consumer-view)

Every probe sees **only tokens** — simulating the FM builder who has no PF access:

- **Downstream tagging:** linear + small-transformer probes; transfer to JetClass /
  top-tagging (AOJ itself has **no truth labels** — it is real collision data, so
  on-AOJ evaluation must be label-free).
- **Observable recovery:** regress jet pT, mass, N-subjettiness from token sets —
  physics fidelity through the interface.
- **Representation quality:** kNN-PID purity; alignment & uniformity (2005.10242);
  collapse diagnostics (effective rank, per-dim variance, off-diag covariance).
- **Robustness:** token stability under smearing / pileup proxies;
  MI(token; pileup).
- **Release size:** bytes per jet at equal downstream performance.

**Arms:** 0 / 1 / 2 / 3 / B from Section 7, with a shared encoder budget, shared
preprocessing, and a shared probe protocol. Arms 2/3 test whether *backbone-style*
pretraining yields a better **interface** than tokenizer-native objectives —
directly answering "is MPM/GPT-style advantageous?" in the setting that matters
here.

**Datasets:** Aspen Open Jets (2412.10504; 178M CMS jets, ≤150 constituents with
4-momentum + PID + charge + d₀/d_z + uncertainties) first; CMS event-level later;
JetClass for labeled transfer.

---

## 9. Evidence-integrity register (audit)

- All content claims about papers rest on **search snippets**; full texts were not
  fetchable in the working environment. arXiv IDs come from arXiv's own listings
  (trustworthy); claims about *contents* carry snippet-level confidence.
- JP-JEPA's performance claim (2606.14813) is author-reported abstract text.
- 2511.12829's headline is triangulated across two independent sources; its
  supervised-contrastive winner **uses labels**, and no SSL-only ranking is
  established.
- VQ-collapse citations (2411.16550, 2605.06870) are from vision/speech.

| Claim | Verdict |
|---|---|
| Quantization unnecessary for representation goals | **Solid** (2409.12589 direct experiment; 2512.04149 continuous inputs; MAR existence proof; VQ pathologies) |
| GPT/NTP alone suboptimal for discrimination | **Solid-ish** (2511.12829; 2512.04149 adds MPM to fix classification) |
| Contrastive/JEPA best *unsupervised per-constituent* objective | **Unproven** — jet-level evidence only; needs our benchmark |
| Learned tokenizer beats null (features + linear embed) | **Unproven** — null retained as reference ceiling |

---

## 10. Implementation roadmap (tie-back to heptokens)

- **Encoder backbone:** `Encoder` + `Transformer` (`models/coders.py`,
  `models/transformer.py`); PID/charge are categorical → embed, don't feed as
  floats.
- **Streaming data** for AOJ: `StructuredArrayIterableModule`
  (`data/structured_array.py`).
- **Augmentation hook:** `collate_and_transform` + the `transforms` kwarg;
  `mask_jet` as constituent-dropout primitive (`data/collation.py`).
- **Downstream probes (the existing heptokens benchmarks):** `FeatureClassifier`
  (raw features — the null reference probe), `VectorClassifier` (continuous
  tokens), `TokenClassifier` (discrete tokens) in `models/token_classifier.py`.
- **Metrics slot:** extend `models/utils.py` (beside `compute_codebook_utilization`)
  with alignment/uniformity, collapse, kNN-PID-purity.
- **Export:** continuous analog of `export_tokens.py` — float latents, no codebook
  table; version stamp (latent dim, preprocessing hash, checkpoint id).

**First build:** the contrastive tokenizer arm (VICReg default; physics-SupCon
option), benchmarked against the existing heptokens VQ-VAE and classifier probes;
masked-continuous and JEPA arms follow on the same harness.

---

## 11. References

All arXiv IDs verified against arXiv listings; content claims carry the Section 9
caveats.

| ID | Work |
|---|---|
| 1707.08600 | PUMML — Pileup Mitigation with Machine Learning |
| 1810.05165 | Energy Flow Networks: Deep Sets for Particle Jets |
| 2005.10242 | Alignment & Uniformity on the Hypersphere (Wang & Isola) |
| 2103.03230 | Barlow Twins |
| 2105.04906 | VICReg |
| 2108.04253 | JetCLR — Symmetries, Safety, and Self-Supervision |
| 2201.08187 | LorentzNet |
| 2307.16506 | PELICAN |
| 2312.03067 | DarkCLR / semi-visible jets, energy-based models, self-supervision |
| 2401.13537 | Masked Particle Modeling on Sets |
| 2403.05618 | OmniJet-α |
| 2403.07066 | RS3L — Re-Simulation-based Self-Supervised Learning |
| 2406.11838 | MAR — Autoregressive Image Generation without Vector Quantization |
| 2409.12589 | Is Tokenization Needed for Masked Particle Modelling? |
| 2411.16550 | Representation Collapsing Problems in Vector Quantization |
| 2412.05333 | J-JEPA — Jet-Based Joint Embedding Predictive Architecture |
| 2412.10504 | Aspen Open Jets |
| 2502.03933 | HEP-JEPA |
| 2502.15926 | Anomaly-preserving contrastive neural embeddings |
| 2505.08736 | Foundation models for experimental readout |
| 2509.21434 | Foundation models for high-energy physics (review) |
| 2510.24066 | OmniLearned |
| 2511.12829 | Evaluation of Representation Learning Methods in HEP FMs |
| 2512.04149 | Enhancing next-token-prediction pre-training for jet FMs |
| 2601.11719 | jBOT |
| 2605.06870 | Continuous First, Discrete Later: VQ-VAEs Without Dimensional Collapse |
| 2606.14813 | JetParticle-JEPA |

AnomalyCLR: <https://github.com/bmdillon/AnomalyCLR-jets>.
