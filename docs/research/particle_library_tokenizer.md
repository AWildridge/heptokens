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

## 8. Roadmap

1. **Iteration 2:** kinematic-preservation term; re-run the AOJ campaign.
2. **Release contract:** version tokens (dim, preprocessing hash, checkpoint
   id) via `export_latents.py`; post-hoc compression (fp16 / PQ).
3. **Cross-experiment:** mixed CMS + second-detector batches under the same
   criterion (no paired events needed); measure cross-sample mixing in the
   library metrics.
4. **Downstream:** train FM backbones (masked-continuous / JEPA) on the
   released tokens; consumer-view probes per the survey's benchmark protocol.
