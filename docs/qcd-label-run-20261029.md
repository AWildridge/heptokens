# Slim QCD preprocessing run — 2026-10-29

**Validated 4,000-event prefix; the 210,542,000-event campaign has not been fully preprocessed.**

This note records the historical **Parquet v1** implementation and measurements.
The current plan uses ROOT TTrees (`qcd-labels-root-v2`) readable by uproot; see
[the production instructions](qcd-label-production.md). The measurements below
have not been remeasured for ROOT and must not be quoted as ROOT validation or
ROOT storage/throughput estimates.

The working set includes 1,000 events from each of source tasks 252, 1152, 2546 and 4515, covering all four generator pthat bins. C/A trees and angular targets are postponed. No model training ran.

## Data and reports

- Scratch: `/pscratch/sd/a/anwildri/research/datasets/QCD-multijet-paired/current/`
- Separate input entry points: `.../QCD-multijet-paired/CMS/` and `.../QCD-multijet-paired/ATLAS/`.
- Durable verified backup: `/global/cfs/cdirs/m2612/treasure/datasets/heptokens-anwildri/QCD-multijet-paired/labels/slim-v1-validation-20261029/`.
- That directory contains `validation.md`, `validation.json`, `RUN-RECORD.json`, plots and exact run logs.
- Full definitions, CLI, reader and Purdue commands: [QCD label production](qcd-label-production.md).
- A portable `fnal-main-20261029.bundle` is archived in the durable backup's `code/` directory for transfer without GitHub credentials. Clone it with `git clone -b fnal-main /path/to/fnal-main-20261029.bundle heptokens`.

## Measured size and compute

Parquet tables: **52.95 MiB**. The sample-composition-weighted projection is **2.65 TiB** for all events; budget 3–4 TiB until a broader production benchmark confirms it.

The final run used 104.6 CPU-seconds for 4,000 events. Its full-campaign projection is **1528 CPU core-hours**, excluding full validation and site-specific I/O effects. Peak worker RSS was 203 MiB. The same-value Cartesian fast path improved CPU time by 2.42x; all 88 output tables matched the baseline after excluding provenance metadata.

| Table | Rows |
|---|---:|
| truth/jets | 13,844 |
| truth/particles | 502,862 |
| inputs/CMS/jets | 14,081 |
| inputs/CMS/constituents | 365,271 |
| inputs/ATLAS/jets | 13,375 |
| inputs/ATLAS/constituents | 271,776 |

## Validation

22 focused tests passed; five existing dataset-adapter regressions passed separately. Ruff lint and shell syntax checks passed. All source payload hashes, event joins, stable-state fingerprints, split assignments, constituent memberships, charges and one-to-one matches validated on the working set. The durable backup verified 117 regular files byte for byte.

| Sample | Detector | Selected reco matched | Common truth covered |
|---|---|---:|---:|
| 63193 | ATLAS | 96.87% | 94.25% |
| 63193 | CMS | 96.58% | 98.49% |
| 63217 | ATLAS | 97.18% | 93.30% |
| 63217 | CMS | 96.38% | 97.84% |
| 63229 | ATLAS | 96.79% | 93.45% |
| 63229 | CMS | 96.15% | 98.16% |
| 63233 | ATLAS | 97.07% | 93.82% |
| 63233 | CMS | 97.24% | 98.49% |

## Limits and next work

- The samples are **pthat**, not HT, bins. Generator HT and per-event pthat were not retained; particle and reconstructed HT are distinct saved values.
- Heavy-hadron/parton flavor, reco-object truth TRefs, original ancestry/pileup/time and hard-process/incoming labels are unavailable. Their targets remain masked; discrete flavor probes need additional original generator truth.
- Physical cross-section weights remain invalid until a full measured sum-of-weights accounting pass. No unit-weight or equal-bin normalization assumption was used.
- Synchronization is hash-bound to archived independent ROOT truth-content checks. Compact HDF5 no longer supports a fresh independent detector-truth comparison.
- Full accounting, sidecar production and validation belong on Purdue CPU allocations; batch and Dask commands are prepared. C/A remains deferred.

Label producer revision: `e0f2d41fe07de86ae233568c01a39c58cb8397ca`. Each table and run record pins its actual producer software/configuration and environment hashes.
