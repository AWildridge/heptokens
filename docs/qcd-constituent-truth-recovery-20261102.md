# Constituent truth recovery investigation (2026-11-02)

Exact constituent contributor links require native Delphes ROOT output. Where
that ROOT no longer exists, a Delphes replay is needed to regenerate those
links. The existing compact detector dataset need not be replaced: this
investigation demonstrates adding ROOT link sidecars after exact payload checks.
Jet-level flavor labels do not require those constituent contributor links.

## What was lost, and why

Delphes generated the contributor references. Our compaction omitted them:
`tools/compact_delphes.py:19` saves jet/constituent membership offsets, P4,
class and charge, without `Track.Particle` or `Tower.Particles`.
`tools/replay_cms_collection.py:218` excludes native ROOT from the durable file
list, and line 230 removes the temporary detector output after committing the
compact files. Thus this is a retention omission, not an absence of native
Delphes truth bookkeeping. The generator ancestry backfill restores the
particle record, not the already-executed detector's missing associations.

The loss concerns exact native contributor relations. Angular matching remains
an approximate association option, but does not recover discarded references.
A tower can have several contributors; the native list also does not supply
per-contributor energy fractions or a unique species for a mixed tower.

## Surviving ROOT inventory

Evidence directory outside the Git repository:
`purdue-setup/results/constituent-truth-recovery-20261102/`.

`retained-inventory.json` identifies **200 native ROOT files**, both CMS and
ATLAS for **100,000 hard-event identities** (25,000 per source 84, 1373, 2785,
4151). Total native bytes: 9,318,065,367. Every file's fresh SHA256 matches its
saved validation-replay hash; no files were missing from that inventory.
These ROOTs are from the separate matched validation replay, not the original
bulk production's deleted temporary files. Their applicability to the compact
production must be checked per chunk, not assumed from a shared seed.

The native files are under:
`purdue-setup/results/fullsim-delphes-validation-100k-20261021/TASK/chunk-NNNNNN/sample/pair/`.

The inspected Purdue transfer snapshot has 4,540 sources and no outer object
named `.root`; bundled archive interiors are not established by that outer-object
count. The production persistence list and inspected source commit explain why
its standard payload is compact HDF5, compressed truth, cards and receipts.
This is not an inventory of every possible separate NERSC copy.

The user confirms the transfer to NERSC is complete. A read-only SSH attempt
using the already trusted `perlmutter.nersc.gov` host reached the login service
but failed authentication (`nersc-access-check.txt`). Therefore this session
has **not ruled out additional native ROOT backups on NERSC**. Inventory those
before launching a large replay. The old Purdue inventory is not evidence of
missing transferred events.

## Measured recovery pilot

Inputs are the first 1,000 frozen production events from each of the four
sources, both detectors. All native `Jet` and CMS `FatJet` objects are included,
without analysis cuts; the stock ATLAS `FatJet` collection remains empty.
Nominal weights are preserved without physical cross-section normalization.
No detector response, event selection, frozen card or original data changed.

New tool: `tools/recover_delphes_constituent_links.py`.
For each detector/chunk it:

1. Checks fresh archived compact/truth/reference hashes and the fully decoded
   HepMC input SHA256 against the original chunk receipt.
2. Binds the native ROOT to a simulation receipt with matching input hash, seed,
   event count and card hash, then checks the native file SHA256.
3. Uses the canonical `compact_delphes.convert` on that native ROOT and compares
   **all 18 HDF5 datasets**, including dtype, shape and every stored bit pattern,
   with the immutable production compact data. Numerical tolerances are not used.
4. Checks native stable PDGs/P4/XYZ against the archived HepMC at stored float32
   precision and binds native truth indices to the original HepMC barcodes.
5. Resolves every native contributor reference into that event's `Particle`
   collection and emits an additive, ordinary ROOT TTree sidecar.

All eight detector/chunk checks passed for **4,000 unique paired events**.
They recovered **1,032,515 stored constituent rows** and **1,296,243 contributor
edges**, with zero unresolved or empty contributor lists in this pilot.
Counts include repeated appearances in `Jet`/`FatJet` and across detector views;
they are not counts of distinct physical particles. Of these stored rows,
121,100 tower rows have multiple contributors. 2,069,151 stable inputs were checked independently in both views.

The eight sidecars total **18,141,452 bytes**. Producer/extractor run times are
recorded per file in `pilot-results.json`; the bounded ten-job extraction study,
including the two replay checks below, used two CPU processes and 59.61 s wall.
Inputs were already on WORK; this does not measure campaign remote I/O.

The native files for the other 96,000 validation events were hash-inventoried,
but compact parity and link extraction have not yet been checked for those chunks.
No claim of complete 100k link recovery or campaign-wide association closure is made.

## Deterministic replay feasibility

A separate probe replayed **one original 1,000-event chunk** from source 84
through both unchanged default detector configurations:

- original saved stable HepMC input SHA256;
- original archived card and seed-wrapper SHA256;
- seed 504000;
- existing Delphes executable/library, with fresh hashes recorded;
- no new event generation, MiniAOD conversion, pileup overlay or full CMS simulation.

CMS/ATLAS simulation took 6.97/6.02 s respectively with the input already local.
Those times exclude staging, compaction, extraction, publication and full-campaign
validation. The replayed native ROOT file SHA256 values differ from the old files;
ROOT file-level byte identity is neither assumed nor claimed.

Both replayed outputs passed all compact dataset bit-pattern comparisons against
the original production files. Their recovered contributor sets also agree
exactly with those extracted from the retained validation ROOTs: **152,899 CMS**
and **70,243 ATLAS** constituent rows. Contributor list ordering is not a physics
observable (`TreeWriter::FillParticles` uses pointer-sorted sets); comparison
uses sets, and in this probe even the recorded ordering agreed.

Inspect `replay-production.json`, `pilot-results.json` and
`independent-audit.json`. The latter independently checks event IDs/weights,
complete compact-row coverage, jet/constituent offsets, truth-index bounds,
file hashes and retained-versus-replayed contributor sets.
Eight targeted tests passed, including wrong native input/seed/card/hash/event
count, compact value/dtype changes, and signed-zero bit-pattern differences.
This small probe establishes a feasible recovery method, not an unconditional
reproducibility guarantee for all 210,542,000 events or future library versions.

## Recommended recovery procedure

- Read the full NERSC inventory and reuse any surviving native ROOTs. The
  100k validation ROOTs are an available starting point, subject to per-chunk checks.
- For chunks without native ROOT, replay the original **stable detector input**
  with its saved card, seed, chunk boundaries and compatible pinned runtime.
  Do not feed the newly expanded full generator graph into Delphes: that would
  change its input and defeat reproduction of the original detector state.
- Require exact parity of every compact payload before attaching recovered links.
  Quarantine a mismatch; do not substitute a merely similar detector event.
- Write only the additional truth-link sidecar. Join through source, chunk,
  run/lumi/event, collection, jet index and compact constituent row. Preserve
  native truth indices and original stable barcodes, which can then join to
  the generator backfill. Towers retain list-valued references.
- Keep original detector files immutable. Before any future temporary ROOT
  cleanup, commit and verify the native association sidecar as well.

The pilot's sidecars are ordinary TTrees `links` and `events`, with a metadata
TObjString containing source/detector/hash provenance and runtime versions.
They have not yet been integrated into the ROOT v3 label producer. The current
investigation does not start a bulk replay or replace the detector dataset.

The new outputs and proof remain on persistent WORK. EOS publication was not
attempted because the available VOMS proxy reports zero remaining lifetime.
No validated payload or surviving native ROOT was deleted. A renewed proxy is
needed for a new remote publication/readback proof, not for these completed
local extraction and parity checks.
