# Slim CMS/ATLAS QCD analysis dataset

The default produces a self-contained, column-prunable dataset on scratch,
with reconstructed inputs, truth targets and bookkeeping in separate tables.
No models are trained. C/A trees and angular targets are **postponed by default**
at the user's request; enable them only in a separate, versioned production.

## Locations and status

Durable input:

`/global/cfs/cdirs/m2612/treasure/datasets/heptokens-anwildri/QCD-multijet-paired/`

- `raw/`: original campaign, moved by one directory rename; event files are unchanged.
- `CMS/sources.jsonl`, `ATLAS/sources.jsonl`: detector-specific catalogs of all 4,546 sources.
- `labels/`: namespace reserved for durable label backups.
- The former `cms-qcd-default-pair-20261013` path is a compatibility symlink to `raw/`.

Scratch outputs are under:

`/pscratch/sd/a/anwildri/research/datasets/QCD-multijet-paired/`

Each version has its own `metadata.json`, source partitions, `COMMIT.json`
checksums, `validation.json`, `validation.md`, plots and `READY.json`.
`READY.json` enumerates only validated partitions and explicitly states whether
the **entire campaign** is complete. A validation prefix is not the full dataset.
Never combine different output versions without checking their contracts.

The initial slim benchmark used 256 events and 3.48 MB, extrapolating to about
2.6 TiB for the 210,542,000-event campaign. Small files have extra overhead;
the larger validation run's `resources` report supersedes this estimate.
Scratch is purgeable; retain code, locks, provenance and validation reports
durably and regenerate or explicitly back up the working set.

## Inspected inputs and important corrections

The source manifest records **generator pthat bins, not HT bins**:

| CMS record | pthat bounds [GeV] | Source events | Published cross section [pb] |
|---|---:|---:|---:|
| 63193 | 300–470 | 55,264,000 | 6,832 |
| 63217 | 470–600 | 52,408,000 | 551.9 |
| 63229 | 600–800 | 65,088,000 | 156.5 |
| 63233 | 800–1000 | 37,782,000 | 26.25 |

Frozen source records and the official GEN configurations were inspected.
`docs/label-provenance/` preserves the four Pythia8/CP5 `HardQCD:all` fragments,
full GEN configurations, source URLs and SHA256 checksums. They specify
`PhaseSpace:pTHatMin/Max`. Published cross sections in the source manifest
are used; the fragments' configured cross-section annotations differ and
are not substituted for the published values. Published cross sections
already include the filter efficiency; do not apply that factor twice.

The campaign exporter retained packed stable MiniAOD particles, nominal and
available generator/LHE variation weights, run/lumi/event and production
positions. It omitted pruned particles, mother/daughter links, incoming
flavors, hard-process identifiers and event-level generator pthat/HT.
Stable truth is quantized and rapidity-limited. Original pileup is absent;
vertex time was set to zero and is explicitly invalid as a measured target.

The stock cards and original compacting/validation tools were inspected:

- CMS `Jet`: anti-kt R=0.5 from `EFlowMerger/eflow`.
- ATLAS `Jet`: anti-kt R=0.6 from calorimeter towers.
- Both cards start jet reconstruction at 20 GeV; the analysis applies the
  common 30 GeV, abs(eta)<2.5 selection below.
- CMS also has FatJet; ATLAS does not. This slim production uses `Jet` only.

Consequently this comparison changes both detector response and the stock
reconstruction definition. It is not an isolated detector-only variation.

## Synchronization, IDs, splitting and weights

Each chunk requires a complete receipt and passed archived detector validation.
The producer freshly verifies compressed truth, reference, CMS/ATLAS HDF5,
cards and validation SHA256 hashes against the receipt. It checks the shared
HepMC input digest and joins each detector's events by run/lumi/event, checking
export IDs and weights. It checks fresh HepMC particle counts against both
detectors' archived per-event truth counts.

The original validator compared particle PDG, P4, vertices and weights in the
independent ROOT outputs, before those ROOT files were discarded. HDF5 no
longer contains independent truth branches: **fresh independent detector-truth
comparison is unavailable**. This limitation is recorded, rather than claiming
that matching event order establishes synchronization. Archived content checks
are freshly hash-bound to the preserved files.

Synchronization is certified at the retained stable-particle state, which
contains the original shower/hadronization outcome. It is stronger than a
shared hard event alone, but does not establish any unretained shower genealogy.
Only targets derived from that stable state are shared between detectors.

- `generator_event_id`: length-delimited SHA256(namespace, sample, run, lumi, event).
  It excludes detector and source filename, grouping replicas together.
- `truth_fingerprint`: SHA256 of lexicographically sorted PDG/P4/mass/position
  rows, independent of particle order.
- `event_id`: SHA256(generator_event_id, truth_fingerprint), so differing stable
  states are never assigned a shared truth target.
- Truth jet IDs include event, the common clustering definition and sorted
  constituent barcodes; toggling trees or matching cuts does not change them.
- Particle key: `(event_id, particle_index)`; reco constituent key:
  `(jet_id, constituent_index)`. Barcodes/indices refer to immutable source
  records, not inferred identities of calorimeter particles.
- Split: first 64 hash bits of generator_event_id modulo 100; 0–79 train,
  80–89 validation, 90–99 test. Both views, every object and every replica
  share this group split. The old source-dependent split is retained separately
  as `original_split` for audit; use the new bookkeeping split for analysis.

Original weights remain float64. Cross section and boundaries are preserved.
Generator HT and event pthat are null with false validity masks. The sample
selection boundaries are not event measurements. Physical stitching weight is
`original_weight * cross_section_pb / measured_sample_sum_weights`.
It is null until a complete, replica-aware weight-accounting pass validates
the sample denominator. Never assume unit weights from event counts or equal
normalization across the four bins. Histogram files retain signed sumw,
sumw2 and under/overflow; raw-weighted plots are diagnostic shapes.

## Physics definitions

Common truth jets use FastJet anti-kt, R=0.5, E-scheme recombination. Select
retained status-1, non-pileup particles with pT>0 and abs(eta)<5; exclude
abs(PDG) in {12,14,16,18}; include muons by default. The invisible list and
muon treatment are configurable. These are retained-particle jets, not a
claim to recover generator particles outside MiniAOD's acceptance.

Select truth and reconstructed jets with pT>=30 GeV and abs(eta)<2.5.
Keep every event, including events with no selected jets. Store constituent
membership, multiplicity and truth charged multiplicity. The latter is
invalid if any constituent's charge cannot be obtained from the pinned PDG
particle database. The default saves only particles belonging to selected
truth jets and constituents of selected reconstructed jets; full stable
truth and unselected jets remain in the original files.

Jet/event observables use float64; constituent/particle Cartesian P4 uses
float32 to match the compact input precision and reduce storage. Derived
constituent eta/phi/pt are computed on demand by vector. The original compressed
truth retains its original precision. `mass2_raw` and `mass_valid` flag materially
spacelike jet/tree P4; numerical negative mass2 within 1e-6*max(1,E²) is clamped
to zero. Event/jet continuous targets remain unbinned.

Event HT is the scalar pT sum of the selected jets. Save multiplicity,
leading-dijet invariant mass, signed y1-y2, and phi1-phi2 wrapped to [-pi,pi].
Leading jets are ordered by pT, then stable ID. Dijet targets are masked when
fewer than two jets exist or rapidity is invalid. Truth HT, reconstructed HT,
generator HT and the generator pthat sample boundaries are separate quantities.

Each detector is matched independently to common truth jets using a global
maximum-cardinality, minimum-total-deltaR one-to-one assignment. Candidates
require eta/phi deltaR<0.25 and 0.25<=pT_reco/pT_truth<=4 by default. Dummy
assignment cost prioritizes cardinality before distance. Candidate multiplicity
on either endpoint sets ambiguity, which is retained by default and can be
rejected by configuration. Save truth IDs, deltaR, pT/mass responses,
unmatched/ambiguous flags and truth-jet coverage. Cross-detector associations
are joins through truth IDs; pT rank does not establish correspondence.

## Unavailable labels and what would recover them

All missing targets have explicit masks/reasons in metadata and tables:

- Heavy-hadron flavor: no decayed b/c hadrons or ancestry are retained.
  Never classify their stable daughters or reconstructed b-tags as truth flavor.
  A future re-export should retain terminal weakly decaying b/c hadrons and
  mother links for ghost association. Record charm from b separately;
  use b precedence for b-containing jets and exclude b-descended charm from
  prompt-c counts, while retaining detailed hadron PDG IDs and ancestry flags.
  That is a future definition, not an available label in this version.
- Generator-associated parton flavor: no supported parton association exists.
  Need outgoing hard partons/status flags and ancestry with an explicit
  association definition. Current label is `unknown`, not light flavor.
- Reco-object species/ancestry: compacting discarded `Track.Particle`,
  `Tower.Particles` and other truth TRefs. Need original ROOT or a reference-
  preserving re-compaction. The association schema is list-valued so a tower
  can retain many truth particles; no one-to-one particle labels are invented.
  Available truth species, charge and P4 remain in the common truth table.
- Incoming flavors, hard-process IDs, generator HT/pthat, original pileup/time:
  need retained GenEventInfoProduct/LHE/pruned/GEN information as appropriate.
  Synthetic HepMC header zeros are not recorded hard-process labels.
- Helicity, density matrices and spin-on/off labels are not manufactured.

Discrete flavor probes cannot be supported by this stable-only replay as it
stands. Kinematic and matching studies are available; flavor and generator-
shower studies require restoring additional truth information.

## Optional postponed C/A stage

Set `include_trees: true` in a new output's JSON config. This is deliberately
off in the scratch slim dataset. The implementation and synthetic tests retain
the entire ragged binary C/A tree, including soft branches, using exclusive
Cambridge–Aachen at FastJet's maximum radius to force one root per input jet.
C/A distances use rapidity/phi and E-scheme recombination. Store every node's
parents/children, membership, P4, mass, deltaR, z=min(pT1,pT2)/(pT1+pT2), and
kT=min(pT1,pT2)*deltaR. Reco trees use each detector's own constituents and
are never identified with truth tree nodes.

All angular momenta use the common lab frame, with the beam along z.
For A->B+C, evaluate secondary branchings on **both** possible C daughters.
Within each secondary split D is the harder daughter, ties broken by sorted
constituent indices; E is the other. For the chosen C, B is its sibling.
Use n1=unit(pB cross pC), n2=unit(pD cross pE), and
psi=atan2(unit(pC) dot (n1 cross n2), n1 dot n2). Save psi, cos(2psi), sin(2psi).
Plane sine below 1e-10 is degenerate. By default both branchings must satisfy
z>=0.02, deltaR>=0.001, kT>=0.5 GeV for a resolved angle; raw nondegenerate
angles and distinct geometric/resolution masks are preserved.
q->qg/g->gg/g->qqbar categories remain unknown: a C/A tree is not shower history.

## Run and analyze

On this installation, the unchanged explicit CMS environment supplies the
analysis stack. Three pinned wheel dependencies live in the separate
`/global/u2/a/anwildri/code/label-production-deps` overlay. The launcher limits
BLAS/OpenMP to one thread per worker. The root Pixi workspace and CMS lock
were not relocked. On Purdue, install `environments/labels/requirements-cpu.txt`
in a CPU environment, then set `LABEL_PYTHON` and `LABEL_DEPS` appropriately;
the repository's `src` directory is added by the launcher. It requires no torch.

```bash
bash tools/run_qcd_labels.sh produce \
  --campaign /path/to/QCD-multijet-paired/raw \
  --output /path/to/scratch/slim-v1 \
  --source-task 252 --source-task 1152 --source-task 2546 --source-task 4515 \
  --max-chunks 1 --events-per-chunk 1000 --workers 2

bash tools/run_qcd_labels.sh validate --output /path/to/scratch/slim-v1 --plots
```

The login-node guard allows at most four sources, one chunk each, up to
1000 events per chunk, and two single-threaded workers. The full campaign is
not appropriate for a shared login node. Do not bypass this guard by setting
a fake Slurm environment variable.

Tables are Parquet, organized as `source-NNNNN/{bookkeeping,truth,inputs/CMS,
inputs/ATLAS,targets/CMS,targets/ATLAS}/TABLE.parquet`. Arrow schemas include
units and configuration/software hashes, including typed empty postponed tables.
Use column pruning and small Awkward batches:

```python
from heptokens.labels.analysis import detector_batches, scan_table

for batch in detector_batches('/path/to/scratch/slim-v1', 'CMS'):
    jets = batch['inputs']['jets']             # vector Momentum4D
    constituents = batch['inputs']['constituents']  # full ragged arrays
    keys = batch['keys']                       # IDs, outside feature arrays
    # No truth targets, weights or sample IDs are included in these inputs.

book = scan_table('/path/to/scratch/slim-v1', 'bookkeeping/events',
                  columns=['event_id', 'sample_id', 'split', 'original_weight'])
for record_batch in book.to_batches():
    pass  # join bookkeeping explicitly when doing weighted analysis
```

## Full production on Purdue AF

Run metadata-only weight accounting first in a CPU batch allocation:

```bash
bash tools/run_qcd_labels.sh account-weights \
  --campaign "$QCD_CAMPAIGN" --output /path/to/durable/weight-accounting-v1
```

It streams retained reference weights, verifies their hashes/source identities,
deduplicates sample/run/lumi/event in SQLite, checks complete source/event
coverage, and writes `normalization.json`. The sample denominator is valid
only when all expected sources and unique events are accounted for. The
on-disk identity database can be large; allocate scratch for it.

Then submit `tools/qcd_labels_cpu.sbatch` after setting `QCD_REPO`,
`QCD_CAMPAIGN`, `QCD_OUTPUT`, `LABEL_PYTHON`, `LABEL_DEPS`, and optionally
`QCD_NORMALIZATION` to that JSON file. Supply Purdue's actual account and CPU
partition to `sbatch`. It requests one CPU/4 GiB per source worker, a 4-hour
limit and at most 16 concurrent tasks. These are conservative starting values,
not measured Purdue performance. Increase concurrency after an I/O benchmark.
If the site's array limit is below 4546, submit smaller array ranges. Retry
completed tasks safely: file/config/software hashes are checked before reuse.

For an existing Dask cluster with shared input/output storage and the same
code/environment on every worker:

```bash
bash tools/run_qcd_labels.sh produce --campaign "$QCD_CAMPAIGN" \
  --output "$QCD_OUTPUT" --all-sources --dask-address tcp://SCHEDULER:PORT \
  --normalization /path/to/weight-accounting-v1/normalization.json
```

Only source-level jobs are distributed; each worker streams chunks and writes
one partition using atomic publication and a source lock. The full validation
pass also belongs in a CPU batch job: it scans the produced data in 32-event
batches and uses a disk-backed global replica/split check. It does not load
the entire campaign into memory. Plotting is separate and reuses the saved
validation histograms. No jobs are submitted to Purdue by this implementation.

Costly/deferred work: the full input decompression and clustering pass,
global sum-of-weights/duplicate accounting, full validation scans, and
especially optional C/A trees/full membership output. Restoring missing
pruned ancestry or Delphes TRefs requires source re-export or replay; it
cannot be recovered by an Awkward transform of the current compact files.

Run focused tests with the label Python/overlay:

```bash
PYTHONPATH="$LABEL_DEPS:$PWD/src" "$LABEL_PYTHON" -m pytest tests/test_qcd_labels.py -q
```
