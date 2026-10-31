# Purdue ROOT label pilot, 2026-10-30

The generator-backed label pilot completed **4,000 events on four CPU Dask
workers**, with 24 ordinary TTrees in each source's `labels.root`. Structural
validation and a separate comparison with the stable-only baseline passed.
No 1,000-worker production run has been launched. The interpretation of missing
information was corrected following the user's 31 October review; see
[clarifications and revised label plan](qcd-label-review-20261031.md).

## Inputs, method and scope

Evidence directory (relative to the workspace, outside the Git repository):
`purdue-setup/results/qcd-label-pilot-20261030/`.
The release is `dask-root-gen-v3/`; the earlier `gen-root-v3*` directories are
superseded development outputs and are not releases.

The pilot takes the first 1,000 frozen replay events from each of source tasks
84, 1373, 2785 and 4151: one file per pThat bin. Its 44 staged input objects
passed fresh size/SHA256 verification against immutable inventories
(`input-staging-verified.json`). Each complete generator file is bound to the
completed generator-backfill receipt, both remote full-readback hashes, the
frozen source manifest and a fresh local SHA256.

`src/heptokens/labels/generator.py` joins run/lumi/event and nominal/variation
weights, then checks every packed particle against the original stable replay.
The original exporter preserves packed order within each spatial vertex;
matching uses that ordering, exact PDG/mass/vertex values and P4 tolerance
rtol=2e-12, atol=1e-10 GeV. No geometric nearest-neighbor assignment is used.
The pilot exposed ROOT's signed-mass convention: `M2=m*abs(m)`, not `m*m`.
The corrected energy uses the original ROOT convention; the join tolerance was
not relaxed. Original archived HepMC P4 remains the clustering input.

All retained pruned nodes, status flags, P4, spatial vertices and graph indices
are exposed in `truth/generator_particles`. Selected stable truth particles
link to original packed indices and pruned mothers/last-pruned indices.
`truth/generator_records` retains the full generator archive's URL, SHA256 and
entry, counts, process/PDF metadata, binning values and qScale. The full native
archive preserves all packed particles and generator products; the slim label
file does not duplicate the entire native archive.

Heavy-hadron labels use terminal retained b/c hadrons and the inspected CMS
PDG predicate, ghost-associated to common anti-kt R=0.5 truth jets. There is no
additional heavy-hadron pT cut. Ghost P4 scale is 1e-20, audited at 1e-18 and
1e-22 for every pilot event. Use b precedence; c requires associated charm with
no retained b ancestor. `none` means no retained b or prompt-c association,
not a parton-level light label. Detailed b, c, b-descended-c and prompt-c indices
remain available. The pinned reference sources are saved in `provenance/`.
This is a declared MiniAOD-retained definition, not an unpruned history or a
claim of equivalence to every official CMS flavor convention.

Cuts remain pT>=30 GeV, |eta|<2.5 for truth/reco jets. Truth constituents use
status-1 retained particles, pT>0, |eta|<5, excluding neutrinos and including
muons. The native detector definitions, matching cuts, event IDs and split
rules are unchanged. Added truth labels live in truth/target tables, not reco
input features. C/A trees and angular targets remain off.

## Measured checks

The inspectable records are `dask-production.json`,
`dask-validation-result.json`, `independent-pilot-audit.json`, the per-source
`COMMIT.json` files, and `label-tests-final.txt`.

- 4,000 unique events; 2,069,151 packed particles joined; 408,494 pruned nodes.
- Maximum packed P4 reconstruction difference 5.6844e-14 GeV.
- Identical heavy labels at all three ghost scales. Physical jet memberships
  are identical, with zero P4 change measured within each worker's clustering.
- 13,971 selected truth jets: 574 b, 1,286 prompt-c, 12,111 `none`.
- Matched CMS jets: 567 b, 1,270 c, 11,855 `none`; 482 unmatched.
  Matched ATLAS jets: 533 b, 1,225 c, 11,288 `none`; 424 unmatched.
- All 292 unchanged logical columns/source were compared with the validated
  local stable-only baseline. IDs, splits, weights, physical P4 components,
  constituent memberships and matching assignments agree exactly. Derived
  floating values have recorded machine-level differences between local and
  Dask execution; all pass rtol=atol=1e-12. Largest absolute difference is
  4.657e-10 in GeV-squared mass arithmetic. These are not byte-identical ROOTs.
- Missing object species/ancestry and parton labels remain invalid; pThat is
  available, generator HT and physical-pb weights remain invalid.
- An independent prior native CMSSW FWLite audit agrees on all 3,488 pruned
  nodes and 4,519 selected stable links across 40 matched events: PDG, status,
  P4, mass, vertices, all 15 status bits and graph indices
  (`native-reference-audit.json`). This small cross-check does not establish
  unpruned ancestry recovery.
- 32 targeted tests passed, including negative ROOT mass, failed ancestry
  references/cycles, stable ordering and ghost-scale invariance.

These are unweighted prefix diagnostics, not cross-section-weighted population
flavor fractions, detector closure, flavor efficiency validation or ML closure.
MiniAOD pruning, sample-prefix representativeness and object-definition
limitations remain. No physics uncertainty interval is inferred from these counts.

## Runtime and storage

Measured with inputs already staged on WORK: four-worker controller wall time
114.10 s; source producer wall times 100.54–109.53 s; combined measured CPU
406.89 s; maximum producer RSS 308.75 MiB. ROOT outputs total 89,120,040 bytes.
The generator ROOT is read with explicit MemmapSource. Actual package versions,
Python executable and production module hashes are recorded in metadata.

Scaling each 1,000-event prefix separately to its parent sample gives an
**approximate 4.665 TB decimal** of label ROOT files and 5,920 core-hours with
three ghost scales. At 1,000 continuously busy cores, that is approximately
5.9 hours of computation. It excludes remote staging, duplicate/sum-weight
accounting, final validation, upload/readbacks, scheduling and retries. Larger
partitions may amortize ROOT overhead. No campaign-wide size or 1,000-worker
throughput measurement has been made. The existing 4.56 TB generator archive
is additional storage and remains referenced, not deleted.

`scale-plan.json` records the requested 1,000 single-core, 4 GiB CPU workers
and remaining preflight gates. WORK is for bounded control/evidence, not the
full output; the centrally reported quota was 93.79/100 GB before final release
packaging. Use worker-local scratch, existing shared admission/I/O controls,
isolated source subprocesses, independent retries and canonical two stable
full SHA256 remote readbacks. A prefix success does not validate unbenchmarked
end-to-end remote production at 1,000 workers.

## Clarifications after user review (2026-10-31)

The immutable 30 October pilot measurements above are unchanged. Its original
`missing-information.json` is a historical record, not the current list of
approval gates. The user confirmed the transfer and completion records are on
NERSC, requested light/strange/gluon tags, and accepted the retained-simulation
scope. See the [point-by-point correction](qcd-label-review-20261031.md).

- All 4,546 source identities belong in the final catalog. The Purdue snapshot
  covers only its 4,540 owned sources; it did not establish missing NERSC data.
  Reading and verifying the completed NERSC inventory is an access/catalog task.
- Exact reconstructed constituent contributor links were discarded by our
  compaction. They are normally stored in native Delphes ROOT TRefs. This limits
  exact constituent truth labels, not jet-level parton flavor tagging.
- Light u/d, strange and gluon jet tags can use the retained partons and an
  inspected CMS selection/ghost-association prescription. They are planned for
  the next pilot; ROOT v3 above does not already contain those labels.
- Accepted missing original times/spin/unpruned history remain provenance facts,
  not new approval requirements. C/A shower targets remain deferred.
- Truth/CMS/ATLAS selected-jet HT is already calculated. A literal scalar sum of
  final retained hadron pT can be added separately. Undefined stage/object naming
  for the extra `generator_ht` field was not a fundamental data limitation.

The next implementation checks are the expanded flavor/HT pilot, complete
input catalog, sample-weight accounting and bounded end-to-end I/O validation.
No additional flavor labels, full-campaign completion or 1,000-worker throughput
are established by this documentation correction.
