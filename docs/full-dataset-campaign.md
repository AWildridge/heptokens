# Full four-dataset staging and original CMS QCD replay

Submitted 2026-10-13. Dataset validations now run independently (updated 2026-10-14).
Configuration, frozen file inventories, tools, job IDs and experiment record:
`perlmutter-setup/data-campaigns/four-datasets-20261013/`.

| Stage | Slurm job | Required predecessor |
| --- | --- | --- |
| Download all four prepared datasets | 59753955, tasks 0–11, four concurrent | None |
| Aggregate verification of all prepared shards | 59753966 | Aspen retry 59778150 successful |
| Replay all CMS QCD source files with both cards | 59753968, tasks 0–4545, four concurrent | CMS QCD validation 59778429 |
| Verify complete paired training collections | 59753969 | All source tasks successful |

Permanent root:
`/global/cfs/cdirs/m2612/treasure/datasets/heptokens-anwildri/`.
Prepared datasets: `omnilearned-full/{aspen,jetclass,jetclass2,cms_qcd}/{train,val,test}`,
2,377 files, 6.05 TiB. JetClass, JetClass-II and CMS QCD transfers are complete; Aspen has 69/80
files. Retry 59778150 downloads only its remaining validation split.
Independent validation jobs are JetClass 59778424, JetClass-II 59778427,
CMS QCD 59778429, and Aspen 59778430 (after its retry). Each writes its own
`DATASET/READY.json`. CMS replay checks only `cms_qcd/READY.json` and does not
wait for Aspen or either JetClass dataset. `omnilearned-full/READY.json` is
published only after every frozen shard is complete. Transfers use pinned
ETags, resumable partial files, full-file SHA256 receipts, and HDF5 checks.
Small earlier software samples remain separate on scratch.

Existing shared `AspenOpenJet` has 79 reconstructed-data HDF5 files;
`JetClass/Pythia` has all 1,250 original jet ROOT shards (1000/50/200 across
train/val/test). These differ from the prepared OmniLearned schema and were
not overwritten. `AspenOpenJetMC` contains QCD and BSM HDF5 files, but this
account cannot read them; `JetClass/Ali` is also inaccessible. Their contents
cannot establish generator availability. No shared permissions were changed.

Durable replay: `cms-qcd-default-pair-20261013/sources/SOURCE/chunk-NNNNNN/`.
Scratch training root:
`/pscratch/sd/a/anwildri/research/datasets/cms-qcd-default-pair-20261013/`.
Training shards: `training/SOURCE/chunk-NNNNNN/{CMS,ATLAS}.h5`.
Both durable and scratch replay roots receive `READY.json` only after full accounting.
Only original CMS QCD MiniAOD generator collections are eligible for the
production Delphes array. Missing packed/gen collections fail export before
simulation. Aspen is real data; JetClass and JetClass-II prepared shards do
not provide complete event-level generator inputs and are validated only.

The source manifest pins 4,546 original MiniAOD files, 210,542,000 events,
across CMS records 63193, 63217, 63229, 63233 (pT 300–470, 470–600,
600–800, 800–1000 GeV). Each source is downloaded temporarily to scratch and
checked against its published Adler32 and size. FWLite exports all retained
stable hard-scatter particles, including neutrinos, in 1000-event chunks.
There are no event selections or newly generated substitute events.
Packed truth is quantized and rapidity-limited; original pileup, ancestry and
production time are unavailable. Positions are converted cm→mm, ct=0.

Delphes 3.5.1 uses unchanged stock CMS and ATLAS cards, concurrently per chunk.
Seed = 420000 + source-task*1000 + chunk-number. Every output must preserve
event count, stable-particle identities/P4/positions and nominal weights.
Generator and available LHE variation weights remain in durable truth metadata;
they are not cross-section or luminosity normalized. Source cross sections
and uncertainties remain in the frozen manifest. No truth flavor labels are
invented from stable-only inputs.

The compact HDF5 schema retains every default `Jet` and `FatJet` object and
all resolved constituents, with exact ragged event/jet/constituent offsets,
Cartesian P4 in GeV (float32), constituent class and charge, event IDs, and
float64 nominal weights. ATLAS's stock card has no FatJet branch; it is stored
as an empty collection. Default ordinary jet radii differ: CMS R=0.5,
ATLAS R=0.6. This is not a matched large-R analysis or an isolated detector
variation with otherwise identical reconstruction.

Choose `datamodule=cms_qcd_delphes_cms` or `cms_qcd_delphes_atlas` after
readiness. Both select ordinary `Jet`. Source URI plus event identity defines
a shared SHA256 80/10/10 event split, preventing the two detector views from
crossing training/evaluation boundaries. The training adapter uses Cartesian
P4, sorts constituents by pT and pads/truncates to 128 for batches; the saved
shards retain every constituent. Do not mix these P4 features with the prepared
OmniLearned delta-eta/delta-phi/log-pT/log-E features without an explicit transform.
Training refuses incomplete replay roots by default. No training was submitted.

Only after both detector validations and byte-verified durable/training copies
are committed are new temporary ROOT files removed. Compressed HepMC retains
particle truth once; compressed metadata retains IDs and all saved weights.
Cards, seeds, validation reports, hashes, source inventories, tool snapshots and recorded dependency revisions,
environment locks and code revision are archived under durable `provenance/`.
Scratch copies can be restored from CFS without re-running simulation.

CFS had 15.25 TiB free at submission. Prepared data consumes 6.05 TiB;
the replay has a locked 7 TiB budget, leaving about 2 TiB project headroom.
The 1000-event pilot compressed truth to 22.4 MB and both compact detector
views to 3.47 MB. This size extrapolation is provisional; the storage guard
stops rather than overruns its budget. Shared project usage can change.
Full ROOT outputs are not retained: their estimated 17 TiB alone exceeds
the intended storage layout. HPSS's current 1 TiB user quota is insufficient.

For a prepared training working set, copy a selected dataset from CFS to
scratch from an allocated shell, then override `datamodule.root`:

```bash
source /global/u2/a/anwildri/code/perlmutter-setup/env.sh
srun -n 1 -c 4 --gpus=1 rsync -a --partial --info=progress2 \
 /global/cfs/cdirs/m2612/treasure/datasets/heptokens-anwildri/omnilearned-full/jetclass/ \
 "$RESEARCH_SCRATCH/datasets/omnilearned-full/jetclass/"
```

Monitor `squeue -u anwildri`; inspect failures with `sacct` and the campaign's
`download-*`, `replay-*` logs. A failed/timed-out task prevents downstream
readiness. Resubmit only affected task indices using the same frozen sbatch
script; committed chunks and transfer receipts are reused. Re-submit the
appropriate verification job with an `afterok` dependency on successful retries,
then update downstream dependencies; an original failed dependency does not
automatically become successful. Do not remove committed chunks or the ledger.

To restore a durable truth chunk for a separate replay, run
`tools/restore_cms_chunk.py --chunk DURABLE_CHUNK --output FRESH/events.hepmc`
in the locked CMS environment on Slurm, then use `simulate_delphes_pair.py`
with the receipt's seed. Plotting/inference should reuse saved compact shards.

Validation allocation 59753182: five loader tests passed; 2001 original events
passed both detector truth/weight checks across chunks of 1000,1000,1, including
saved constituent/P4 comparisons. Exploratory checks establish software
correctness on these samples, not validated physics performance. Failed pilots
are recorded in the experiment log. CMS manifest/lock remain unchanged.
Eight-hour full-array submission exceeded the remaining allocation; the
four-hour limit passed site policy. Jobs use shared GPU allocations because
the CPU allocation is exhausted; Delphes itself is CPU work. Every srun
requests its GPU explicitly. No ATLAS Top original events were located and
no replacement events were generated.


## Replay recovery, 2026-10-14

Original array 59753968 produced 128,000 committed paired events before
CERN HTTP 503/504 failures, Pixi startup failures, and four compact-output
failures on Muon constituents. One complete source and all committed chunks
are preserved. The unstarted tasks were held and replaced by a batch launch
using the installed CMS environment (`--locked --no-install`).

See the current job IDs in `data-campaigns/four-datasets-20261013/jobs.json`
and exact failure/recovery commands in `recovery-20261014.json`. Five original
sources (1,2,3,5,6) test the recovery with two concurrent tasks. The replacement
full array waits for every recovery task to pass. It excludes completed source
0 and reuses committed chunks. Detector simulation ROOT pairs from failed
preprocessing are reused only when input, seed, detector-card and output hashes
match. Muon/Electron/Photon constituent classes preserve their existing P4
and memberships. Source transfers can fall back to the identical published
XRootD URI; size and Adler32 validation remain mandatory. No detector card,
physics selection, generator input or environment lock was changed. Recovery
was submitted; full successful recovery is not yet claimed.


## Additional pre-scale tests, 2026-10-14

Preflight 59785092 checks transport/storage fault injection, training readiness
and the four real failed Muon chunks (both cached detector outputs, no repeat
simulation). Array 59785126 tests up to 2001 original events from each of the
four pT bins, including live XRootD fallback, detector truth/weight checks,
compact outputs, matched event IDs/splits, and an idempotent completed restart.
The existing five-source recovery 59784948 also waits for preflight. Gate
59785127 requires all these jobs and their reports to pass against identical
production code/input/environment hashes. Full array 59784970 now depends on
that gate. Tests were submitted; outcomes are not yet claimed. Exact inputs,
commands, hashes, resource requests and job IDs are recorded in
`pre-scale-tests-20261014.json` and durable provenance.


## CPU packing for faster replay, 2026-10-15

Pilot 59808050 processes eight disjoint original sources in one shared-GPU
allocation with 32 CPU threads and 48 GiB memory; only two source transfers
may run at once within a job, controlled by node-local locks. The unchanged
tested physics/export/validation code is checked by SHA256 before launch.
Packed array 59808107 has 567 groups, at most eight sources per group, and
four concurrent groups (32 source files concurrently). It waits for pilot
success. The previous unstarted single-source array was cancelled to avoid
duplicate work and free submit slots. Pilot sources and five already completed
sources are excluded from the packed groups. Source 6 remains isolated behind
its existing CFS lock, which was not removed or bypassed; that source remains
required by final readiness checks and no events are silently dropped.

This uses eight times the prior processing concurrency within the same
four-GPU allocation footprint. A provisional 4-7 processing-day estimate needs
confirmation from packed throughput; queue delays, storage contention and
retries remain additional. Exact scheduling, resource evidence, code hashes
and job IDs are in `packed-optimization-20261015.json` and durable provenance.
