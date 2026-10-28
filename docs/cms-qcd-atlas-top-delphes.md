# Paired default-card Delphes replays: CMS QCD and ATLAS Top Tagging

2026-10-13. User constraint: **only replay original dataset events**; no newly
generated substitutes. All downloads of physics files, conversion, detector
simulation, validation and artifact compression ran on Slurm allocation
59751211 (nid001400). Independent ATLAS inspection and CMS artifact preparation
ran concurrently as separate steps. The paired runner also runs CMS and ATLAS
concurrently within one 4-CPU step; a check on the saved 100-event input passed.
Explicit GPUs were requested for all steps
on the GPU allocation; Delphes itself used CPUs. The CMS Pixi manifest/lock and
its EAF references remain unchanged. No training or inference was performed.

## Completed CMS QCD sample

The same 1,000 original CMS QCD source events were run through **both** unchanged
Delphes 3.5.1 default detector cards. A wrapper sets detector RNG seed 42.

| Input source | Events | CMS default ordinary jets | ATLAS default ordinary jets |
| --- | ---: | ---: | ---: |
| CMS QCD Pt 300–470, TuneCP5, 13 TeV, UL16 MiniAODv2 | 1,000 | 4,032 | 3,840 |

Before scaling to 1,000, a 100-event pilot passed on both cards (400 CMS jets,
364 ATLAS jets). Counts include all jets emitted by the default cards, not an
analysis selection. These are exploratory replay/software checks, not validated
physics results and not the entire CMS QCD dataset.

Source: [CERN Open Data record 63193](https://opendata.cern.ch/record/63193),
`/QCD_Pt_300to470_TuneCP5_13TeV_pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM`.
Selected source file:
`270000/4FEB1BEF-87BD-5A4B-83A3-48916581AB45.root`.
Its 204,059,105 bytes and Adler32 `8dd24a90` were checked against the public
record before conversion; SHA256 is recorded in `source.json`. The smallest
listed file was selected to bound transfer costs, and its first 1,000 events
were used. This is an original source-file sample, not a representative physics
sample or a claimed row-for-row match to OmniLearned's selected jet shards.

FWLite in the pinned CMSSW_10_6_30 Shifter container exported
`packedGenParticles` (stable hard-scatter particles) and the nominal
`GenEventInfoProduct` event weight. Original run/lumi/event IDs are retained.
All exported particles were required to have status 1 and finite values.
Momentum units are GeV; vertex coordinates were converted from cm to mm.

MiniAOD packed truth is quantized and rapidity-limited; it is not the original
full HepMC history. Production time is not retained by the exported packed
representation, so ct is explicitly zero. The pruned collection's existence
and counts were recorded, but its genealogy was not reconstructed. Original
pileup was not replayed, and no new pileup was generated. No ancestry-based jet
flavor labels are valid for these stable-only exports. Nominal raw weights are
preserved; no luminosity, cross-section or QCD-bin normalization was applied,
and generator systematic weights are not exported by this pilot workflow.

The stock CMS ordinary jets use R=0.5; the stock ATLAS jets use R=0.6. They
therefore differ from the original CMS AK8 reconstructed-jet dataset. These
files are default-card fast simulations, not replacements for CMS full
simulation or a detector-only comparison at identical jet definitions.

Validation checked every event: identical stable-particle count and IDs,
four-momenta, masses, vertices and represented times (rtol=1e-5, atol=1e-7),
nominal weights (rtol=1e-6), and finite positive reconstructed jet pT with finite
eta/phi/mass. Both cards passed for all 1,000 events. ROOT timestamp metadata
can vary on reproduction; compare physical arrays at the stated tolerances.

Reusable scratch outputs:

- `/pscratch/sd/a/anwildri/research/runs/cms-qcd-local-delphes-59751211/events-1000.hepmc`
- `/pscratch/sd/a/anwildri/research/runs/cms-qcd-local-delphes-59751211/events-1000.hepmc.json`
- `/pscratch/sd/a/anwildri/research/runs/cms-qcd-local-delphes-59751211/default-pair-1000/CMS.root`
- `/pscratch/sd/a/anwildri/research/runs/cms-qcd-local-delphes-59751211/default-pair-1000/ATLAS.root`

Durable validation artifacts and code/environment snapshots:
`perlmutter-setup/dataset-runs/cms-qcd-delphes-59751211/`.
`replay-artifacts.tar.gz` contains saved truth inputs, references, both pilots'
ROOT outputs, exact default cards, seeds, logs and validation summaries.
The raw MiniAOD stays on scratch and is reproducible from its recorded URL and
checksum. No shared project storage was modified. Full dataset production
requires a defined analysis, representative selections, systematic/normalization
handling and an approved durable destination for large outputs.

## ATLAS Top Tagging: original inputs unavailable

The [original tagging release](https://opendata.cern.ch/record/80030), its
[official documentation](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data),
and the [simulation paper](https://arxiv.org/html/2407.20127v2) were inspected.
The nominal signal uses Z-prime to ttbar at 2 TeV; nominal background uses QCD.
These underwent ATLAS Geant4 full simulation and UFO R=1.0 jet reconstruction.
The public release contains reconstructed objects, not original EVNT/HepMC.

A direct Slurm inspection downloaded and verified the original
`train_nominal_928.h5.gz` (19,879,805 bytes, Adler32 `557968ed`). Its 20,427 rows
contain 28 fields, including `EventInfo_mcEventNumber`,
`EventInfo_mcEventWeights`, `fjet_clus_{pt,eta,phi,E,taste}`, jet summary
quantities, labels and training weights. There is no full generator-particle
collection, PDG list, particle statuses or production-vertex record.
Event numbers and shower weights alone cannot reconstruct those missing inputs.
The field inventory and source provenance are saved in `atlas-inspection.json`
under the durable CMS run directory.

The ATLAS HepMC [summary record 160000](https://opendata.cern.ch/record/160000)
currently links only specialised Z-boson generation (record 160002). A CERN
catalogue query for ATLAS Z-prime samples returned no matches. No matched
original top-tagging EVNT/HepMC collection was located. An ATLAS Open Data
representative also [explains the reduced truth content of reconstructed public
formats](https://opendata-forum.cern.ch/t/particle-level-truth-information/570).

Consequently **neither CMS nor ATLAS Delphes output for the original ATLAS Top
Tagging dataset was produced**. Feeding reconstructed UFOs into Delphes would
apply detector effects again; generating new Z-prime/QCD samples would violate
the user's explicit constraint. The required next input is the original event
collection (EVNT converted to HepMC, HepMC, or an equivalent complete generator
particle record) used to create that release, with event/sample identifiers and
weights. No messages were sent to dataset authors.

## Separate reproducible stages

From an allocated compute shell, first source `perlmutter-setup/env.sh`.
Export original CMS truth with CMSSW, without activating Pixi in the container:

```bash
srun -N 1 -n 1 -c 4 --mem=8G --gpus=1 \
  shifter --image=cmsopendata/cmssw_10_6_30-slc7_amd64_gcc700:latest --module=cvmfs \
  bash "$RESEARCH_CODE/perlmutter-setup/cmssw-exec.sh" \
  python "$RESEARCH_CODE/heptokens/tools/export_cms_miniaod_hepmc.py" \
  --input /path/to/original-miniaod.root --output /fresh/path/events.hepmc --events 1000
```

Simulate both detectors using saved inputs; a fresh output directory is required:

```bash
srun -N 1 -n 1 -c 4 --mem=8G --gpus=1 \
  pixi run --locked \
  --manifest-path "$RESEARCH_CODE/heptokens/environments/cms-gpu/pixi.toml" \
  python "$RESEARCH_CODE/heptokens/tools/simulate_delphes_pair.py" \
  --input /saved/path/events.hepmc \
  --delphes "$RESEARCH_SCRATCH/software/Delphes-3.5.1" \
  --output /fresh/path/default-pair --seed 42
```

`events.hepmc.json` must contain the corresponding particle/weight reference;
the paired runner checks its input hash before running. Inference and plotting
must reuse these saved ROOT/truth outputs, with separate configurations and
commands; no inference/plotting is required for this replay task.


## Full-source campaign

The subsequent [full four-dataset campaign](full-dataset-campaign.md) submits
all 4,546 original CMS QCD sources with both default detector cards, conditioned
on verified complete local prepared datasets. Compact training views live on
scratch with durable CFS backups and saved compressed truth; full ROOT files
are temporary. Bulk jobs are queued, not reported as completed results.
