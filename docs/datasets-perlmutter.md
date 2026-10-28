# Aspen Open Jets, JetClass, JetClass-II and CMS QCD on Perlmutter

Setup validated on 2026-10-13. Native lazy loader:
`heptokens.data.omnilearned.OmniLearnedModule`. Hydra datamodules:
`aspen`, `jetclass`, `jetclass2`, `cms_qcd`. Uses OmniLearned's public prepared HDF5
files without installing its separate ML stack. CMS environment manifest and
lock remain byte-identical to the EAF reference; always use the explicit CMS
manifest. All validation and compilation used Slurm compute nodes.

## Current bulk campaign

Full four-dataset staging and the all-source CMS/ATLAS replay jobs are submitted.
See [the storage layout, job IDs, readiness checks and training interfaces](full-dataset-campaign.md).
The full prepared-data configs point to the dedicated CFS directory under
`/global/cfs/cdirs/m2612/treasure/datasets/heptokens-anwildri/omnilearned-full`.
Each dataset is independently usable after its dataset-level READY.json.
The root READY.json indicates all four are complete. CMS QCD replay depends
only on its own validation; datasets without event-level generator inputs
receive data validation only.

## Earlier small-sample validation

Scratch root: `/pscratch/sd/a/anwildri/research/datasets/omnilearned`.
There are **1,024-row software-validation samples**, one shard per dataset per
train/val/test split (9,216 rows total), not complete datasets. Original source
URLs, shard lengths, row ranges, HTTP ETags, output hashes and job ID are in
`stage-20261013T202400067383Z.json`, also backed up under
`perlmutter-setup/dataset-runs/setup-20261013/`.

The loader opens local HDF5 shards lazily per worker, preserves supplied splits,
and does not load the full dataset or a billion-row index into memory. It emits
`csts` (first four upstream data columns: delta-eta, delta-phi, log-pT, log-E),
`mask`, and `jets` (upstream `global`, passed through). Additional columns are
available as `pid_codes` and `add_info`. Upstream PID codes are categorical;
they must not be interpreted as PDG identifiers by the particle encoder.
The upstream transformed global columns differ between datasets (Aspen: 4,
JetClass/I-II: 3); they are not raw jet four-vectors. No physical-unit inverse
transformation or additional scaler is assumed. No scaler was fitted.

The validity mask follows upstream's `data[...,2] != 0` rule. This preserves its
padding convention, including its ambiguity for a physical log-pT value of
exactly zero. No re-selection, particle reordering, weight normalization or
augmentation is applied. JetClass labels use fixed offset 2 and 10 classes;
JetClass-II uses fixed offset 12 and 188 classes. Aspen's source bucket 200 is
not emitted as a truth label. These prepared shards have no physics weights.

Four focused tests passed (including multiworker access, shard boundaries,
fixed labels across splits, and rejection of nonfinite input). All nine real
samples passed Hydra construction, finite-input checks and batch loading.
These are software checks, not physics validation or a training benchmark.
First-row samples are not representative physics samples. Existing model test
failures from the EAF handoff remain outside this dataset change.

## Use from an allocated compute shell

```bash
source /global/u2/a/anwildri/code/perlmutter-setup/env.sh
cd "$RESEARCH_CODE/heptokens"
# Verify samples or already-staged full shards.
srun -N 1 -n 1 -c 4 --gpus=1 pixi run --locked \
  --manifest-path "$RESEARCH_CODE/heptokens/environments/cms-gpu/pixi.toml" \
  python tools/validate_omnilearned.py \
  --root "$RESEARCH_SCRATCH/datasets/omnilearned"
```

For model configuration, select `datamodule=aspen`, `datamodule=jetclass`, or
`datamodule=jetclass2`; override `datamodule.root` for another staging root.
For example, configuration-only inspection (no training):

```bash
srun -N 1 -n 1 -c 4 --gpus=1 pixi run --locked \
  --manifest-path "$RESEARCH_CODE/heptokens/environments/cms-gpu/pixi.toml" \
  heptokens-train datamodule=jetclass2 --cfg job
```

Full shards use exactly the same loader. The following command stages all
prepared shards for the selected datasets and all three splits. Use a fresh
root: samples must never be mixed into a full-data directory. `--rows 0`
downloads complete shards; `--max-files 0` selects all listed shards. The full
download was **not run during the initial small-sample setup**. Use the
[resumable submitted campaign](full-dataset-campaign.md) for the current full transfer. Define the analysis and validate its selections,
units, labels, normalization and uncertainty treatment before scaling up.

```bash
srun -N 1 -n 1 -c 4 --gpus=1 pixi run --locked \
  --manifest-path "$RESEARCH_CODE/heptokens/environments/cms-gpu/pixi.toml" \
  python tools/stage_omnilearned.py \
  --root "$RESEARCH_SCRATCH/datasets/omnilearned-full" \
  --datasets aspen jetclass jetclass2 --rows 0 --max-files 0
```

Staging fails rather than overwriting existing files and records failed attempts.
Do not use partial `.partial` files for training. Use a fresh root after a
failed staging attempt, or resume only after verifying each existing shard
against its recorded provenance. Large file downloads need a suitably sized
allocation/runtime. Valuable full inputs/results require a project-approved
durable backup destination; scratch is purgeable.

## Detector provenance and the missing ATLAS simulation

| Dataset | Original detector treatment | Retained public generator information |
| --- | --- | --- |
| Aspen Open Jets (`aspen`) | Real CMS 2016 collision data; no detector simulation | None: reconstructed collision data cannot be converted into generator events |
| JetClass | MadGraph + Pythia + Delphes with CMS detector configuration and additional impact-parameter smearing; anti-kT R=0.8 | Jet ntuples and selected truth information; no full-event HepMC release located |
| JetClass-II | MadGraph + Pythia + CMS-based Delphes cards, track smearing, mean pileup 50 and PUPPI; AK8 and AK15 in official production | Parquet `genpart_*`: matched GEN-jet constituents (neutrinos excluded), positions and PDG IDs; `aux_genpart_*`: selected truth particles; a ten-event full-event Delphes example in the generation repository |

Sources: [Aspen release](https://www.fdr.uni-hamburg.de/record/16505),
[JetClass detector description](https://huggingface.co/datasets/jet-universe/jetclass/blob/main/README.md),
[JetClass-II data specification](https://huggingface.co/datasets/jet-universe/jetclass2/blob/main/README.md),
[JetClass-II generation scripts](https://github.com/jet-universe/jetclass2_generation),
[OmniLearned data tooling](https://github.com/ViniciusMikuni/OmniLearned).
Aspen's separate [MC companion](https://www.fdr.uni-hamburg.de/record/18610)
is processed CMS Open Simulation, not the real-data `aspen` download. It is not
included in this setup's samples; QCD-bin combination requires the release's
`pt_weights.py`. Its original CMS MINIAODSIM sources are listed on that record.

The missing detector for JetClass and JetClass-II is ATLAS. The public
JetClass-II Parquet GEN particles are a jet-selected subset, not a complete
collision record. They omit neutrinos, out-of-jet particles and event genealogy;
replaying them cannot reproduce full-event detector simulation. OmniLearned's
prepared HDF5 files discard these truth branches altogether.

A more complete input was found at
`perlmutter-setup/jetclass2-generation-source/delphes_analyzers/events_delphes_higgs2p_example.root`:
**ten full-event examples** with `Particle` generator branches. Stable non-PU
particles, including neutrinos, were exported to HepMC2 in GeV/mm; ROOT's stored
time in seconds was converted to ct in mm. Shower genealogy was not restored.
Both default Delphes 3.5.1 ATLAS and CMS cards were run on these inputs. Detector
card files are unchanged; a wrapper sets RNG seed 42. This is a ten-event
hard-scatter software replay with no original pileup, not an ATLAS version of
the full JetClass-II release. It cannot provide ancestry-dependent flavor tags.

Default ATLAS uses R=0.6 jets, default CMS uses R=0.5 ordinary jets (and a CMS
fat-jet module with R=0.8). They do not reproduce the original JetClass-II
pileup/PUPPI and jet definitions. Any downstream comparison must account for
these differences rather than assume a detector-only variation.

Replay passed exact event/particle-count checks, particle IDs, momenta and
vertices within rtol=1e-6/atol=1e-7, finite positive reconstructed jet pT and
preserved event weights. Results, default cards, seeds, input/output checksums,
source revisions and unchanged environment locks are retained at:
`perlmutter-setup/dataset-runs/jetclass2-example-59750270-20261013T202952/`.
ROOT file timestamps can change byte hashes on rerun; compare physics arrays
within the recorded tolerances. No physics inference or plotting was performed.

To rebuild/replay after scratch loss, from an allocation:

```bash
srun -N 1 -n 1 -c 8 --gpus=1 pixi run --locked \
  --manifest-path "$RESEARCH_CODE/heptokens/environments/cms-gpu/pixi.toml" \
  bash "$RESEARCH_CODE/perlmutter-setup/build-delphes.sh"
srun -N 1 -n 1 -c 4 --gpus=1 pixi run --locked \
  --manifest-path "$RESEARCH_CODE/heptokens/environments/cms-gpu/pixi.toml" \
  bash "$RESEARCH_CODE/perlmutter-setup/replay-default-detectors.sh"
```

Full dataset ATLAS production remains unperformed: no public original full-event
HepMC or full Delphes event collections were located beyond the ten-event
example. The official JetClass-II `run.sh` overwrites intermediate HepMC each
batch and deletes its work directory after saving Delphes ROOT output. A retained
full Delphes production ROOT collection (with `Particle`) or archived HepMC
from the dataset authors would be the next input to obtain. New generation
would be a separate production, not reuse of the original dataset events.

Follow-up: [CMS QCD and ATLAS Top Tagging paired Delphes replays](cms-qcd-atlas-top-delphes.md).
Includes completed original CMS QCD source-event samples with both default
cards and direct verification of the missing ATLAS generator inputs.
