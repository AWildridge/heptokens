# Purdue replay and pileup diagnostics

This checkout combines the Purdue replay work with the QCD labeling work on
[`fnal-main`](https://github.com/AWildridge/heptokens/tree/fnal-main).
Clone it with:

```bash
git clone --branch fnal-main https://github.com/AWildridge/heptokens.git
```

## Scope and inputs

The replay exports retained stable hard-scatter `packedGenParticles` from the
original CMS QCD MiniAODSIM sources, then runs the pinned Delphes 3.5.1 default
CMS and ATLAS cards. Export and detector validation are implemented in
`tools/export_cms_chunks.py`, `tools/simulate_delphes_pair.py` and
`tools/validate_detector_replay.py`. Production seeds are
`420000 + source_task * 1000 + chunk_index`; event identities and generator
weights are retained. Spatial vertices are converted from cm to mm, while
unavailable production time is approximated as ct=0. Quantization and rapidity
coverage are inherited from the retained particles. Original pileup particles
and full ancestry are unavailable.

The default production response has no added pileup. The separate pilot tools
compare statistical particle overlays with the CMS FullSim reference. The
ordered pilot reproduces recorded in-time counts and the interaction-sharing
schedule through `tools/plan_matched_pileup_schedule.py`,
`tools/materialize_ordered_pileup_shard.py` and
`tools/audit_scheduled_pileup_mapping.py`. Replacing an identity with a new
particle event is not original-particle replay or CMS detector premixing.
Out-of-time detector response is not reproduced.

Plot selections and normalization are encoded in `tools/plot_pileup_pilot.py`
and `tools/plot_matched_pileup_campaign.py`: the matched diagnostic uses four
25,000-event prefixes, LO sample cross-section weights normalized before cuts,
PF pT > 0.5 GeV and |eta| < 2.4 (native ET for neutral Delphes towers).
Representativeness of those prefixes is an assumption. Statistical errors
exclude detector modeling and finite-library correlations. Comparison curves
use distinct colors and dash patterns without markers.

## Execution and evidence

`tools/replay_runtime.py` validates execution context, limits CPU processes and
serializes transfers. The replay, scratch and transfer modules retain identity,
accounting and checksum checks; interrupted or corrupt transfers must preserve
the input. `tools/replay_import_transfer.py` provides offline import and
independent verification. Read its CLI help before choosing an input manifest.

The repository does not include credentials, environments, original data,
generated ROOT or pileup banks, or the scientific evidence releases. At Purdue,
the site checkout is the sibling `purdue-setup/` directory, and durable evidence
is under its `results/` directory, including:

- `fullsim-delphes-validation-100k-20261021/`
- `matched-pileup-pilot-100k-20261028/`
- `pileup-generator-settings-20261028/`

Those paths are external evidence locations, not portable files supplied by
this Git repository. Campaign coordinators and some plotting utilities retain
Purdue paths and expect this site layout. The calorimeter patch under
`patches/` is an optional source artifact; storing it does not apply it to the
frozen runtime or alter the baseline cards.

## Code validation

Use an existing CPU analysis environment with pytest, NumPy, h5py, Awkward,
uproot and the project dependencies. From this repository:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/test_replay_*.py tests/test_gfal_transport.py \
  tests/test_delphes_replay.py tests/test_omnilearned.py
```

Here `python` means the interpreter of that activated environment; on Purdue,
use the built project environment or the global analysis interpreter, as
required by the site instructions. These tests use synthetic local fixtures;
they do not launch a production campaign or establish detector/ML closure.
`test_storage_watchdog_controls.py` additionally requires the separate Purdue
site checkout and skips when it is absent. Test paths resolve from the
repository itself, without requiring the checkout to be named `heptokens`.

The Git synchronization preserves the imported source snapshot and merges it
with the existing `fnal-main` history. Its backup, hashes and verification logs
stay in the external `purdue-setup/results/git-sync-20261029/` directory.
Synchronizing code does not rerun or revalidate the scientific releases.
