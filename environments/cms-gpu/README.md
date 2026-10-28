# CMS GPU environment for EAF

A fresh Linux Pixi environment migrated from
`/home/anwildri/cms_gpu_environment-handoff.tar.gz` (captured 2026-09-21).
The repository's original root-level Pixi workspace is independent of this one.
This environment also installs the local `heptokens` project in editable mode.

## Use

From the repository root:

```bash
pixi shell --manifest-path environments/cms-gpu/pixi.toml
```

Or run commands without opening a shell:

```bash
pixi run --manifest-path environments/cms-gpu/pixi.toml python your_script.py
pixi run --manifest-path environments/cms-gpu/pixi.toml check
pixi run --manifest-path environments/cms-gpu/pixi.toml test
```

Inside this directory, plain `pixi shell`, `pixi run check`, and `pixi list`
select this workspace. From the repository root, always supply `--manifest-path`
to avoid selecting the original environment.

For the EAF Jupyter interface:

```bash
pixi run --manifest-path environments/cms-gpu/pixi.toml register-kernel
```

Then select **CMS GPU (EAF, Pixi)** in the notebook kernel picker; refresh the
Jupyter page if necessary. Registration must be repeated if the checkout moves.

## GPU configuration

Validated against the current session's NVIDIA A100 80GB GPU with a roughly
10GB MIG allocation, driver **595.71.05**, driver-supported CUDA **13.2**,
and host glibc **2.34**. The manifest declares CUDA **13.0** and glibc **2.34**
as minimum requirements and installs official PyTorch **CUDA 13.0** wheels.
A newer driver can run this older CUDA runtime. CUDA 13 requires an R580-or-newer
driver; another EAF session must satisfy these requirements too.

Use an EAF GPU session. `nvidia-smi` shows the driver's maximum CUDA support,
while `torch.version.cuda` shows the runtime used by PyTorch. GPU memory limits
follow the MIG allocation, not the full card's advertised memory.

PyTorch supplies its CUDA runtime libraries through wheel dependencies. CuPy
uses the CUDA 13 wheel family. Do not overlay the old `cudatoolkit=11.1`,
`pytorch-cuda=12.1`, or CPU-only PyTorch packages. A full CUDA compiler toolkit
is not explicitly requested; custom CUDA extension builds may need one.

## What was migrated

- ML: PyTorch, torchvision, Lightning, Accelerate, Transformers, Datasets,
  Hugging Face Hub, W&B, and the project's vector-quantize-pytorch dependency.
- Scientific Python: NumPy, SciPy, pandas, scikit-learn, Numba, h5py, Arrow,
  SymPy, NetworkX, Matplotlib, Seaborn, Plotly, Altair, and Streamlit.
- HEP: ROOT/PyROOT, Pythia 8, LHAPDF, XRootD/fsspec-xrootd, Awkward, uproot,
  vector, hist/boost-histogram, mplhep, iminuit, lmfit, uncertainties, and HEPData.
- Interactive/development: JupyterLab, Notebook, ipykernel, widgets, ipyparallel,
  C/C++/Fortran compilers, CMake, Ninja, Graphviz, pytest, and pre-commit.
- GPU additions: CuPy and the maintained `nvidia-ml-py` package (replaces
  `nvidia-ml-py3`).

This is a migration of user-facing capabilities, not a copy of every old
transitive library. Pixi resolves current runtime libraries instead of preserving
old compiler/sysroot/BLAS pins. Conda/Mamba themselves, Python 3.8 backports,
legacy Jupyter themes, and the old Qt desktop stack are not explicitly carried
over. Torchaudio is omitted: PyPI's latest release is 2.11.0, whereas this
workspace targets PyTorch 2.14; it is not needed by this HEP project.
LHAPDF PDF sets and external analysis data are not included in the handoff and
must be supplied separately. Old code may require changes for NumPy 2,
pandas 3, newer ROOT, and Transformers 5 APIs.

Python 3.13 is used for broad compiled HEP package compatibility. NumPy has a
2.5.3 minimum and PyTorch/torchvision are constrained to the matching 2.14/0.29
release families. Shared dependencies need two explicit compatibility bounds:
`fsspec <=2026.6.0` for Datasets 5.0.1 and `click >=8.4.2` for Hugging Face Hub.
Streamlit 1.64 also excludes Arrow 25.0.0, so Arrow resolves to 24.0.0.
Other direct package versions are recorded exactly in `pixi.lock`.

## Reproduce and update

```bash
pixi install --locked --manifest-path environments/cms-gpu/pixi.toml
pixi update --manifest-path environments/cms-gpu/pixi.toml
pixi run --manifest-path environments/cms-gpu/pixi.toml check
```

Review lockfile changes after updating. For a later PyTorch release, update
both the torch and torchvision bounds together and check driver compatibility.
Use Pixi to manage packages rather than running ad hoc `pip install` commands
inside the environment. The initial installation used
`PIXI_CACHE_DIR=/tmp/heptokens-pixi-cache` to avoid filling the 30GB home quota
with a second cached copy; the installed environment itself is persistent in
this directory's `.pixi/envs/default`. That temporary download cache is not
needed to run the environment.

References: [Pixi's PyTorch guide](https://pixi.prefix.dev/latest/python/pytorch/),
[PyTorch releases](https://github.com/pytorch/pytorch/releases), and
[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Installed versions and validation (2026-09-21)

| Package | Installed version |
| --- | --- |
| Python | 3.13.15 |
| PyTorch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| pandas / scikit-learn | 3.0.6 / 1.9.1 |
| CuPy | 14.2.0 |
| ROOT | 6.40.04 |
| Pythia / LHAPDF / XRootD | 8.312 / 6.5.6 / 6.1.1 |
| Awkward / uproot | 2.13.0 / 5.7.6 |
| Lightning / Transformers | 2.6.6 / 5.17.0 |

`pixi run check` passed imports, PyTorch GPU matrix multiplication and backward,
CuPy GPU matrix multiplication and elementwise compilation, ROOT C++ JIT,
and a ROOT-file write/read round trip. PyTorch reported runtime 13.0; CuPy
reported runtime 13.2 on this EAF host. `python -m pip check` reported no broken
requirements. The Jupyter kernel was registered as **CMS GPU (EAF, Pixi)**.
The persistent installation occupies approximately 10GB.

The existing project suite completed with **112 passed, 3 failed, and 1 setup
error**. The remaining issues are in the VQ model paths:

- `LitVqVae` passes `commitment` to `ResidualVQ`; the installed
  vector-quantize-pytorch API expects `commitment_weight`.
- `LitVqVae` passes `data_sample` to the project's `ParticleEncoder`, whose
  constructor does not accept that argument.

These affect the VQ training/export integration, tokenizer comparison, and two
contrastive VQ tests. Application source was not changed as part of this
environment migration. The `test` task limits BLAS/OpenMP threads to two because
small synthetic tests otherwise oversubscribe CPUs on this EAF node; ordinary
training commands do not inherit that limit.
