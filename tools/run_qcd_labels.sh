#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
label_python="${LABEL_PYTHON:-$repo_dir/environments/cms-gpu/.pixi/envs/default/bin/python}"
label_deps="${LABEL_DEPS:-/global/u2/a/anwildri/code/label-production-deps}"
export PYTHONPATH="$label_deps:$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qcd-labels-mpl-$UID}"
exec "$label_python" -m heptokens.labels.cli "$@"
