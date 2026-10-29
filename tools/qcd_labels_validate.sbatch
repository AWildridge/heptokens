#!/usr/bin/env bash
# Submit only after all expected producer array tasks have completed successfully.
#SBATCH --job-name=qcd-label-validation
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=qcd-label-validation-%j.log
set -euo pipefail
: "${QCD_REPO:?Set QCD_REPO to the heptokens repository}"
: "${QCD_OUTPUT:?Set QCD_OUTPUT to the completed versioned scratch directory}"
: "${LABEL_PYTHON:?Set LABEL_PYTHON to the CPU analysis Python}"
: "${LABEL_DEPS:?Set LABEL_DEPS to the pinned dependency overlay}"
srun --ntasks=1 --cpus-per-task=1 bash "$QCD_REPO/tools/run_qcd_labels.sh" validate \
  --output "$QCD_OUTPUT" --plots
