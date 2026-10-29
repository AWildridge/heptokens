#!/usr/bin/env bash
# Supply the Purdue account/CPU partition at submission; no NERSC resources are assumed.
#SBATCH --job-name=qcd-labels
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-4545%16
#SBATCH --output=qcd-labels-%A_%a.log
set -euo pipefail
: "${QCD_REPO:?Set QCD_REPO to the checked-out heptokens repository}"
: "${QCD_CAMPAIGN:?Set QCD_CAMPAIGN to the original raw campaign}"
: "${QCD_OUTPUT:?Set QCD_OUTPUT to a new versioned scratch directory}"
: "${LABEL_PYTHON:?Set LABEL_PYTHON to the CPU analysis Python}"
: "${LABEL_DEPS:?Set LABEL_DEPS to the pinned dependency overlay}"
options=()
if [[ -n "${QCD_NORMALIZATION:-}" ]]; then
  options+=(--normalization "$QCD_NORMALIZATION")
fi
if [[ -n "${QCD_CONFIG:-}" ]]; then
  options+=(--config "$QCD_CONFIG")
fi
srun --ntasks=1 --cpus-per-task=1 bash "$QCD_REPO/tools/run_qcd_labels.sh" produce \
  --campaign "$QCD_CAMPAIGN" --output "$QCD_OUTPUT" \
  --source-task "$SLURM_ARRAY_TASK_ID" "${options[@]}"
