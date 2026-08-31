#!/usr/bin/env bash
# Campaign 7 — downstream QCD-vs-top classification on AOJ MC (record 18610),
# comparing frozen real-AOJ tokenizer exports against the heptokens VQ baseline
# and raw-particle / untrained references. Design + protocol:
# docs/research (Campaign 7) and benchmarks/train_probes_mc.py docstring.
#
# Stages:
#   prepare : download + window + subsample + assemble aojmc_structured.h5
#   probes  : exports (idempotent) + 9 arms x 3 seeds, strictly sequential
#   all     : prepare -> probes
#
# No `set -e`: failures are recorded and the sweep continues. One GPU job at a
# time on the shared MIG slice.

cd "$(dirname "$0")/.." || exit 1
PY=.pixi/envs/default/bin/python
STAGE="${1:-all}"
FAILED=""

if [ "$STAGE" = "prepare" ] || [ "$STAGE" = "all" ]; then
  echo "===== Campaign 7: prepare AOJ MC ====="
  if [ -f results/benchmarks/data/aojmc_structured.h5 ]; then
    echo "[skip] aojmc_structured.h5 exists"
  elif $PY benchmarks/prepare_aoj_mc.py; then
    echo "[ok  ] prepare_aoj_mc"
  else
    echo "[FAIL] prepare_aoj_mc"
    FAILED="$FAILED prepare"
  fi
fi

if [ "$STAGE" = "probes" ] || [ "$STAGE" = "all" ]; then
  echo "===== Campaign 7: exports + probes (9 arms x 3 seeds) ====="
  if [ ! -f results/benchmarks/data/aojmc_structured.h5 ]; then
    echo "[FAIL] aojmc_structured.h5 missing — run the prepare stage first"
    FAILED="$FAILED probes"
  # 16 h hard timeout: a silent hang must fail loudly, not eat days. The
  # harness is idempotent, so a rerun resumes from the completed probes.
  elif timeout 57600 $PY benchmarks/train_probes_mc.py; then
    echo "[ok  ] train_probes_mc"
  else
    echo "[FAIL] train_probes_mc"
    FAILED="$FAILED probes"
  fi
fi

echo "===================================================="
if [ -n "$FAILED" ]; then
  echo "FAILED stages:$FAILED"
  exit 1
else
  echo "campaign-7 stage(s) '$STAGE' ok"
fi
