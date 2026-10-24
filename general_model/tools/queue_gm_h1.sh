#!/usr/bin/env bash
# Official-scorer check of the general model on held-out H1 (fit without H1).
cd /home/zxc29051617/VCC
PY="scDFM/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for n in "$@"; do
  step $PY pred  --heldout H1 --method effectfile --effect-file data/stage3/effects/H1_$n.npz
  step $PY score --heldout H1 --method effectfile_H1_$n
  grep -E "^(pds|de_|expr|avg)" data/stage3/loco/H1/score_effectfile_H1_$n/score.csv | cut -d, -f1,3
done
echo "=== $(date +%T) GM H1 QUEUE DONE"
