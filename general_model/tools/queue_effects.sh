#!/usr/bin/env bash
cd /home/zxc29051617/VCC
PY="scDFM/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for n in mean_x1.5 gw_x1.5; do
  step $PY pred  --heldout H1 --method effectfile --effect-file data/stage3/effects/H1_$n.npz
  step $PY score --heldout H1 --method effectfile_H1_$n
done
echo "=== $(date +%T) EFFECTS QUEUE DONE"
