#!/usr/bin/env bash
cd /home/zxc29051617/VCC
PY="pertrelay/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for n in "$@"; do
  step $PY pred  --heldout H1 --method effectfile --effect-file data/stage3/effects/H1_$n.npz
  step $PY score --heldout H1 --method effectfile_H1_$n
  rm -f data/stage3/loco/H1/pred_effectfile_H1_$n.h5ad
done
echo "=== $(date +%T) DONE"
