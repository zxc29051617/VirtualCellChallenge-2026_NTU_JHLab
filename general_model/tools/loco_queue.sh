#!/usr/bin/env bash
# Stage-3 LOCO baselines, strictly one heavy job at a time (15 GB RAM).
cd /home/zxc29051617/VCC
PY="scDFM/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for L in HCT116 HEK293T; do
  S=$([ $L = HCT116 ] && echo HEK293T || echo HCT116)
  step $PY prep  --heldout $L
  [ -f data/stage3/loco/$L/pred_control_resample.h5ad ] || step $PY pred --heldout $L --method control_resample
  step $PY score --heldout $L --method control_resample
  [ -f data/stage3/loco/$L/pred_transfer_$S.h5ad ] || step $PY pred --heldout $L --method transfer --sources $S
  step $PY score --heldout $L --method transfer --sources $S
done
echo "=== $(date +%T) QUEUE DONE"
