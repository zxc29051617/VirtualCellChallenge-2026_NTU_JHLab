#!/usr/bin/env bash
cd /home/zxc29051617/VCC
while kill -0 163751 2>/dev/null; do sleep 30; done
PY="pertrelay/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for s in 1.5 2 3; do
  step $PY pred  --heldout H1 --method transfer --sources K562 HCT116 HEK293T --scale $s
  step $PY score --heldout H1 --method transfer --sources K562 HCT116 HEK293T --scale $s
done
echo "=== $(date +%T) SCALE QUEUE DONE"
