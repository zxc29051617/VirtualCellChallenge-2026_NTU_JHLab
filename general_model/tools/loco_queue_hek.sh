#!/usr/bin/env bash
cd /home/zxc29051617/VCC
PY="scDFM/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
step $PY prep  --heldout HEK293T --max-cells 120
step $PY pred  --heldout HEK293T --method control_resample
step $PY score --heldout HEK293T --method control_resample
step $PY pred  --heldout HEK293T --method transfer --sources HCT116
step $PY score --heldout HEK293T --method transfer --sources HCT116
echo "=== $(date +%T) QUEUE DONE"
