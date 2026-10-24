#!/usr/bin/env bash
cd /home/zxc29051617/VCC
PY="scDFM/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
step $PY prep  --heldout H1 --roles h1_target vcc_target --max-cells 80 --targets-file data/stage3/loco/h1_eval_targets.txt
step $PY pred  --heldout H1 --method control_resample
step $PY score --heldout H1 --method control_resample
step $PY pred  --heldout H1 --method transfer --sources K562 HCT116 HEK293T
step $PY score --heldout H1 --method transfer --sources K562 HCT116 HEK293T
echo "=== $(date +%T) H1 QUEUE DONE"
