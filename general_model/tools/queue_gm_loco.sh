#!/usr/bin/env bash
# Official-scorer check of general-model effect files on a held-out line:  queue_gm_loco.sh <heldout> <name> ...
# (effect file data/stage3/effects/<heldout>_<name>.npz; one heavy job at a time)
cd /home/zxc29051617/VCC
H=$1; shift
PY="pertrelay/.venv/bin/python -I tools/loco.py"
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
for n in "$@"; do
  step $PY pred  --heldout $H --method effectfile --effect-file data/stage3/effects/${H}_$n.npz ${LOCO_PRED_ARGS:-}
  step $PY score --heldout $H --method effectfile_${H}_$n
  echo "RESULT $H $n $(grep avg data/stage3/loco/$H/score_effectfile_${H}_$n/score.csv | cut -d, -f3) rawmse $(python3 -I -c "import csv,sys; r={x[\"statistic\"]:x for x in csv.DictReader(open(sys.argv[1]))}[\"mean\"]; print(round(float(r[\"expr_mse_unbiased_capped_norm\"]),3), \"rawpds\", round(float(r[\"pds_cosine\"]),3))" data/stage3/loco/$H/score_effectfile_${H}_$n/agg_results.csv)"
  rm -f data/stage3/loco/$H/pred_effectfile_${H}_$n.h5ad      # 1-2 GB each; the score directory is kept
done
echo "=== $(date +%T) QUEUE DONE $H"
