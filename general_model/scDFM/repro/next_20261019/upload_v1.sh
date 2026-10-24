#!/usr/bin/env bash
# Build, validate and upload gm_v1_vip with exact per-gene group totals (V1 rendering), approved by the user 2026-10-19.
# Pauses the local scoring queue (render_eval.py, resumable) while building to stay inside 15 GB RAM, resumes it after.
set -euo pipefail
HERE=/home/zxc29051617/VCC/scDFM/repro/next_20261019; V=/home/zxc29051617/VCC/vccsub
RUNNER=${1:?runner pid}
log() { echo "$(date '+%F %T') $*"; }
until grep -q "score 21/27" $HERE/run.log; do sleep 10; done
for c in $(pgrep -P "$RUNNER" || true); do kill "$c" || true; done
kill "$RUNNER" || true; sleep 3; rm -f $HERE/pred/*.h5ad
log "local scoring paused after run 21/27"
cd $V
log "build start"
../scDFM/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects ../data/stage3/effects/VCC_gm_vip.npz \
    --render exact --out out/gm_vip_exact.h5ad 2>&1 | grep -E "targets with|wrote|Error|Traceback"
log "pack start"
~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/gm_vip_exact.h5ad out/gm_vip_exact.vcc.part \
    --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv 2>&1 | tail -2
mv out/gm_vip_exact.vcc.part out/gm_vip_exact.vcc; rm -f out/gm_vip_exact.h5ad
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta
p = '$V/out/gm_vip_exact.vcc'; validate_vcc(p); print('validated', read_vcc_meta(p))")
sha256sum out/gm_vip_exact.vcc | cut -c1-16
log "resume local scoring"
cd $HERE && nohup env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 ../../.venv/bin/python -I render_eval.py run H1 KOLF200 HCT116 >> render_scoring.log 2>&1 &
log "upload start"
cd /home/zxc29051617/VCC && tools/submit_queue.sh "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "out/gm_vip_exact.vcc:gm_v1_vip_exact:general model v1 + VIPerturb (same effects as gm_v1_vip), cells emitted with exact per-gene group totals - rendering-only change"
log "upload script finished"
