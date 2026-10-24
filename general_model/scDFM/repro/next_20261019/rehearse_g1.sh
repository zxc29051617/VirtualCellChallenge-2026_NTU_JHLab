#!/usr/bin/env bash
# Rehearsal of the D/E/F pipeline with RENDER=g1 on the A/B/C controls (2026-10-20): checks that final_pipeline.sh runs end
# to end (two exports incl. n0=200, per-context totals with gamma 3 and budget = claim, cells, pack, validate) and that the
# result matches the G2 file scheduled for 10-21 (cache-built; RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5). No upload.
set -euo pipefail
cd /home/zxc29051617/VCC
RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5 tools/final_pipeline.sh data/vcc2026 rehearsal_g1 A B C
scDFM/.venv/bin/python -I - <<'PY'
import numpy as np
E = "/home/zxc29051617/VCC/data/stage3/effects"
a = np.load(f"{E}/rehearsal_g1_d200.npz")["lfc"]; b = np.load(f"{E}/VCC_gm_vip_d200.npz")["lfc"]
print("export n0=200 vs cache d200: max abs diff", float(np.abs(a - b).max()))
for c in "ABC":
    x = np.load(f"{E}/rehearsal_g1_bulk_g1_{c}.npz")["lfc"]; y = np.load(f"{E}/VCC_gm_vip_g1_{c}.npz")["lfc"]
    print(c, "totals vs G1: max abs diff", float(np.abs(x - y).max()), "rel", float(np.linalg.norm(x - y) / np.linalg.norm(y)))
PY
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import read_vcc_meta
for n in ('rehearsal_g1', 'gm_vip_g2'): print(n, read_vcc_meta('/home/zxc29051617/VCC/vccsub/out/%s.vcc' % n))")
rm -f /home/zxc29051617/VCC/vccsub/out/rehearsal_g1.vcc
echo "rehearsal done"
