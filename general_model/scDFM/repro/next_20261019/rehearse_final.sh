#!/usr/bin/env bash
# Rehearsal of the D/E/F pipeline with the full 2026-10-20 evening config on A/B/C; compares with G3c (cache-built).
set -euo pipefail
cd /home/zxc29051617/VCC
RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5 CELLAGREE=1 TWPOW=0.5 ENTRYAGREE=1 tools/final_pipeline.sh data/vcc2026 rehearsal_final A B C
scDFM/.venv/bin/python -I - <<'PY'
import numpy as np
E = "/home/zxc29051617/VCC/data/stage3/effects"
a = np.load(f"{E}/rehearsal_final_cells_agree.npz")["lfc"]; b = np.load(f"{E}/VCC_gm_vip_c12.npz")["lfc"]
print("cells vs c12: max abs diff", float(np.abs(a - b).max()))
for c in "ABC":
    x = np.load(f"{E}/rehearsal_final_bulk_g1_{c}.npz")["lfc"]; y = np.load(f"{E}/VCC_gm_vip_g3a_{c}.npz")["lfc"]
    print(c, "totals vs g3a: max abs diff", float(np.abs(x - y).max()), "rel", float(np.linalg.norm(x - y) / np.linalg.norm(y)))
PY
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import read_vcc_meta
for n in ('rehearsal_final', 'gm_vip_g3c'): print(n, read_vcc_meta('/home/zxc29051617/VCC/vccsub/out/%s.vcc' % n))")
rm -f /home/zxc29051617/VCC/vccsub/out/rehearsal_final.vcc
echo "rehearsal done"
