#!/usr/bin/env bash
# Build + validate V8 for A/B/C (cells: VCC_gm_vip; totals: per-context VCC_gm_vip_v8_{A,B,C}). No upload.
set -euo pipefail
cd /home/zxc29051617/VCC/vccsub; E=../data/stage3/effects
echo "$(date '+%F %T') build gm_vip_v8"
../scDFM/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects $E/VCC_gm_vip.npz \
    --effects-bulk A=$E/VCC_gm_vip_v8_A.npz B=$E/VCC_gm_vip_v8_B.npz C=$E/VCC_gm_vip_v8_C.npz --render dual --out out/gm_vip_v8.h5ad 2>&1 | grep -E "wrote|Error|Traceback"
~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/gm_vip_v8.h5ad out/gm_vip_v8.vcc.part --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv 2>&1 | tail -1
mv out/gm_vip_v8.vcc.part out/gm_vip_v8.vcc; rm -f out/gm_vip_v8.h5ad
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta
p = '/home/zxc29051617/VCC/vccsub/out/gm_vip_v8.vcc'; validate_vcc(p); print('validated gm_vip_v8', read_vcc_meta(p))")
echo "$(date '+%F %T') V8 file ready (not uploaded)"
