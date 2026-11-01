#!/usr/bin/env bash
# Build + validate G4V for A/B/C: cells VCC_gm_vip (V1, trans x3), totals VCC_gm_vip_g4c_{A,B,C} (G3c totals with cis x3), added counts in the top 2 % of cells. No upload.
set -euo pipefail
cd /home/zxc29051617/VCC/vccsub; E=../data/stage3/effects
echo "$(date '+%F %T') build gm_vip_g4v"
../pertrelay/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects $E/VCC_gm_vip.npz \
    --effects-bulk A=$E/VCC_gm_vip_g4c_A.npz B=$E/VCC_gm_vip_g4c_B.npz C=$E/VCC_gm_vip_g4c_C.npz --render dual --top-frac 0.02 --out out/gm_vip_g4v.h5ad 2>&1 | grep -E "wrote|Error|Traceback"
~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/gm_vip_g4v.h5ad out/gm_vip_g4v.vcc.part --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv 2>&1 | tail -1
mv out/gm_vip_g4v.vcc.part out/gm_vip_g4v.vcc; rm -f out/gm_vip_g4v.h5ad
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta
p = '/home/zxc29051617/VCC/vccsub/out/gm_vip_g4v.vcc'; validate_vcc(p); print('validated gm_vip_g4v', read_vcc_meta(p))")
echo "$(date '+%F %T') G4V file ready (not uploaded)"
