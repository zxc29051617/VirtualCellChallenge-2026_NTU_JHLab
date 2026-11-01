#!/usr/bin/env bash
# Build + validate G3 for A/B/C: cells VCC_gm_vip_a6 (trans x6), totals VCC_gm_vip_g4_{A,B,C} (gamma 4), dual render with added counts in the top 2 % of cells. No upload.
set -euo pipefail
cd /home/zxc29051617/VCC/vccsub; E=../data/stage3/effects
echo "$(date '+%F %T') build gm_vip_g3"
../pertrelay/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects $E/VCC_gm_vip_a6.npz \
    --effects-bulk A=$E/VCC_gm_vip_g4_A.npz B=$E/VCC_gm_vip_g4_B.npz C=$E/VCC_gm_vip_g4_C.npz --render dual --top-frac 0.02 --out out/gm_vip_g3.h5ad 2>&1 | grep -E "wrote|Error|Traceback"
~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/gm_vip_g3.h5ad out/gm_vip_g3.vcc.part --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv 2>&1 | tail -1
mv out/gm_vip_g3.vcc.part out/gm_vip_g3.vcc; rm -f out/gm_vip_g3.h5ad
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta
p = '/home/zxc29051617/VCC/vccsub/out/gm_vip_g3.vcc'; validate_vcc(p); print('validated gm_vip_g3', read_vcc_meta(p))")
echo "$(date '+%F %T') G3 file ready (not uploaded)"
