#!/usr/bin/env bash
# Build + validate G2C for A/B/C: cells VCC_gm_vip_c12 (trans x4.5 x agreement), totals VCC_gm_vip_g1_{A,B,C}, added counts in the top 2 % of cells. No upload.
set -euo pipefail
cd /home/zxc29051617/VCC/vccsub; E=../data/stage3/effects
echo "$(date '+%F %T') build gm_vip_g2c"
../pertrelay/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects $E/VCC_gm_vip_c12.npz \
    --effects-bulk A=$E/VCC_gm_vip_g1_A.npz B=$E/VCC_gm_vip_g1_B.npz C=$E/VCC_gm_vip_g1_C.npz --render dual --top-frac 0.02 --out out/gm_vip_g2c.h5ad 2>&1 | grep -E "wrote|Error|Traceback"
~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/gm_vip_g2c.h5ad out/gm_vip_g2c.vcc.part --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv 2>&1 | tail -1
mv out/gm_vip_g2c.vcc.part out/gm_vip_g2c.vcc; rm -f out/gm_vip_g2c.h5ad
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta
p = '/home/zxc29051617/VCC/vccsub/out/gm_vip_g2c.vcc'; validate_vcc(p); print('validated gm_vip_g2c', read_vcc_meta(p))")
echo "$(date '+%F %T') G2C file ready (not uploaded)"
