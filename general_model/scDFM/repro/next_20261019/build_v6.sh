#!/usr/bin/env bash
# Build + validate gm_v1_vip V6 (cells: VCC_gm_vip trans x3; gene totals: VCC_gm_vip_pc10, i.e. shared trans components removed).
set -euo pipefail
cd /home/zxc29051617/VCC/vccsub
echo "$(date '+%F %T') build start"
../scDFM/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects ../data/stage3/effects/VCC_gm_vip.npz \
    --effects-bulk ../data/stage3/effects/VCC_gm_vip_pc10.npz --render dual --out out/gm_vip_v6.h5ad 2>&1 | grep -E "targets with|wrote|Error|Traceback"
echo "$(date '+%F %T') pack start"
~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/gm_vip_v6.h5ad out/gm_vip_v6.vcc.part --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv 2>&1 | tail -1
mv out/gm_vip_v6.vcc.part out/gm_vip_v6.vcc; rm -f out/gm_vip_v6.h5ad
(cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta
p = '/home/zxc29051617/VCC/vccsub/out/gm_vip_v6.vcc'; validate_vcc(p); print('validated', read_vcc_meta(p))")
echo "$(date '+%F %T') ready (not uploaded)"
