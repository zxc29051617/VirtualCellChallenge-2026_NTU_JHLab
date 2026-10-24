#!/usr/bin/env bash
# Build + validate V7a / V7b (cells: VCC_gm_vip; gene totals: V6 totals scaled by 0.5 / 0.25). No upload.
set -euo pipefail
cd /home/zxc29051617/VCC/vccsub
for tag in ${TAGS:-s050 s025}; do
  name=gm_vip_v7_$tag
  echo "$(date '+%F %T') build $name"
  ../scDFM/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects ../data/stage3/effects/VCC_gm_vip.npz \
      --effects-bulk ../data/stage3/effects/VCC_gm_vip_pc10_$tag.npz --render dual --out out/$name.h5ad 2>&1 | grep -E "wrote|Error|Traceback"
  ~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/$name.h5ad out/$name.vcc.part --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv 2>&1 | tail -1
  mv out/$name.vcc.part out/$name.vcc; rm -f out/$name.h5ad
  (cd /tmp && ~/.local/share/uv/tools/vcc-cli/bin/python -I -c "
from vcc.vccfile import validate_vcc, read_vcc_meta
p = '/home/zxc29051617/VCC/vccsub/out/$name.vcc'; validate_vcc(p); print('validated $name', read_vcc_meta(p))")
done
echo "$(date '+%F %T') V7 files ready (not uploaded)"
