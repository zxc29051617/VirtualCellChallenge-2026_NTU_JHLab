#!/usr/bin/env bash
# One heavy job at a time on the 15 GB machine.
cd /home/zxc29051617/VCC
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
B=https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025
for f in validation/adata_Validation.h5ad test/adata_Test.h5ad train/adata_Training.h5ad; do
  [ -s data/external/h1_2025/$(basename $f) ] || step python3 -I tools/pget.py $B/$f data/external/h1_2025/$(basename $f) --conns 12 --chunk-mb 16
done
( cd tools && step /home/zxc29051617/VCC/pertrelay/.venv/bin/python -I extract_replogle.py K562_gwps_raw_singlecell_01.h5ad K562 replogle_K562_gwps )
( cd tools && step /home/zxc29051617/VCC/pertrelay/.venv/bin/python -I extract_xatlas.py HCT116 )
( cd tools && step /home/zxc29051617/VCC/pertrelay/.venv/bin/python -I extract_xatlas.py HEK293T )
( cd pertrelay && step .venv/bin/python -m vcc.stats K562 HCT116 HEK293T )
echo "=== $(date +%T) MASTER DONE"
