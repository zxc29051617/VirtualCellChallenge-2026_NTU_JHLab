#!/usr/bin/env bash
# VIPerturb-seq (Satija lab 2026, Zenodo 18460279, CC BY 4.0): genome-wide K562 CRISPRi in 10x Flex, all 300 VCC targets.
cd /home/zxc29051617/VCC/data/external/viperturb
P="python3 -I /home/zxc29051617/VCC/tools/pget.py"
for f in genome_wide_manifest.txt genome_wide_filtered.rds; do
  [ -f $f ] || $P https://zenodo.org/api/records/18460279/files/$f/content $f --conns 8 --chunk-mb 16
done
echo VIPERTURB_DL_DONE
