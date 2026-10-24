#!/usr/bin/env bash
# Resume the Feng fitness-gene download, then build both Feng statistics (gene symbols from 'ENSG:SYMBOL:type').
cd /home/zxc29051617/VCC/data/external
f=feng2025/GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz
for try in 1 2 3; do [ -f $f ] && break; python3 -I /home/zxc29051617/VCC/tools/pget.py https://ndownloader.figshare.com/files/51059483 $f --conns 8 --chunk-mb 16; done
cd /home/zxc29051617/VCC/scDFM; export PYTHONPATH=.
for x in "FengNF GenomeWideScreen_NonFitnessGenes" "FengF GenomeWideScreen_FitnessGenes"; do
  set -- $x; echo "=== $(date +%T) $1"
  .venv/bin/python -m vcc.stats_ext feng $1 /home/zxc29051617/VCC/data/external/feng2025/$2 || echo "FAILED $1"
done
echo "=== $(date +%T) FENG DONE"
