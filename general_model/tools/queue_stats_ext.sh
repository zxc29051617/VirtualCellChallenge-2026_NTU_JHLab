#!/usr/bin/env bash
# Pseudobulk statistics for the extra lines, one at a time (15 GB RAM).
cd /home/zxc29051617/VCC/scDFM
export PYTHONPATH=.
D=/home/zxc29051617/VCC/data/external
run() { echo "=== $(date +%T) $*"; .venv/bin/python -m vcc.stats_ext "$@" || echo "FAILED $*"; }
run replogle K562ess $D/replogle2022/K562_essential_raw_singlecell_01.h5ad
run replogle HepG2   $D/nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad
run replogle Jurkat  $D/nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad
run feng FengNF $D/feng2025/GenomeWideScreen_NonFitnessGenes
until grep -q FENG_FITNESS_DONE $D/feng_fitness_download.log; do sleep 30; done
run feng FengF  $D/feng2025/GenomeWideScreen_FitnessGenes
echo "=== $(date +%T) ALL DONE"
