#!/usr/bin/env bash
# Feng 2025 hiPSC genome-scale CRISPRi, fitness-gene set (adds 31 VCC targets to the non-fitness set already here)
cd /home/zxc29051617/VCC/data/external
P="python3 -I /home/zxc29051617/VCC/tools/pget.py"
for id_name in 51051803:GenomeWideScreen_FitnessGenes_Cell-Metadata.tsv.gz 51051812:GenomeWideScreen_FitnessGenes_Guide-UMI-Counts.csv.gz 51059483:GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz; do
  [ -f feng2025/${id_name#*:} ] || $P https://ndownloader.figshare.com/files/${id_name%%:*} feng2025/${id_name#*:} --conns 16 --chunk-mb 16
done
echo FENG_FITNESS_DONE
