#!/usr/bin/env bash
cd /home/zxc29051617/VCC/data/external
P="python3 -I /home/zxc29051617/VCC/tools/pget.py"
G=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl
# Nadig 2025: Jurkat is a T-cell leukaemia line (context A is T-cell-like); NCBI throttles, so few connections
$P $G/GSE264667_jurkat_raw_singlecell_01.h5ad nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad --conns 4 --chunk-mb 8
$P $G/GSE264667_hepg2_raw_singlecell_01.h5ad  nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad  --conns 4 --chunk-mb 8
# Feng 2025 hiPSC genome-scale CRISPRi, the non-fitness gene set (non-essential targets, like VCC's)
for id_name in 51051806:GenomeWideScreen_NonFitnessGenes_Cell-Metadata.tsv.gz 51051818:GenomeWideScreen_NonFitnessGenes_Guide-UMI-Counts.csv.gz 51059558:GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz; do
  $P https://ndownloader.figshare.com/files/${id_name%%:*} feng2025/${id_name#*:} --conns 16 --chunk-mb 16
done
echo SQUEEZE_DL_DONE
