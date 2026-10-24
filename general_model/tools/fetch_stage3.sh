#!/usr/bin/env bash
# Fetch stage-3 raw single-cell CRISPRi files into data/external (resumable).
cd /home/zxc29051617/VCC/data/external
while IFS=$'\t' read -r f u; do
  [ -s "$f" ] && { echo "have $f"; continue; }
  case "$u" in *geo/download*) u="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/$(basename "$f")";; esac
  python3 -I /home/zxc29051617/VCC/tools/pget.py "$u" "$f" --conns 32 --chunk-mb 4
done < download_list.tsv
echo ALL_DONE
