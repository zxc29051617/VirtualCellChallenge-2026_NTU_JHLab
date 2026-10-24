#!/usr/bin/env bash
# Fetch stage-3 raw single-cell CRISPRi files into data/external (resumable).
LIST=$(dirname "$(readlink -f "$0")")/download_list.tsv       # the list ships with the code (tools/)
mkdir -p /home/zxc29051617/VCC/data/external && cd /home/zxc29051617/VCC/data/external
while IFS=$'\t' read -r f u; do
  [ -s "$f" ] && { echo "have $f"; continue; }
  mkdir -p "$(dirname "$f")"
  case "$u" in *geo/download*) u="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/$(basename "$f")";; esac
  python3 -I /home/zxc29051617/VCC/tools/pget.py "$u" "$f" --conns 32 --chunk-mb 4
done < "$LIST"
echo ALL_DONE
