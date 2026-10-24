#!/usr/bin/env bash
# KOLF2.1J pan-genome h5ad (figshare+ 27261219): only the raw-count layer, bytes [94,919,293,556, end) = layers/counts
# data chunks (with interleaved B-tree nodes), indices chunks and indptr. ~94.5 GB at ~5 MB/s from here (~5 h).
cd /home/zxc29051617/VCC/data/external/kolf
for try in 1 2 3 4 5; do
  [ -f counts_region.bin ] && break
  python3 -I /home/zxc29051617/VCC/tools/pget.py https://ndownloader.figshare.com/files/64650261 counts_region.bin \
    --conns 32 --chunk-mb 8 --range 94919293556:189393177972
done
echo KOLF_DL_DONE
