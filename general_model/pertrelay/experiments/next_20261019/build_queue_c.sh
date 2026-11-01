#!/usr/bin/env bash
# Build G3c, G2c, G3v one at a time; each build needs ~37 GB at its peak (17 GB h5ad + ~16 GB temporary copy while packing
# + 3.3 GB .vcc), so each starts only with >= 40 GB free.
cd /home/zxc29051617/VCC/pertrelay/experiments/next_20261019
for b in g3c g2c g3v; do
  while [ "$(df --output=avail -B1G /home | tail -n 1 | tr -d ' ')" -lt 40 ]; do echo "$(date '+%T') waiting for 40 GB free before $b"; sleep 60; done
  ./build_$b.sh > build_$b.log 2>&1
  grep -q "file ready" build_$b.log && echo "$(date '+%T') $b ok" || echo "$(date '+%T') $b FAILED"
done
