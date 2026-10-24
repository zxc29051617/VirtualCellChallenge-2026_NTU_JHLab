#!/usr/bin/env bash
# After the fp32 token run: the effect-conditioned run with bf16 autocast (grad cosine 0.9994 vs fp32).
cd /home/zxc29051617/VCC/scDFM
while kill -0 97947 2>/dev/null; do sleep 30; done
export PYTHONPATH=.
F=/home/zxc29051617/VCC/data/stage3/model/h1_pert_holdout
echo "=== $(date +%T) effect run (bf16)"
.venv/bin/python -m vcc.train --fold $F --out runs/h1ho_effect --cond effect --amp bf16 --steps 20000 --batch 16 --line-weights H1=10 --eval-every 1000 --eval-targets 60 --eval-cells 64
echo "=== $(date +%T) ALL DONE"
