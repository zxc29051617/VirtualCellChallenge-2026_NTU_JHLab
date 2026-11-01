#!/usr/bin/env bash
# H1 (2025) as TRAINING data, with 100 of its perturbations held out as the test.
cd /home/zxc29051617/VCC/pertrelay
while kill -0 87175 2>/dev/null; do sleep 30; done
step() { echo "=== $(date +%T) $*"; "$@" || echo "!!! FAILED: $*"; }
F=/home/zxc29051617/VCC/data/stage3/model/h1_pert_holdout
step .venv/bin/python -m vcc.prepare --train K562 HCT116 HEK293T H1 --heldout H1 --holdout-targets /home/zxc29051617/VCC/data/stage3/h1_test_targets.txt --out $F
export PYTHONPATH=.
step .venv/bin/python -m vcc.train --fold $F --out runs/h1ho_token --cond token --steps 20000 --batch 16 --line-weights H1=10 --eval-every 1000 --eval-targets 60 --eval-cells 64
step .venv/bin/python -m vcc.train --fold $F --out runs/h1ho_effect --cond effect --steps 20000 --batch 16 --line-weights H1=10 --eval-every 1000 --eval-targets 60 --eval-cells 64
echo "=== $(date +%T) ALL DONE"
