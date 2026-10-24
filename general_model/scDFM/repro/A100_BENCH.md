# A100 80GB (Brev, Crusoe `a100-80gb.1x`, US$1.98/hr) benchmark — 2026-10-09

Instance: 12 vCPU, 117 GiB RAM, 124 GB disk (103 GB free), Python 3.10 system, torch 2.7.1+cu126 in venv.
Model: run.sh architecture at the width the checkpoints use (d_model 512, 60.4M params), 1,000 genes per cell.

## Training (`repro/bench.py`), seconds per step / peak GPU memory

| batch | original attention | fused attention (SCDFM_SDPA=1) |
| --- | --- | --- |
| 8 | 0.284 s / 12.0 GiB | 0.238 s / 6.4 GiB |
| 16 | 0.558 s / 22.8 GiB | 0.462 s / 11.4 GiB |
| 32 | 1.088 s / 44.4 GiB | 0.902 s / 21.2 GiB |
| 48 (run.sh) | 1.637 s / 66.0 GiB | **1.355 s / 31.1 GiB** |

Fused path equals the original on the same weights/inputs/RNG: loss identical, gradient rel. diff 1.8e-5 (fp32, TF32 off).
Original run.sh recipe (batch 48, 200k steps, fused): 200,000 x 1.355 s = 75 h = ~US$149. The released checkpoints used 80k steps: 30 h = ~US$60.
vs the local RTX 4060 Ti: ~3x faster per cell (0.028 vs 0.09 s/cell at batch 8).

## Generation (1,000 genes, 20-point rk4 = 76 forwards, fused attention)

| cells | rk4-20 | euler-10 | one step |
| --- | --- | --- | --- |
| 128 | 65 s | 7.7 s | 0.9 s |
| 400 | 204 s | 24 s | 2.7 s |

One VCC submission is 900 blocks of 400 cells (300 perturbations x 3 contexts):
rk4-20 = 51 h (~US$101), euler-10 = 6 h (~US$12), one step = 41 min (~US$1.4) -- for 1,000 genes.
VCC scores 18,533 genes; the model only handles a 1,000-gene subset per call, so covering the full
gene axis multiplies these numbers and is a design problem for stage 3/4, not a settings change.

## Operational note
A Brev login via a pasted 15-minute JWT expires with no refresh token, after which `brev stop` is
unavailable; direct SSH (`ssh -F ~/.brev/ssh_config <name>`) keeps working. Use an API key
(`brev login --api-key`) for anything long-lived.
