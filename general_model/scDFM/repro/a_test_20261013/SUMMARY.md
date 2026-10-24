# Option A test (2026-10-13): sparse trans (top-200 |s_p| genes at alpha 3, rest at 0.5) vs base (alpha 3 everywhere)

One fit per held-out set (all training contexts except the held-out cell type; incl. VIPK562), four variants from that
fit, generation seeds 0/1/2 with the fixed tools/loco.py (original-library up formula). Official cell-eval2 0.18.0
vcc2026 scorer, each set's own bundle. H1: 188 targets; KOLFbig: 279 targets, 400 predicted cells, 20k controls.
Pre-agreed criteria in evaluate.py; result in verdict.json (runs in runs.csv).

| set | variant | raw mse_norm (seeds 0/1/2) | avg (seeds 0/1/2) |
| --- | --- | --- | --- |
| H1 | base plain | 1.451 / 1.451 / 1.451 | 0.349 / 0.347 / 0.345 |
| H1 | base shuffled | 1.857 / 1.847 / 1.851 | 0.095 / 0.102 / 0.105 |
| H1 | A plain | 0.985 / 0.983 / 0.982 | 0.204 / 0.212 / 0.205 |
| H1 | A shuffled | 1.125 / 1.124 / 1.127 | 0.040 / 0.049 / 0.027 |
| KOLFbig | base plain | 2.224 / 2.219 / 2.223 | 0.188 / 0.183 / 0.183 |
| KOLFbig | base shuffled | 2.293 / 2.283 / 2.302 | 0.136 / 0.136 / 0.131 |
| KOLFbig | A plain | 1.187 / 1.200 / 1.177 | 0.125 / 0.110 / 0.129 |
| KOLFbig | A shuffled | 1.222 / 1.219 / 1.219 | 0.086 / 0.079 / 0.085 |

Per-target pds differences (seed-averaged, bootstrap over targets):
- A − base: H1 −0.038 [−0.058, −0.018]; KOLFbig −0.004 [−0.030, +0.022]
- A − A shuffled: H1 +0.308 [+0.269, +0.348]; KOLFbig +0.067 [+0.036, +0.098]
- base − base shuffled: H1 +0.348 [+0.304, +0.392]; KOLFbig +0.078 [+0.034, +0.121]

Verdict: criterion 1 (raw mse improves) PASS on both; criterion 3 (pairing beats shuffle) PASS on both;
criterion 2 (pds not clearly lower) FAIL on both (H1 clearly lower; KOLFbig CI lower bound −0.030 < −0.02).
**Overall FAIL — not submitted.** The local official-scale average also drops (H1 0.347 → 0.207, KOLFbig 0.185 → 0.121)
because the DE metrics lose more than mse gains (mse stays floored at 0: raw values are still ≥ 0.98).
Side result: at full size, correctly paired base signatures beat shuffled ones on both held-out sets, unlike the
12-target KOLF pilot.
