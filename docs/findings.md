# Findings

Measurements from the local evaluation loop (`scripts/evaluate_local.py`), K562 → RPE1 on
Replogle 2022. How the loop works and where its metrics depart from the official suite is in
`docs/validation_strategy.md`; read that before acting on a number here.

All correlations below are over the 7,225 genes expressed in both lines, in log-fold-change
space, across 1,827 shared perturbations.

---

## 1. Cross-line fold changes identify a perturbation but do not place its response

| model | pds ↑ | expr_error ↓ | direction ↑ | jaccard ↑ |
| --- | --- | --- | --- | --- |
| `control` | 0.5000 | 1.0000 | 0.0000 | 0.0138 |
| `mean_response` | 0.5000 | **0.7270** | **0.7671** | **0.0989** |
| `transfer` | 0.8208 | 1.1613 | 0.5928 | 0.0243 |
| `scaled` | **0.8273** | 0.9785 | 0.5806 | 0.0242 |
| `calibrated` | 0.6301 | 0.9507 | 0.7163 | 0.0234 |

`transfer` reaches 0.82 on discrimination against a 0.50 floor while scoring *worse than
predicting nothing* on expression error. `mean_response` reads the answer key, so it is the
yardstick rather than a competitor.

This is the leaderboard's own shape in miniature: discrimination nearly saturated, the
DE-based metrics not.

## 2. The fault is composition, not magnitude

The transferred prediction is smaller than the truth, not larger — ‖predicted‖ / ‖measured‖
is 0.637 — so overshoot is not the explanation. What differs is the mix:

| | shared across perturbations | perturbation-specific |
| --- | --- | --- |
| measured in RPE1 | 27.5% | 72.5% |
| transferred from K562 | 4.2% | 95.8% |

It carries almost none of the common response and is otherwise mostly noise.

## 3. A uniform scale is free; reweighting the two halves is not

`scaled` moves expression error from 1.16 to 0.98 and leaves discrimination untouched. It
cannot do otherwise: cosine distance ignores a uniform scale.

`calibrated` reaches 0.95 by pushing the shared half up (×1.38) and the specific half down
(×0.23), and pays 0.20 of discrimination, because the shared half is by definition identical
across perturbations and swamps what separates them.

**Least-squares shrinkage is optimal for squared error and actively harmful for the metric
that rewards telling perturbations apart.** The competition averages six metrics precisely to
punish this; the effect is reproducible on public data in one afternoon.

## 4. The shared response is a real, highly reproducible property of a cell line

Split each line's perturbations into disjoint halves and measure its shared response twice:

| | split-half reliability |
| --- | --- |
| RPE1 | **r = 0.987** |
| K562 | r = 0.850 |

So the target is not noise. It is one of the most stable quantities in the dataset.

## 5. But it does not transfer, and that is biology rather than measurement

| | r |
| --- | --- |
| K562 shared response → RPE1, measured | 0.316 |
| same, corrected for measurement noise | **0.345** |

Correcting for attenuation barely moves it, because there is barely any attenuation to
correct. Two cell lines genuinely respond differently to having something broken, and no
amount of extra sequencing changes that.

## 6. And it is not predictable from the control profile

Candidate predictors of RPE1's shared response, none of which use RPE1 perturbation data:

| predictor | r |
| --- | --- |
| K562's shared response | +0.316 |
| own baseline expression level | +0.139 |
| expression → response curve learned on K562 | +0.141 |
| drift toward K562's state | −0.164 |
| drift toward the average of the two | −0.164 |
| **all of the above, oracle weights** | **+0.367** |

Even the oracle combination leaves 72.9% of the shared response unexplained. The obvious
features do not carry it.

## 7. A per-gene correlation of 0.024 is enough to identify a perturbation

The perturbation-specific component transfers at **r = 0.024** across the two lines — for
practical purposes, zero. Yet the model built on it scores 0.82 on discrimination.

The likely mechanism is dimensionality: the metric ranks by cosine distance over 7,225 genes,
and aggregating a negligible per-gene signal across that many coordinates separates the right
match from 913 wrong ones. Accuracy and identifiability are close to independent here.

---

## What this adds up to

The shared response is reliably measurable (r = 0.987), worth 27.5% of the signal, and the
single thing standing between a transfer model and the baseline on every magnitude metric.
It cannot be transferred from one other cell line (r = 0.345 ceiling) and cannot be read off
the control profile (r = 0.367 with oracle weights).

**The bottleneck is the number of cell contexts available to learn from, not the modelling of
any one pair.** Two lines cannot teach a mapping from control profile to perturbation
response; they can only be compared. This is consistent with where the leading approaches put
their effort — STATE pretrains across 70 cell contexts and over 100 million perturbed cells —
and it reframes the problem from "find a better estimator" to "assemble enough contexts."

### The test that would settle it

`scPerturb` (Zenodo 13350497) carries CRISPRi screens in contexts beyond K562 and RPE1:

| file | context | size |
| --- | --- | --- |
| `TianKampmann2019_iPSC.h5ad` | iPSC | 351 MB |
| `TianKampmann2019_day7neuron.h5ad` | day-7 neuron | 269 MB |
| `TianKampmann2021_CRISPRi.h5ad` | neuron | 289 MB |

With four or five contexts instead of two, hold one out and ask whether a shared response
predicted from the remaining ones beats the r = 0.345 that a single donor line achieves. That
is a direct test of the claim above, and it is the smallest experiment that can refute it.
