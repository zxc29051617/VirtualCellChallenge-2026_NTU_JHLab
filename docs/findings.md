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

## 8. Predicting the control unchanged forfeits a metric outright

The direction metric treats a predicted fold change of exactly zero as a **miss**, not as an
abstention: "A gene the submission calls but leaves without a direction, through a null, NaN
or exactly zero predicted fold change, stays in n_pred and scores as a miss."

A control-pasting submission has zero fold change everywhere, so it scores 0 on a metric
whose baseline is 0.51 and whose span is the narrowest of the six. Working the spec's scale
`(u − b)/(r − b)` through all six:

| metric | raw | b | scaled |
| --- | --- | --- | --- |
| `pds` | 0.500 | 0.500 | 0.00 |
| `mse` | ~1.008 | 0.989 | 0.00 (clamped) |
| `fid` | **0** | 0.51 | **≈ −1.7** |
| `reach` | ~0 | 0.07 | ≈ −0.08 |
| `jac` | ~0 | 0.03 | ≈ −0.08 |
| `nmae` | 1.008 | 1.001 | ≈ −0.01 |
| **overall** | | | **≈ −0.31** |

Roughly −0.28 of that comes from one metric, and from a property of the submission rather
than from anything the model got wrong.

The fix is to carry a direction over from public data: the mean log fold change across all
perturbations in Replogle's two lines, which is the generic consequence of breaking
something. It covers 9,357 of the challenge's 18,533 genes and 93% of the 3,000 a submission
carries. Applied at the measured magnitude it moves 734 genes — 615 down, 119 up, which is
the expected shape for essential-gene knockdown — at a median |log fold change| of 0.223.

### Integer rounding caps this at about a quarter of the genes

Only 26% of carried genes move, because predictions must be integer counts and the generic
response is small: its median |log fold change| is 0.055, while a gene averaging 5 counts
needs 10% to reach 6. Amplifying to compensate is the wrong fix — it distorts magnitude to
work around a format artifact, and even ×8 only reaches 52%.

The right fix is that the 400 cells of a perturbation need not be identical. Metrics read
group-summed profiles, so a target mean of 5.3 is representable exactly as 70% of cells at 5
and 30% at 6. That would lift coverage from 26% to the 93% the public data supports, and it
survives virtual storage: the same 400 distinct rows serve all 300 perturbations of a
context, so 1,200 stored rows still back all 360,000. Not yet built.

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

**Prediction registered before running it**, so the outcome cannot be rationalized after the
fact: averaging several donor contexts should beat a single one, because the part of a shared
response that is idiosyncratic to the donor averages out while the part common to
perturbation response everywhere survives. If instead a four-context average lands at the
same r ≈ 0.345, then shared responses are close to context-specific, collecting more contexts
will not help, and the approach in section 7 — leaning on identification rather than
magnitude — is the one worth developing.

Two caveats the result will have to be read against. The Tian contexts come from a different
lab, chemistry and perturbation panel than Replogle, so some of any gap will be batch rather
than biology. And iPSC and day-7 neuron are related samples from one study, so they count as
fewer than two independent contexts.

### The experiment was attempted and cannot be run on this data

The three files were downloaded and inspected. It fails on two independent counts.

**There is no perturbation panel shared across contexts.** Single-gene perturbations with at
least 25 cells, and their pairwise overlap:

| | K562 | RPE1 | iPSC | neuron d7 | neuron 2021 |
| --- | --- | --- | --- | --- | --- |
| **K562** | 2,003 | 1,790 | 10 | 10 | 51 |
| **RPE1** | 1,790 | 2,122 | 10 | 10 | 59 |
| **iPSC** | 10 | 10 | 26 | 26 | 2 |
| **neuron d7** | 10 | 10 | 26 | 26 | 2 |
| **neuron 2021** | 51 | 59 | 2 | 2 | 184 |

Shared by all five: **0**. Shared between the Replogle pair and the Tian trio: **0**.
Comparing shared responses measured on disjoint gene panels confounds context with panel, so
a low correlation would say nothing about context.

Tian 2019 also turns out to be a combinatorial screen — 39,162 labels of the form
`ATP5A1_HMGCS1_UQCRQ`, of which only 26 are single-gene.

**The one overlap that exists is too small to resolve anything.** K562 ∩ neuron-2021 is 51
perturbations. Subsampling the Replogle pair to that size to measure the noise floor:

| panel size | RPE1 reliability | K562 reliability | cross-line r |
| --- | --- | --- | --- |
| 50 | 0.669 | **0.222** | 0.176 |
| 100 | 0.803 | 0.294 | 0.205 |
| 400 | 0.942 | 0.568 | 0.279 |
| 1,827 | 0.987 | 0.842 | 0.316 |

At 50 perturbations K562's shared response cannot even be reproduced from two halves of its
own data. An experiment meant to distinguish 0.345 from something better cannot be run at a
noise floor of 0.222.

**So the claim in this section stands untested.** It is not supported by a failed experiment
and it is not refuted by one; the public CRISPRi data available here does not contain the
matched multi-context panel the test requires. Saying so is more useful than reporting a
number produced at a noise floor above the effect.
