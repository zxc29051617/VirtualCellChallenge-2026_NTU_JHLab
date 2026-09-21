# Findings

> **2026-09-21 correction and follow-up:** See
> [output roundtrip, null DE, new-data coverage, and generative pilot](offline_validation_20260921.md).
> Historical continuous-profile scores below did not exercise integer output/readback.
> Scaling before integerization changes cell distributions; floor/ceil is not a biological
> noise model; independent-gene NB is not equivalent to flow/diffusion. The original
> scaled/calibrated local models use target-context perturbations for fitting and are not
> source-only zero-shot estimates. The new experiments report their own fixed splits.

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

### Identical cells threw away three quarters of it; varying them recovered it

Writing one integer profile and repeating it 400 times moved only 26% of carried genes. The
generic response has a median |log fold change| of 0.055, while a gene averaging 5 counts
needs 10% to reach 6, so most of the signal rounded away. Amplifying to compensate is the
wrong fix — it distorts magnitude to work around a format artifact, and even ×8 reaches only
52%.

The cells of a perturbation need not be identical. Metrics read group-summed profiles, so a
mean of 5.3 is representable exactly as 70% of cells at 5 and 30% at 6, and each gene's
surplus cells are drawn independently so the cells differ the way real ones do.

| | identical cells | varied cells |
| --- | --- | --- |
| genes with a nonzero fold change | 734 (26%) | **2,733 (95%)** |
| direction | 615 down, 119 up | 1,951 down, 782 up |
| median abs. log fold change | 0.223 | 0.049 (as measured, not amplified) |

Virtual storage survives it: one context's 400 cells serve all 300 of its perturbations, so
1,200 stored rows still back all 360,000. The file goes from 70 MB to 107 MB, the submission
sits at 23% of the density cap, and `vcc prep` passes.

## 9. The first real score: −0.9934, and a metric floored by a format decision

`transfer.vcc`, submitted 2026-09-20. Scaled score above, raw below:

| metric | scaled | raw | baseline `b` |
| --- | --- | --- | --- |
| `pds` | 0.002 | 0.501 | 0.500 |
| `mse` | 0.000 | **990.4** | 0.989 |
| `jac` | 0.001 | 0.031 | 0.03 |
| `nmae` | **−6.000** | **52.6** | 1.001 |
| `fid` | −0.045 | 0.499 | 0.51 |
| `reach` | 0.082 | 0.150 | 0.07 |
| **overall** | **−0.9934** | | |

`nmae` alone contributes −1.0 of the −0.993; the other five together are about zero.

### An unstored gene is not an abstention, it is a claim of zero expression

The submission carried the top 3,000 genes by expression. The DE metrics test the 9,923
genes whose control expression exceeds 5 CPM, so **6,923 of the tested genes — 70% — were
sent as zero**, and a sparse matrix's absent entries are zeros, not gaps. Their predicted
log fold change is `log2(1e-9 / control)`, around −32, against a reference of order ±0.5.

Each such gene contributes roughly 50 to a metric whose no-skill point is 1.0. Observed 52.6
is what ~70% coverage predicts. `mse` at 990 is the same wound: the denominator is the true
perturbation effect, which for this panel is small, so a systematic distortion swamps it.

Predicting nothing at all would have scored better than predicting something for a third of
the genes and zero for the rest.

### Partial coverage does not help, because the floor binds at 4.6

`nmae` scales as `(u − 1.001)/(0.40 − 1.001)`, so the −6 floor binds at a raw value of 4.6.
At ~50 per uncovered gene, staying under it needs **93% coverage or better**. Going from 30%
to 50% moves the raw value from 52 to 35 and changes the score not at all.

### Coverage is cheap; the strategy was buying the wrong thing

With varied cells a gene is nonzero in `min(mean, 1)` of them, so one averaging 0.1 counts
costs a tenth of one averaging 10. Taking the top N by expression buys the most expensive
genes first. Spending the same budget cheapest-first over the DE-admitted set instead:

| entries/cell | top by expression | cheapest-first | prep RAM |
| --- | --- | --- | --- |
| 3,000 | 3,000 (30%) | **5,181 (52%)** | 13 GB |
| 5,000 | 5,000 (50%) | 7,181 (72%) | 22 GB |
| 7,000 | 6,022 (61%) | 9,181 (93%) | 30 GB |
| 7,742 | 9,923 (100%) | 9,923 (100%) | 33 GB |

Full coverage costs 2.79e9 entries — 59% of the scoring cap, so the cap was never the
binding constraint. What binds is that `vcc prep` loads the indices array whole as int64:
33 GB against 31.2 GB of RAM, of which 16.9 GB was free.

### The other result: per-target transfer bought nothing

`pds` came back at 0.501 against a floor of 0.500. Predicting each of the 300 targets from
its own measured effect in gwps — which locally reaches 0.82 between Replogle's two lines —
moved discrimination by 0.001. Whether the transfer itself fails across this distance or the
zeroed genes swamped it cannot be separated from one submission.

`reach` is the one metric clearly above baseline: 0.150 against 0.07, scaled 0.082.

## 10. Where the leaders' points actually come from

Top three on 2026-09-20, scaled score with raw beneath:

| | Illumina AI | cqawesome | chill |
| --- | --- | --- | --- |
| submissions | 36 | 61 | 20 |
| **overall** | **0.3449** | **0.2967** | **0.2611** |
| `pds` | 0.853 (0.884) | 0.826 (0.873) | 0.748 (0.837) |
| `mse` | 0.423 (0.586) | 0.389 (0.619) | 0.232 (0.767) |
| `reach` | 0.373 (0.409) | 0.278 (0.326) | 0.241 (0.292) |
| `nmae` | 0.293 (0.825) | 0.258 (0.844) | 0.276 (0.836) |
| `fid` | 0.111 (0.545) | 0.019 (0.518) | 0.040 (0.524) |
| `jac` | 0.016 (0.037) | 0.011 (0.035) | 0.030 (0.041) |

**Discrimination is 41% of the leader's total**, and `pds` + `mse` + `reach` together are 79%.

### This refutes section 3's strategic reading

Section 3 argued that `jac` deserved the effort because exact reproduction scores 2.48–2.85
there, the largest headroom in the suite. The leaderboard says otherwise: the top three score
0.011 to 0.030 on it, having made 20 to 61 submissions between them. Headroom is not the same
as reachable. `fid` is the same story — raw 0.518 to 0.545 against a 0.51 baseline, so even
the leader is 4.5% better than a coin flip.

The metrics people actually move are the ones this repo's transfer model targets. That is
encouraging for the approach and a correction to where the effort was pointed.

### What our own first submission establishes

| | ours | leader |
| --- | --- | --- |
| `pds` | 0.002 | 0.853 |
| `mse` | 0.000 | 0.423 |
| `nmae` | −6.000 | 0.293 |
| overall | −0.9934 | 0.3449 |

Of the −0.9934, −1.0 is the `nmae` floor and that is a format defect rather than a model one
(section 9). Fixing it should return roughly 0 — the starting line, not a result. Whether
`pds` moves off 0.501 is the first clean test of whether cross-context transfer works at this
distance at all; the previous attempt could not answer it, because the zeroed genes distorted
every profile the metric compared.

## 11. A fold-change heuristic matches STATE's published zero-shot discrimination

Arc publishes `ST-SE-Replogle` on HuggingFace: STATE trained on Replogle, with zero-shot
checkpoints for four cell lines. Each carries the evaluation `cell-eval` produced, and the
aggregate CSVs are 3.5 KB each — the comparison costs a 14 KB download, not the 71 GB repo.

STATE's zero-shot numbers, each line held out of training entirely:

| | RPE1 | K562 | HepG2 | Jurkat |
| --- | --- | --- | --- | --- |
| `discrimination_score_cosine` | 0.617 | 0.732 | 0.646 | 0.673 |
| `overlap_at_N` | 0.027 | 0.094 | 0.120 | 0.060 |
| `de_direction_match` | 0.556 | 0.634 | 0.662 | 0.604 |
| `pearson_delta` | 0.104 | 0.200 | 0.258 | 0.157 |

Against `scripts/evaluate_local.py`'s `transfer` — carry the source line's log fold change
onto the target's control profile, and nothing else:

| task | ours | STATE zero-shot |
| --- | --- | --- |
| predict RPE1 | **0.812** | 0.617 |
| predict K562 | **0.753** | 0.732 |

Candidate count is not the explanation. PDS is a rank among candidates, so a larger panel is
a harder task; scoring 914, 1,462 and 1,736 perturbations gives 0.821, 0.812 and 0.813, so
the comparison at STATE's 1,499 holds.

### What this does and does not show

Three differences remain, and two of them favour us:

- STATE's zero-shot trains on the other three lines; `transfer` uses one.
- Our gene axis is the 7,226 shared between two lines; STATE's is Replogle's full set.
- `discrimination_score_cosine` in `cell-eval` may not remove the whole target panel from
  the feature space, which the vcc2026 metric does and our implementation does. That would
  make the two numbers not quite the same statistic.

And one caveat outweighs the result: **our local 0.82 did not survive contact with the
Challenge**, where the same model scored 0.501. Local numbers, ours and STATE's alike, are
measured between two Replogle cell lines grown in one lab. They do not establish what
survives the move to an unseen context.

### The pattern that does hold everywhere

STATE's own DE metrics sit near chance on held-out lines — `overlap_at_N` 0.027 to 0.120,
`de_direction_match` 0.556 to 0.662 — which is the leaderboard's shape (section 10) and our
own. Three independent measurements agree: discrimination transfers across contexts, and
differential expression does not. That is the finding to build on, and it is not one this
repo produced alone.

## 12. The cell construction is fixable; packaging it on this machine is not

`cells_from_nb` draws each gene's per-cell weights from a gamma and distributes its fixed
total by a multinomial — a negative binomial conditioned on that total. The group mean
survives exactly while the cells spread. Against real control cells with no perturbation at
all:

| construction | depth | false DE |
| --- | --- | --- |
| real cells | — | **0** |
| floor/ceil | 1.00 | 7,581 |
| **conditioned NB** | **1.00** | **8** |
| conditioned NB | 0.50 | 6,638 |
| conditioned NB | 0.15 | 9,884 |

**The fix only exists at full depth.** Below it our counts sit on a coarser grid than the
reference's, and the rank test reads that as a difference however the profile is normalized
— which is what the "every metric normalizes, so scaling is free" argument in section 9
missed. At 0.15 the NB is worse than the floor/ceil it replaces.

### And full depth costs more than vcc prep can package

NB is about 25% cheaper per gene than floor/ceil, but depth 1.0 more than undoes that:
5,778 entries per cell, 2.08e9 total. That is 44% of the scoring cap — the rules were never
the constraint — and the same density as a real control cell, which carries 5,975 nonzero
genes.

`vcc prep` fails on it, three times, at 24, 30 and 40 GB of commit free, always on the same
allocation. With `VCC_TRACEBACK=1`:

```
File "vcc/prep.py", line 376, in _all_values_integer
    return not bool(np.any(np.abs(frac) > 1e-6))
numpy MemoryError: Unable to allocate 7.75 GiB for shape (2080105930,) float32
```

It materializes the whole value array to check the counts are whole numbers, then copies it
again for `abs`. `--no-require-counts --allow-discrete` skips that check and the run reaches
a later 1.94 GiB bool allocation, which fails with 41.8 GB free — so by then it holds nearly
40 GB. Chasing allocations one at a time has no visible end.

**Measured ceiling: 0.94e9 entries packages, 2.08e9 does not.** The official density cap is
4.75e9, so prep tops out around a fifth of what the rules allow.

### No compromise keeps the benefit

| candidate | entries | packages? | what it costs |
| --- | --- | --- | --- |
| v4 as built | 2.08e9 | no | — |
| depth 0.8 | 1.99e9 | no | saves 4% |
| depth 0.5 | 1.55e9 | maybe | false DE 8 → 6,638 |
| depth 0.15 + NB | 0.79e9 | yes | false DE 9,884, worse than floor/ceil |
| genes ≥1 count only | 1.63e9 | maybe | zeroes 4,280 gate genes → nmae back to −6 |

The cost is concentrated in the ~5,600 genes the control expresses above one count, each
nonzero in nearly every cell. That is the price of cells that look like real ones, and it is
not negotiable downward.

**So the next gains have to come from somewhere that does not touch density**: the transfer
itself. `pds` is 41% of the leader's total and the one metric where we score (0.315 against
0.853), and improving it costs no entries at all — more source coverage (CD4 adds 19 targets
past QC), a better effect scale than the assumed 0.3, better fold-change estimates.

`transfer_v4.h5ad` is kept as built. It is a legal submission that this machine cannot
package; a machine with more memory, or a prep that chunks that check, would.

## 13. A knockdown's specific effect survives leukemia line → primary T cell

Section 7 measured the perturbation-specific component transferring between Replogle's two
lines at r = 0.024, and read that as enough to identify a perturbation but little else. That
number admits two readings with opposite consequences: the signal is real but measured noisily
in any one screen, in which case a second independent measurement of the same knockdown
denoises it; or responses genuinely are context-specific, in which case no amount of averaging
helps. Section 12 left the transfer as the only place gains can come from, so which of the two
holds decides whether more sources are worth assembling at all.

It is the first. Marson 2025's genome-wide CRISPRi screen in primary CD4 T cells overlaps
Replogle's K562 genome-wide screen on 218 targets past the authors' QC flags, over 6,629
shared genes. Correlating every target's CD4 fold change against every target's K562 fold
change and splitting the matrix on its diagonal:

| | mean r | sd |
| --- | --- | --- |
| matched — both sides knocked down the same gene | **+0.0566** | 0.0539 |
| mismatched — different genes | +0.0009 | 0.0196 |

15.2 standard errors apart, and a target's own match outranks 85.9% of its mismatches. Scored
as the Challenge scores discrimination — is each prediction closest to its own target, with
the whole target panel removed from the feature space — CD4 predicts K562 at **0.6611**
against a no-information value of 0.5.

Two things make that stronger than it looks. The mismatched mean is +0.0009, so what the two
screens share is almost entirely target-specific rather than a generic CRISPRi stress response
that would inflate both. And the context gap here — a chronic myeloid leukemia line against
primary CD4 T cells, different lab, different chemistry — is larger than anything the
Challenge's six undisclosed lines are likely to span, so 0.6611 reads as a floor.

So r = 0.024 was measurement noise on a real signal, and a second source should denoise it.
This costs no stored entries, which is the constraint section 12 ran into.

### Reading it off S3 rather than downloading it

The CD4 statistics are a 16.8 GB h5ad. `layers/log_fc` is contiguous, so a row is 10,282
float64 of adjacent bytes and h5py will fetch it through a file object that issues HTTP range
requests. 645 rows cost 0.12 GB in 452 requests over 15 minutes. Neither ros3 nor fsspec is
available here; `src/vcc2026/remote.py` is 90 lines. Every request retries with backoff —
the first run died after several hundred good ones on a single timed-out SSL read.

### What this does not yet show

That the blend scores better, which is a separate claim and is being measured against a held-
out answer key. The first attempt at that could not run: CD4 was fetched for the Challenge's
300 targets, and those overlap Replogle's RPE1 and K562 profiles on **0** perturbations — both
libraries target ~2,400 essential genes and the Challenge's targets are not essential. The
scorable fold is RPE1 ∩ K562gw ∩ CD4 instead, which is 744 targets in 1,894 rows.

That test carries a control worth stating, because without it a gain proves nothing. Blending
is also scored with CD4's numbers attached to the wrong knockdowns: magnitude, sparsity and
gene-level structure all survive the shuffle and only the target-specific part dies. A blend
that beats the single source but not the shuffle gained nothing from knowing which gene was
knocked down. Two ways it could win falsely — least-squares calibration shrinks a weakly
agreeing source hard, and shrinkage toward no effect lowers squared error on its own; and a
shared generic response would lift the blend without touching discrimination.

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
