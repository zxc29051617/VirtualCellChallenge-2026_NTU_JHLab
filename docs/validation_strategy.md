# Validation Strategy

## The constraint that drives everything

The 2026 Challenge is zero-shot. No training set is released, and no ground truth ships with
the challenge data — not for the final contexts, and not for `A`, `B`, `C` either. Every cell
in `dataset/` is an unperturbed control.

This rules out the obvious approach. You cannot hold out perturbations from the challenge
data, because there are no measured perturbation responses in it to hold out. The only
feedback loop the challenge itself offers is the leaderboard, which is slow, rate-limited,
and ranks on your *most recent* submission rather than your best.

So local validation has to be built on an external dataset that has measured perturbation
effects. The organizers explicitly permit this: participants may use public perturbation
data or their own.

## The external dataset: Replogle et al. 2022

Downloaded by `scripts/download_external_data.ps1` into `external/`. Contents as verified on
2026-09-14:

| File | Cell line | Cells | Genes | Perturbations |
| --- | --- | --- | --- | --- |
| `ReplogleWeissman2022_K562_essential.h5ad` | K562 | 310,385 | 8,563 | 2,058 |
| `ReplogleWeissman2022_rpe1.h5ad` | RPE1 (retinal pigment epithelial) | 247,914 | 8,749 | 2,394 |

**2,056 perturbations are measured in both cell lines** — all but two of K562's targets — and
7,226 genes are shared. That is what makes this dataset the right choice: the same
interventions were applied to two different cellular contexts, so a model can be fit on one
context and scored on the other over an identical perturbation set. It is a structural
analogue of the Challenge, not merely a source of perturbation data.

Both use CRISPRi, matching the Challenge's perturbation modality. The assay generation
differs (Perturb-seq rather than 10x Flex), which limits leakage risk.

The relevant `obs` columns are `perturbation` (target gene symbol), `cell_line`, and
`perturbation_type`; `guide_id` and `batch` are available if technical covariates matter.

### Provenance check (done 2026-09-14)

Both papers were read and the files inspected directly. Findings:

- **Counts match exactly.** scPerturb Table 1 lists 310,385 cells / 2,058 perturbations for
  K562 essential and 247,914 / 2,394 for RPE1 — verbatim what we measured. Replogle's own
  text reports the K562 essential screen targeted 2,057 genes (+1 non-targeting label = 2,058).
- **scPerturb repackages rather than re-processes**, by their own account: most source
  datasets supplied only processed matrices, and guide-assignment procedures are
  setup-specific. So the data is Replogle's, reformatted — not independently re-derived.
- **`.X` holds raw integer counts.** Verified by sampling: dtype is float32 but every value
  is an exact integer, min 0, max 953 (K562). `ncounts` equals the row sum of `.X`.
  `UMI_count`, `core_scale_factor` and `core_adjusted_UMI_count` are Replogle-derived
  quantities that are *not* applied to `.X`. The raw-count requirement is satisfied.
- **Method is sound.** Two sgRNAs per gene, cells bearing mismatched guides excluded, median
  on-target knockdown 85.5% (K562) / 91.6% (RPE1). Because CRISPRi lets knockdown be
  measured, guides that failed to silence are detectable rather than silently mislabeled.

### Caveats, in order of how much they matter

1. **The shared perturbation set is not independent of K562.** The RPE1 library was DepMap
   essentials *plus genes hand-selected for interesting phenotypes in the K562 genome-wide
   screen*. A K562→RPE1 result is therefore optimistic relative to a random gene panel.
   Treat local scores as a relative signal for model iteration, never as an estimate of
   Challenge performance.
2. **Technical differences between the two lines.** Different KRAB effector (KOX1 for K562,
   ZIM3 for RPE1), different knockdown efficiency, different harvest day (6 vs 7), different
   media. Part of any cross-line gap is technical, not biological.
3. **Essential-gene bias.** Both panels are common-essential genes with strong phenotypes,
   easier than the Challenge's broader 300-gene panel.
4. **Gene space is ~8.5k, not the Challenge's 18,533.** Score magnitudes are not comparable
   to the leaderboard.

Reassuringly, Replogle report median phenotype correlation of only r = 0.23 between K562 and
RPE1, so cross-line transfer remains a genuinely hard test despite caveat 1.

### Reduced profiles

`scripts/build_external_profiles.py` collapses each file to per-perturbation mean profiles
(`vcc2026.io.group_mean_profiles`), writing `external/profiles/<line>_perturbation_means.npz`
with `means`, `perturbations`, `counts`, `genes`. Roughly 33 MB and under a minute per file,
versus ~10 GB and repeated decompression for the raw matrices.

Operational facts confirmed from the reduced profiles:

- The control label is `control` (not `non-targeting` as in the Challenge data), and it is the
  largest group in both lines: 10,691 cells (K562) and 11,485 (RPE1).
- Cells per perturbation is very uneven: median 121 (K562) / 72 (RPE1), but the minimum is
  5 and 2 respectively. Perturbations at the low end give noisy means — filter by `counts`
  before treating a profile as ground truth.

### End-to-end validation: on-target knockdown is visible

For every perturbation whose target gene is also measured, the target's own expression should
fall relative to control. Measured from the reduced profiles:

| | Targets checked | Median residual vs control | Below 70% of control |
| --- | --- | --- | --- |
| K562 | 1,866 | 16.7% | 97% |
| RPE1 | 2,106 | 9.8% | 97% |

That is ~83% and ~90% median knockdown, against the 85.5% and 91.6% Replogle report. The
agreement confirms three things at once: the data is what the paper describes, the streaming
reader aligns codes to rows correctly, and the perturbation labels are trustworthy.

What it is **not** is a route to a better Challenge score. The official metric spec excludes
each perturbation's own target gene from all six metrics, and `pds_cosine` removes all 300
panel targets from the feature space entirely. Predicting the knockdown itself earns nothing;
it is treated as restating the experiment's premise. Use this check to validate data and code,
and look for signal in the downstream response instead. See `docs/challenge_brief.md`.

### Practical: `.X` is stored dense

A naive `anndata.read_h5ad` allocates roughly 10.6 GB (K562) and 8.7 GB (RPE1). Read in row
blocks, the way `vcc2026.io.control_gene_profile` already streams the challenge files.

Optional: `-IncludeGwps` adds the genome-wide K562 screen (8.81 GB, ~2.5M cells). Not needed
to get started, and large enough to slow every iteration. Skip it until the pipeline works.

## The splits, in order of usefulness

### A. Random split — pipeline smoke test only

Split cells at random, ignoring which perturbation or cell line they came from.

Use it to catch loader bugs, shape mismatches, and metric wiring errors. Nothing else. It
leaks both perturbation and context across train and validation, so its score is meaningless
as a measure of quality and will be wildly optimistic.

### B. Perturbation holdout — unseen targets, seen cell line

Hold out a subset of target genes entirely; train on the rest within the same cell line.

This tests whether the model extrapolates to perturbations it has never seen. It is a real
test, and it maps onto the 2025 Challenge, but it is *not* this year's task: it still lets the
model learn what perturbation responses look like in the cell line it is being scored on.

### C. Cell-line holdout — the one that matters

Fit on K562, evaluate on RPE1 (and the reverse, as a second fold).

This is the local analogue of the 2026 task: the evaluation cell line contributes no
perturbation data to fitting. Its score is the one that should drive model selection.

Restrict evaluation to targets present in both cell lines, so that differences in score
reflect context generalization rather than differences in which genes were screened.

Expect scores here to be much worse than under A or B. That gap is the point — it is the
thing the Challenge is measuring, and a model that looks good under B but collapses under C
is exactly the failure mode the 2026 format was designed to expose.

## Scoring locally

Score with `cell-eval` so that local numbers are comparable in kind to the leaderboard, and
normalize against a baseline as the Challenge does (`cell-eval score`), so that "0 means no
better than predicting the control mean" holds locally too.

Two caveats:

- `cell-eval` v0.8.2's `vcc` profile computes the 2025 three-metric panel
  (`mae`, `discrimination_score_l1`, `overlap_at_N`), not the 2026 six-metric panel. Local
  scores indicate direction of travel; they do not reproduce leaderboard values.
- Of the 2026 metrics, `fid` (direction correctness) and `jac` (significant-DE-gene overlap)
  are where the entire field is stuck — the current leader scores 0.027 and 0.026 on them.
  `jac` also carries the largest headroom in the suite by construction: exact reproduction of
  the reference scores 2.48–2.85 there, against 1.00–1.17 for the expression error and
  discrimination. Prefer local proxies that track those two.
- Every metric excludes the perturbation's own target gene, so a local proxy that rewards
  predicting the knockdown itself measures nothing the Challenge scores.
  See `docs/challenge_brief.md`.

## The loop, and what it says so far

`scripts/evaluate_local.py` fits on one cell line and scores the other. Metrics live in
`vcc2026.local_eval`. Three reference models are built in:

- `control` — predict no change.
- `mean_response` — one averaged response assigned to every perturbation. This is the shape
  of the official baseline `b`. It reads the answer key, so it is the yardstick, not a
  competitor.
- `transfer` — carry the fold change measured in the source line onto the target's own
  control profile. The first model here that is actually zero-shot.
- `scaled` — `transfer` with one weight on the whole log fold change.
- `calibrated` — `transfer` with separate weights on its shared and perturbation-specific
  halves.

The two weighted models are fit on the fit half of the perturbations and scored on the other
half, so no model is tuned on what it is scored against. In the Challenge those constants
would come from public cell-line pairs rather than from the scored context.

K562 → RPE1, 1,827 usable perturbations split 913 fit / 914 scored, 7,226 shared genes:

| model | pds ↑ | expr_error ↓ | direction ↑ | jaccard ↑ |
| --- | --- | --- | --- | --- |
| `control` | 0.5000 | 1.0000 | 0.0000 | 0.0138 |
| `mean_response` | 0.5000 | **0.7270** | **0.7671** | **0.0989** |
| `transfer` | 0.8208 | 1.1613 | 0.5928 | 0.0243 |
| `scaled` | **0.8273** | 0.9785 | 0.5806 | 0.0242 |
| `calibrated` | 0.6301 | 0.9507 | 0.7163 | 0.0234 |

Two implementation checks land exactly on values the official spec predicts, which is good
evidence the discrimination metric is faithful: a constant prediction scores exactly 0.5000,
and a zero-effect prediction scores exactly 1.0000 on the expression error.

`control` scoring 0.0000 on direction is not a bug. A predicted fold change of exactly zero
carries no direction, and the official rule counts that as a miss rather than dropping it.

### What the numbers say

**Cross-line fold changes identify a perturbation but do not place its response.** `transfer`
reaches 0.82 on discrimination against a 0.50 floor while scoring *worse than predicting
nothing* on expression error. Measured directly: the two lines' shared response correlates at
r = 0.475, but the perturbation-specific part correlates at only r = 0.158 — weak enough to
be useless for magnitude, strong enough to rank.

**The composition, not the magnitude, is what is wrong.** The measured response is 27.5%
shared and 72.5% perturbation-specific; the transferred one is 4.2% and 95.8%. It is missing
almost all of the common response, and what it does carry is mostly noise.

**A single weight is nearly free; reweighting the two halves is not.** `scaled` moves
expression error from 1.16 to 0.98 and leaves discrimination untouched — cosine distance
ignores a uniform scale, so it cannot do otherwise. `calibrated` reaches a better 0.95 by
pushing the shared half up and the specific half down (×1.38 and ×0.23), and pays 0.20 of
discrimination for it, because the shared half is by definition identical across
perturbations and swamps what distinguishes them.

Least-squares shrinkage is optimal for squared error and actively harmful for the metric that
rewards telling perturbations apart. Given that the competition averages six metrics, this is
the trap its scoring design is meant to punish.

**The standing gap: nothing here beats `mean_response` on magnitude.** The best transfer
variant sits at 0.95 against its 0.73, and `mean_response` reads the answer key to get there.
Closing that gap means estimating the target context's own shared response from its control
profile — which is the actual open problem, not a tuning exercise.

### Known weakness in the direction and jaccard proxies

`mean_response` scores 0.7622 on `direction` here, while the official baseline scores
0.505–0.522 on `de_wilcoxon_direction_fidelity_yield_raw`. The gap is the proxy, not the data.

The official gate is the reference's *significant* set, which is perturbation-specific. This
proxy gates on the top `--top-n` genes by |log2 fold change|, and those are dominated by the
response signature shared across perturbations — exactly what an averaged prediction gets
right. So the proxy over-credits `mean_response`, and by extension under-states how much room
a real model has above it.

Treat `pds` and `expr_error` as trustworthy, and the other two as rough. Closing this gap
means computing the Wilcoxon table from cell-level data rather than from pseudobulk means.

## Recommended order

1. Random split, once, to prove the pipeline runs end to end.
2. Cell-line holdout as the standing evaluation, from that point on.
3. Perturbation holdout only as a diagnostic, when you want to separate "cannot generalize
   to new targets" from "cannot generalize to new contexts".
