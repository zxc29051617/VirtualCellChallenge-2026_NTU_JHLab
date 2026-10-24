# Virtual Cell Challenge 2026 Brief

Primary source: Arc Virtual Cell Initiative Team and Goodarzi, H. (2026). "Virtual Cell
Challenge 2026: Benchmarking zero-shot generalization across cellular contexts." *Cell* 189,
5827–5830. doi:10.1016/j.cell.2026.08.004 (open access, CC BY-NC-ND). Local copy in
`papers/`.

Corroborated against the live leaderboard API (`https://virtualcellchallenge.org/api/leaderboard`)
and the installed `cell-eval` source. Verified 2026-09-14.

## What the challenge asks

- Predict CRISPRi knockdown responses from unperturbed control cells only.
- Six cell lines from different tissues, profiled with CRISPRi and 10x Flex scRNA-seq.
  Three are used for validation (`A`, `B`, `C`); three distinct ones for the final test.
- Both phases have the same shape: 3 cell lines × 300 perturbations × 400 cells per
  perturbation per cell line.
- Inputs per context: expression profiles of cells expressing non-targeting guides, plus the
  gene identifiers for the CRISPRi knockdowns to be predicted. Nothing else.

The paper is explicit about why the data is shaped this way: "This design intentionally
separates access to the target context from access to perturbation outcomes in that context.
The non-targeting guide profiles define the basal state of each cell line, while the
perturbation identities define the interventions to be simulated. The withheld measurements
define the ground truth."

## This year is zero-shot: there is no training set

The 2026 challenge releases **no challenge-specific training data**. Unlike 2025 — which
provided 150 training perturbations inside a single cell line (H1 hESC) and asked for
generalization to held-out perturbations in that same line — 2026 provides no perturbed
cells at all, for any context.

Consequences for this workspace:

- There is no local ground truth, not even for `A`, `B`, `C`. The `ground_truth_cells`
  counts in `dataset/manifest.json` describe data the organizers hold, not data we have.
- Local scoring of a prediction is therefore impossible using challenge data alone.
- Participants may use public perturbation datasets or their own data. Building a local
  evaluation loop requires such an external dataset — see `papers/README.md`.

## Official submission constraints

There are **two contracts at two stages, and they deliberately differ**:

| | upload (`vcc prep`) | scoring (`cell_eval2`) |
| --- | --- | --- |
| non-targeting rows | forbidden | required, label must match the reference |
| cells per perturbation | exactly 400 | unconstrained |
| perturbation column | `target_gene` | whatever the run declares |

They are not the same file. The platform bridges them: it reads the held-out *real* control
cells out of the reference, concatenates them onto your prediction, and stamps the scorer's
perturbation column. That is why you must omit controls — the control side of every metric is
then provably held-out reference data rather than anything the submission supplied.

What `vcc prep` enforces on upload:

| Requirement | Value |
| --- | --- |
| perturbations | exactly those in `pert_counts.csv` — no extras, none missing |
| non-targeting rows | none (they are model *input* only, never copied into the prediction) |
| cells per perturbation | exactly 400, in every context |
| genes in `.var` | exactly 18,533, in the order given by `gene_names.csv` |
| stored entries in `.X` | at most 4,750,000,000 |
| `.X` values | non-negative, integral, finite, no cell total above 10⁶ |

Each of these is a rejection rather than a repair.

- One file covers all three contexts; every cell carries a `context` label in `.obs`, reused
  exactly as it appears in the control files (`A`/`B`/`C` now, `D`/`E`/`F` in the final phase).
  Contexts are scored separately and averaged, so a mislabelled cell is scored against the
  wrong reference.
- `target_gene` holds **gene symbols** (`ADNP`), not construct identifiers (`ADNP-1`). A label
  that does not resolve keeps the perturbation's own transcript in the scored vector.
- Do **not** apply `normalize_total` or `log1p`. The expression metrics compare group-summed
  profiles, which cannot be recovered from log-normalized data.
- **Store the prediction sparse.** The cap is on stored entries, not file size. A dense matrix
  at the standard shape carries 6,671,880,000 entries — 1.40× the cap — and is rejected on its
  own with no adversarial content at all.
- There is a daily submission limit. Confirmed 2026-09-20 by hitting it; the exact allowance
  is still unstated in the Commentary and the metric spec. Budget submissions accordingly —
  a wasted one costs a day.

## Timing

- 2026-08-20: validation data and live leaderboard open.
- 2026-10-22: final test inputs released.
- 2026-11-05 23:59 UTC: final submissions due.
- Mid-to-late November 2026: winners announced.

Prizes: $100,000 / $50,000 / $25,000 for first through third.

## Evaluation

Scoring changed for 2026. In the paper's words: "This year, final scoring will use a
reference-normalized aggregate across six metrics. This preserves a broad evaluation of model
behavior while reducing the incentive to optimize for a single metric."

The live leaderboard ranks on the validation set using a subset of `cell-eval`'s metrics; the
six below are what it displays.

Field names as they appear in the leaderboard API:

| Score field | Underlying metric | What it measures |
| --- | --- | --- |
| `score_pds` | `pds_cosine` | Perturbation discrimination: can predictions be matched back to the right perturbation |
| `score_mse` | `expr_mse_unbiased_capped_norm` | Expression-level squared error |
| `score_nmae` | `de_wilcoxon_lfc_nmae` | Normalized MAE of log fold changes |
| `score_fid` | `de_wilcoxon_direction_fidelity_yield_raw` | Whether predicted changes move in the correct direction |
| `score_reach` | `de_wilcoxon_direction_reach_raw` | Directional coverage across affected genes |
| `score_jac` | `de_wilcoxon_sig_jaccard` | Jaccard overlap of significant DE gene sets |

`score_avg` is the unweighted mean of the six scaled scores and is what the leaderboard ranks
on. Verified against the top entry on 2026-09-14:
`(0.830 + 0.199 + 0.026 + 0.261 + 0.027 + 0.345) / 6 = 0.2813`, matching its stated 0.2812.

The leaderboard displays, per metric, a scaled score above the raw metric value. It marks a
ceiling at 1 and a floor at ≤ 0.

The 2025 metric fields (`de_score`, `pert_score`, `mae_score`, `pearson_delta`, `pr_auc`,
`de_spearman_sig`) are still present in the API payload but are null for 2026 entries.

### Ranking rule

Rank is based on the score of each team's **most recent** submission, not its best. A worse
late submission replaces a better earlier one.

### Scores are scaled between a baseline and a replicate

From the official metric spec (`papers/vcc2026 Metric Suite (official spec).pdf`). Writing
`u` for a metric's aggregate over the panel:

```
s = (u - b) / (r - b)
```

- `b` — **baseline**: the context's mean perturbation response, assigned identically to every
  perturbation and scored as an ordinary submission.
- `r` — **replicate**: the reference compared with itself. Cells are split into two disjoint
  halves per perturbation and per control, one half scored against the other, over five splits
  (seeds derived from base seed 0).

So **0 is the baseline and 1 is a repeat of the experiment — not perfection.** A context's
score is the equal-weight mean of the six.

Values above 1 occur and **are reported rather than clipped**, because the replicate is
estimated from half-depth data. Negative values occur for a submission worse than baseline and
are likewise reported. Only two members are bounded: the expression error is clamped to [0, 1]
and the fold-change error is floored at −6.

Measured `b` and `r` on the three reference bundles, with what a submission reproducing the
reference exactly would score:

| Metric | baseline `b` | replicate `r` | exact-reproduction score |
| --- | --- | --- | --- |
| `pds_cosine` | 0.500 | 0.927 – 0.984 | 1.03 – 1.17 |
| `expr_mse_unbiased_capped_norm` | 0.986 – 0.992 | 0.028 – 0.045 | 1.000 (clamped) |
| `de_wilcoxon_direction_fidelity_yield_raw` | 0.505 – 0.522 | 0.795 – 0.832 | 1.51 – 1.72 |
| `de_wilcoxon_direction_reach_raw` | 0.047 – 0.097 | 0.958 – 0.978 | 1.02 – 1.05 |
| `de_wilcoxon_sig_jaccard` | 0.021 – 0.037 | 0.375 – 0.423 | **2.48 – 2.85** |
| `de_wilcoxon_lfc_nmae` | 1.0009 – 1.0017 | 0.369 – 0.431 | 1.58 – 1.75 |

`sig_jaccard` has by far the most headroom: its replicate is only 0.38–0.42, so the scale
divides by a small span and an excess over a half-depth replicate gains more on the average
than an equally large excess anywhere else. `direction_reach` has the least, and structurally
cannot score below 1 at exact reproduction.

The context-mean predictor in `src/vcc2026/baseline.py` is essentially the definition of `b`,
so it scores 0 by construction. On `pds_cosine` this is exact rather than approximate: a
submission emitting the same profile for every perturbation produces one shared distance row,
every perturbation takes the mid-rank, and the raw value is exactly 0.500.

### The perturbed gene's own row is excluded from every metric

This is the single most consequential rule for model design. The spec:

> "Knocking a gene down and then reporting that gene as changed is the premise of the
> experiment rather than a prediction, and the row is worth 6–11% of the expression metric's
> range on the reference contexts."

`pds_cosine` goes further and removes **all 300 panel target genes** from the feature space,
not one at a time — otherwise a submission could spike the other 299 targets and win ranks on
nothing but the published target list.

So predicting that a knocked-down gene's own expression falls, however correct biologically,
earns exactly nothing. Signal must come from the *downstream* response.

Two further restrictions apply to the four DE metrics: a gene is tested only if its mean
expression exceeds 5 counts per million **in the control cells**, and significance is
Benjamini-Hochberg corrected at α = 0.05, computed within each perturbation after the
low-expression filter.

### Tooling

The reference bundles were built with **`cell-eval2` 0.15.0** under competition
`rule_version` 3 — not the `cell-eval` 0.8.2 installed here, whose `vcc` profile computes the
2025 three-metric panel. Local `--profile vcc` runs do not reproduce leaderboard values.

State of the leaderboard on 2026-09-14 (validation partition `vcc2026-val-1`, 943 teams):
top `score_avg` is 0.281 (Illumina AI, "PerturbationAI", 24 submissions). The whole field
sits far closer to the baseline than to a perfect score.

The leader's per-metric breakdown shows where the difficulty is concentrated —
scaled score (raw value):

| Metric | Scaled | Raw |
| --- | --- | --- |
| `pds` | 0.830 | 0.874 |
| `reach` | 0.345 | 0.384 |
| `nmae` | 0.261 | 0.845 |
| `mse` | 0.199 | 0.799 |
| `fid` | 0.027 | 0.520 |
| `jac` | 0.026 | 0.040 |

Discriminating between perturbations is largely solved; recovering the correct significant
DE gene sets (`jac`) and the correct direction of change (`fid`) is not. Because the overall
score is an unweighted mean, the two near-zero metrics cap everyone's total.

### Caveat on the local tooling

The installed `cell-eval` v0.8.2 exposes a `vcc` profile containing only three metrics
(`mae`, `discrimination_score_l1`, `overlap_at_N`) — the 2025 panel. The six 2026 metric
names above do not appear in its registry, so the 2026 panel is computed server-side.
Running `--profile vcc` locally does not reproduce the leaderboard score.

## Context from the 2025 challenge

Scale of the inaugural run: "More than 5,000 people registered across 114 countries, more
than 1,200 teams submitted results, and more than 300 teams made final submissions."

The organizers' own read on the outcome, which is the stated motivation for the 2026 format:
"The strongest models improved on important capabilities such as perturbation discrimination
and differentially expressed genes, but the results also made clear that current models are
not yet consistently replacing experimental measurements across all evaluation criteria."

That assessment matches the 2026 leaderboard breakdown above — discrimination is close to
saturated, the DE-based metrics are not.

Further reading cited by the Commentary:

- Wei, Z., et al. (2026). Benchmarking algorithms for generalizable single-cell perturbation
  response prediction. *Nat. Methods* 23, 451–464. doi:10.1038/s41592-025-02980-0
- Adduri, A.K., et al. (2025). Predicting cellular responses to perturbation across diverse
  contexts with State. bioRxiv. doi:10.1101/2025.06.26.661135 — local copy in `papers/`
- Virtual Cell Challenge 2025 Wrap-Up: Winners and Reflections.
  arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up

## Notes from this workspace

- `dataset/` contains the validation control bundle for contexts `A`, `B`, and `C`.
- `dataset/gene_names.csv` has a header row locally; the official tooling expects a
  headerless gene list, so the helper code strips it when needed.
- The first baseline is a context-mean predictor, sparsified to keep the official prep tool
  within memory.
