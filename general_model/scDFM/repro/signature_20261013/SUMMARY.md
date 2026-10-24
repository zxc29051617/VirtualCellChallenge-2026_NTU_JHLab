# SIGnature attributions as features for source weights — local hold-out experiment (2026-10-19)

**Question.** Do SIGnature attribution features (SCimilarity encoder, Integrated Gradients, computed on control cells) help a small, regularised model learn better source weights for the general model, beyond expression-profile features? Attribution is only used to compare contexts; it is never used as a fold change, a direction or an amplitude. Own/cis terms, trans amplitude (3), shrinkage (lambda 3), cell generation and scoring are identical across groups.

**Result: no. SIGnature adds nothing detectable over expression features, and the shuffled-attribution control does as well or better. Recommendation: do not invest further in this direction.**

## What was run
- Model: SCimilarity encoder (Zenodo 10.5281/zenodo.15729925, md5 7c572b62787ffa98c027b80bca270cdb, 116 MB), SIGnature repo c1546cb; official preprocessing (`SCimilarityWrapper.preprocess_adata`: model gene order, normalize_total 1e4, log1p) on raw counts keyed by gene symbol (17,457 of the 18,533 challenge genes exist in the model); IG with zero baseline, 50 steps, target_sum 1e3, captum 0.8.0. 120-320 cells/s on the RTX 4060 Ti.
- Controls: 2,000 fixed control cells (seeded) for each of 9 contexts with single-cell data (H1, HCT116, KOLF from the hold-out NTC sets; HEK293T, K562, K562ess, HepG2, Jurkat, VIPK562 from their control cells). Pseudobulk-only contexts (CD4 x3, FengNF, FengF) have no attribution: their attribution features are missing and set to neutral (0 after z-scoring) in C and D; they still enter as sources with expression-feature weights. Nothing pseudobulk was treated as single-cell.
- Groups (identical sources, coverage handling, complexity, control indices, scoring): **A** uniform weights (existing model); **B** weights from 4 expression features (control-profile Pearson, centred Pearson, coverage, source knockdown quality); **C** B + 3 attribution features (Spearman, centred Pearson of log attribution, top-300 Jaccard); **D** as C but each context's attribution vector permuted independently across genes (own seed per context); **A_shuf** = A with each target paired to another target's trans signature (existing control).
- Weight model: w = N*softmax(F.theta), theta=0 reproduces A; fitted by minimising -mean cosine between a unit's own target responses and the weighted trans signature of the other contexts (4,000 highest-variance genes, targets with >=50 cells in the unit and >=3 sources) + kappa*|theta|^2; kappa in {1e-4,1e-3,1e-2} chosen by inner leave-one-exclusion-group-out CV. Outer hold-out never entered fitting or selection. The cache-based effect construction was checked against `vcc.gm.GeneralModel` (max |diff| 8e-7 log2FC; `verify_equivalence.json`).
- Outer hold-outs (training excludes the test screen and same-screen/related backgrounds): **H1** (188 targets; 14,991 real cells + 4,000 NTC; trained on 13 contexts), **HCT116** (263 targets; 48,486 real cells; excludes HCT116 and HEK293T = same X-Atlas screen; 12 contexts), **KOLF200** (279 targets, 51,785 real cells; excludes KOLF, FengNF, FengF; 11 contexts). Target lists are the existing quality-filtered real.json lists (>=50 real cells), not chosen by model performance. Predictions: 200 cells per target, 3 generation seeds, per-target independent random streams, control indices shared across groups, original-library generation formula. Scorer: cell-eval2 0.18.0, vcc2026 preset, de.backend=pdex, existing fixed bundles (digests in manifest.json). 45 scoring runs, serial.
- Deviation: KOLF200 (200 real cells/target) was used instead of KOLFbig (400 cells, 20k controls) to fit the compute budget.

## Results
Official-scale average and panel raw values (mean over 3 seeds; per-seed values in results_per_run.csv). The official MSE member is a panel ratio; it is shown raw per run and never rebuilt from per-target numbers.

| set | group | avg (official scale) | pds | nmae | fid | reach | jac | panel raw mse_norm (seed 0/1/2) | panel raw pds |
|---|---|---|---|---|---|---|---|---|---|
| H1 | A | 0.3466 | 0.805 | 0.191 | 0.702 | 0.171 | 0.212 | 1.448 / 1.451 / 1.450 | 0.8878 |
| H1 | B | 0.3513 | 0.811 | 0.205 | 0.702 | 0.181 | 0.209 | 1.518 / 1.521 / 1.520 | 0.8908 |
| H1 | C | 0.3380 | 0.807 | 0.170 | 0.693 | 0.155 | 0.203 | 1.606 / 1.609 / 1.607 | 0.8891 |
| H1 | D | 0.3509 | 0.805 | 0.203 | 0.702 | 0.187 | 0.209 | 1.478 / 1.481 / 1.480 | 0.8880 |
| H1 | A_shuf | 0.0994 | 0.110 | -0.209 | 0.601 | -0.060 | 0.155 | 1.849 / 1.852 / 1.853 | 0.5386 |
| HCT116 | A | 0.1830 | 0.524 | 0.301 | 0.811 | 0.603 | -1.140 | 13.672 / 13.526 / 13.342 | 0.6004 |
| HCT116 | B | 0.2001 | 0.544 | 0.319 | 0.823 | 0.636 | -1.122 | 10.519 / 10.373 / 10.197 | 0.6031 |
| HCT116 | C | 0.1982 | 0.537 | 0.324 | 0.819 | 0.642 | -1.132 | 11.461 / 11.323 / 11.128 | 0.6021 |
| HCT116 | D | 0.2103 | 0.609 | 0.318 | 0.814 | 0.654 | -1.134 | 12.986 / 12.849 / 12.675 | 0.6116 |
| HCT116 | A_shuf | -0.1351 | -0.383 | -0.067 | 0.777 | -0.001 | -1.137 | 14.582 / 14.506 / 14.195 | 0.4807 |
| KOLF200 | A | 0.1544 | 0.157 | -0.046 | 0.660 | 0.074 | 0.081 | 2.207 / 2.171 / 2.189 | 0.5496 |
| KOLF200 | B | 0.1198 | 0.160 | -0.032 | 0.471 | 0.076 | 0.043 | 1.757 / 1.725 / 1.740 | 0.5506 |
| KOLF200 | C | 0.1392 | 0.170 | -0.030 | 0.611 | 0.061 | 0.023 | 2.242 / 2.208 / 2.225 | 0.5545 |
| KOLF200 | D | 0.1124 | 0.151 | -0.033 | 0.443 | 0.074 | 0.038 | 1.760 / 1.727 / 1.742 | 0.5471 |
| KOLF200 | A_shuf | 0.1022 | 0.000 | -0.100 | 0.616 | 0.022 | 0.074 | 2.257 / 2.228 / 2.232 | 0.4887 |

Target-level paired differences on pds (seeds averaged first; bootstrap over targets, 10,000 draws; seeds are not independent replicates):

| set | comparison | mean diff | 95% CI | targets where first is better |
|---|---|---|---|---|
| H1 | C-B | -0.0017 | [-0.0091, +0.0059] | 48/188 |
| H1 | C-A | +0.0013 | [-0.0085, +0.0119] | 53/188 |
| H1 | C-D | +0.0011 | [-0.0079, +0.0108] | 51/188 |
| H1 | B-A | +0.0030 | [-0.0018, +0.0078] | 62/188 |
| H1 | A-A_shuf | +0.3493 | [+0.3063, +0.3930] | 168/188 |
| HCT116 | C-B | -0.0009 | [-0.0063, +0.0045] | 131/263 |
| HCT116 | C-A | +0.0017 | [-0.0041, +0.0075] | 125/263 |
| HCT116 | C-D | -0.0095 | [-0.0162, -0.0028] | 111/263 |
| HCT116 | B-A | +0.0026 | [-0.0074, +0.0127] | 122/263 |
| HCT116 | A-A_shuf | +0.1197 | [+0.0889, +0.1517] | 184/263 |
| KOLF200 | C-B | +0.0039 | [-0.0040, +0.0119] | 144/279 |
| KOLF200 | C-A | +0.0049 | [-0.0071, +0.0172] | 140/279 |
| KOLF200 | C-D | +0.0074 | [-0.0021, +0.0169] | 150/279 |
| KOLF200 | B-A | +0.0010 | [-0.0073, +0.0093] | 138/279 |
| KOLF200 | A-A_shuf | +0.0609 | [+0.0239, +0.0985] | 156/279 |

Inner cross-validated objective (mean cosine, higher = better; uniform weights vs learned; independent of the outer test):

| outer | uniform | B | C | D |
|---|---|---|---|---|
| H1 | 0.0554 | 0.0634 | 0.0636 | 0.0631 |
| HCT116 | 0.0827 | 0.0944 | 0.0915 | 0.0941 |
| KOLF200 | 0.0787 | 0.0849 | 0.0871 | 0.0842 |

## Pre-set decision rule (needed in >= 2 of 3 sets): (i) raw MSE of C below A and B in all seeds; (ii) pds preserved vs B; (iii) C beats D (raw MSE lower in all seeds and pds higher).
| set | (i) mse | (ii) pds | (iii) beats D |
|---|---|---|---|
| H1 | False | True | False |
| HCT116 | False | True | False |
| KOLF200 | False | True | False |

**Verdict: SIGnature_adds_value = False** (criterion (i) and (iii) fail in all three sets).

## Reading
- Raw MSE: C is higher than B in every set and every seed (H1 1.61 vs 1.52; HCT116 11.3 vs 10.4; KOLF200 2.22 vs 1.74) and than A in two of three; it is lower than D only on HCT116. Attribution features did not lower MSE.
- PDS: C - B differences are -0.002 to +0.004, all with CIs spanning 0; C vs D on HCT116 is significantly worse (-0.0095). No hold-out shows C better than the shuffled-attribution control on both metrics.
- Learned expression-only weights (B) are not reliably better than uniform either: B lowers raw MSE vs A on HCT116 (13.5 -> 10.4) and KOLF200 (2.19 -> 1.74) but raises it on H1 (1.45 -> 1.52), and the official average falls on KOLF200 (0.154 -> 0.120) because fidelity and jaccard drop. Inner CV improvements (cosine +0.006 to +0.012) did not transfer consistently.
- Correct perturbation pairing beats the pairing shuffle in all three sets (A vs A_shuf pds +0.35, +0.12, +0.06), consistent with earlier results.
- Neither learned weighting changes the picture on the raw MSE level (still >= 1.45 on H1, >> 1 on HCT116): source weighting alone cannot fix the MSE gap.

## Limits
- Three hold-outs (188/263/279 targets), 3 generation seeds (not independent), one weight-model family, one inner objective (direction/cosine, not MSE), 7 features, one foundation model and one attribution method; attribution available for 9 of 14 contexts and from 2,000 control cells. A different model, objective or feature set was not tried by design (no search after viewing outer results).
- The 0.5 fidelity calibration column in results_per_run.csv (`fid_scaled_heuristic_b05_NOT_OFFICIAL`) is a heuristic and is not part of any score.
- Resources were shared with an unrelated job early on; one waiting command was stopped by the system under memory pressure (the scoring itself continued). Download volume was well under 10 GB (116 MB weights; a cancelled attempt to pip-install a newer torch/CUDA stack was interrupted by connection resets and replaced by reusing the local torch 2.7.1+cu126; the exact byte count was not measured).
- Compute: ~3.7 h wall clock including ~0.7 h before a pause. No leaderboard submission; nothing existing was modified.

## Files
manifest.json, status.json, run.log, RUN.md (re-run commands), verdict.json, results_per_run.csv, results_per_target.csv, bootstrap.csv, inner_cv.csv, weights_*.json (weights, theta, kappa, standardised features), effects_manifest.json, verify_equivalence.json, attr/*_stats.json, controls/availability.json, scores/<set>_<group>_seed<k>/, and the scripts (sample_controls, compute_attributions, cache_contexts, sigexp, weights, export_effects, evaluate_runner, analyze). Large binaries (env, models, controls, cache, attr arrays, effects) are git-ignored.
