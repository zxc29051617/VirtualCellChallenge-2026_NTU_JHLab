# Stage 3 — cross-cell-line results so far (2026-10-10)

## Validation design
Held-out H1 (VCC 2025, Arc, 10x Flex, thinned to 20k UMI). 190 targets with >= 10 responsive
genes, <= 80 cells each, 4,000 controls; official cell-eval2 `vcc2026` scale (`from_replicate`),
DE engine pinned to pdex. Training lines: K562 (Replogle genome-wide), HCT116 and HEK293T
(X-Atlas/Orion). In those lines the VCC targets respond weakly (<= 34/300 with >= 10 responsive
genes, vs 190/300 of H1's own targets in H1), so a held-out test on them rewards "no change".

## Baselines on held-out H1
| method | overall | pds | fid | jac | reach | nmae | mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control resample | -0.054 | -0.07 | -0.02 | 0.02 | -0.06 | -0.19 | 0 |
| 3-line mean transfer | 0.249 | 0.70 | 0.63 | 0.16 | 0.03 | -0.02 | 0 |
| transfer x1.5 | **0.267** | 0.72 | 0.67 | 0.18 | 0.03 | 0.01 | 0 |
| transfer x2 | 0.256 | 0.70 | 0.67 | 0.18 | 0.01 | -0.02 | 0 |
| transfer x3 | 0.223 | 0.70 | 0.66 | 0.18 | 0.02 | -0.22 | 0 |

## scDFM, multi-line, as in the paper (runs/h1_local_b16)
20k steps, batch 16, d_model 512, 4,565 (line, perturbation) pairs. Loss 1.37 -> 0.44.
H1 proxy flat at chance for all 20k steps (pds_like 0.50, pearson_delta 0.026).

Diagnosis (vcc/diagnose.py, vcc/diag_*.py):
1. The proxy works: the transfer baseline scores pds_like 0.939 on it.
2. **No perturbation-specific signal, even in-distribution**: on K562's 20 strongest perturbations
   (a training line), target-minus-control-token output vs truth: pearson 0.03, pds_like 0.50,
   with one-step and with euler-10 generation alike.
3. **Generated cells have no zeros**: 1-9% zeros vs 76% in real controls; 46-57% of values fall
   in (0, 0.3), where real data has none (the smallest nonzero is ~0.4 at this depth). Converted to
   counts and averaged, this smear inflates low genes and swamps effect vectors (|pred effect|
   5-14x the real one).

Likely cause of 2: each perturbation is seen ~4 times (20k x 16 / 4,565 pairs) vs ~420 in the
paper (80k x 48 / 189), its embedding is learned from scratch from those few batches only, and
most effects are weak. The model learns "what a perturbed cell looks like", not "what knocking
down gene X does".

## Two-stage model, stage 1 (vcc/effects.py), and the effect-conditioned scDFM (2026-10-10)
- Effect-conditioned scDFM (runs/h1_effect_b16): proxy at chance through 4,000 steps (pds_like
  0.51); also with bias-cancelling delta generation. The run was OOM-killed at step ~4,100 when a
  scoring job overlapped it.
- Official H1 scores (from_replicate), effects applied to H1 control cells, log2FC clipped to +-4:
  | method | overall | pds | fid | jac | reach | nmae |
  | --- | --- | --- | --- | --- | --- | --- |
  | transfer x1.5 (tools/loco.py) | 0.267 | 0.716 | 0.667 | 0.179 | 0.031 | 0.008 |
  | mean x1.5 via effect file (same thing, checks the path) | 0.266 | 0.713 | 0.665 | 0.180 | 0.022 | 0.015 |
  | per-gene transfer weights, norm matched to mean x1.5 | 0.234 | 0.668 | 0.637 | 0.169 | 0.032 | -0.103 |
  Per-gene weights improved the pseudobulk proxy on every training line in CV but lose on the
  official H1 score: the proxy does not predict the official metric well enough to select on.
- Unclipped, one effect file produced a 1.6e8-count cell (fold changes up to 2^8.8 from genes
  near-silent in a source line's controls); the scorer refuses cells above 1e6.

## 2026-10-10 evening: amplitude sweep, what limits mse, source agreement, bf16
- Amplitude / dual-amplitude sweep on held-out H1 (official scale, base = 3 lines + CD4, `vcc.effects reshape`):
  x1.5 0.268 | x0.75 0.210 | x0.5 0.150 | x0.25 0.059 | top-1000 genes x1.5 + rest x0.3 0.203 | top-300 0.174 |
  top-1000 x1.5 + rest 0 0.185. **mse is 0.000 in every variant.** Raw expression ratio N/D (1 = predict no change;
  the mean-response baseline is ~0.94): control 1.02, x0.25 0.999, x0.5 1.055, x0.75 1.165, x1.5 1.85, 3-line x1.5 2.41.
  Shrinking only drives the ratio toward 1.0; going below the baseline needs predictions that correlate with the truth.
  Optimal shrinkage gives a ratio of 1 - rho^2, so the leaderboard's mse ~0.44 implies target-specific rho ~0.65.
- Source agreement on the 229 VCC targets responsive in CD4 (mean per-target Pearson of effect vectors): K562 / HCT116 /
  HEK293T vs each other and vs CD4: 0.03-0.05 (noise-dominated: <=34/300 VCC targets respond there); CD4 conditions vs each
  other 0.26-0.35. Only CD4 carries real signal for these targets, consistent with the +0.02 from adding it.
- Effect energy is not concentrated in a few axes: generic (mean) response is only 6-7% of the energy; the top-1 component of
  the target-specific part holds ~10%, top-10 ~46-49% (H1, CD4), top-20 ~60%; HCT116/HEK293T are flat (top-10 8-10%).
- bf16 autocast of the velocity network (`--amp bf16`): 2.4-2.8x faster at a small batch with 40% less memory. Effect-
  conditioned model: loss differs 0.4%, gradient cosine 0.9994 vs fp32. Token-conditioned model: gradient cosine 0.94-0.95
  (the gene-encoder cross-attention path is less stable), so that one stays fp32. TF32 only gains 13%.

## CRISPRa TF data (Hs27 / RPE1, Nature Genetics 2025; from the team's Notion list)
80/300 VCC targets are in the Hs27 and RPE1 mean-population files, but only 726 of 10,916 guides are "active". For those
targets, per-target Pearson of the CRISPRa effect vs the CRISPRi effect: CD4 -0.010, K562 -0.010 (3-4% of targets |r| > 0.1).
Activation and knockdown effects are unrelated in sign and size here; not used as a source.

## Candidate effect files for the next submission (data/stage3/effects/)
VCC_old3cd4_x1.5 (A, B, C of submission #2 B/C), VCC_cd4w3_x1.5 (submission #2's A), VCC_cd4only_x1.5 (A: CD4 only, 290/300
targets), VCC_cd4w10_x1.5 (CD4 weight 10). The builder now uses a per-context RNG, so a change confined to context A leaves
B and C's cells identical between submissions. Plan: one controlled change per submission (A only), attribute the score delta.

## Cell-line identification tool (vcc/identify_line.py, DepMap 24Q4 bulk RNA-seq, 1,673 lines)
Spearman over genes expressed >= 5 CPM in the controls. Sanity: K562 -> K-562 (2nd/3rd hit, myeloid lines on top),
HCT116 -> HCT 116 (1st, r 0.778), HEK293T -> not in DepMap, nearest are neuro/embryonal lines. Practice contexts:
A -> T-ALL lines (SUP-T1 0.695, JURKAT 0.689), B -> HeLa (0.613, clear margin), C -> head-and-neck / squamous carcinoma
lines (CAL 27 0.621). Kept as a generic tool for any new context (the final test is D/E/F, released 2026-10-22);
per the user, do not spend more effort tuning to A/B/C specifically.

## Retrospective and change of direction (2026-10-10 evening)
- Stopped `runs/h1ho_token` at ~2,850 steps (proxy pds_like 0.500 / 0.486 / 0.495 at it 0 / 1k / 2k, i.e. chance) and
  cancelled the queued effect-conditioned run. This is the third scDFM configuration with no perturbation signal.
- What went wrong:
  1. Tuning to the validation contexts (CD4 weight x3 for A, identifying A/B/C, a Jurkat download justified by "A is
     T-cell-like"). The final ranking uses D/E/F, so nothing specific to A/B/C carries over.
  2. Both submissions are statistical transfer of measured effects, not a trained model.
  3. scDFM variants kept being launched although the data could not support them: <= 34/300 targets respond in the
     training lines, and our budget gives ~4 exposures per perturbation versus ~530 in the paper.
  4. Many method comparisons were made on held-out H1, whose score does not track the leaderboard (0.267 vs 0.127).
  5. Too many side threads (CRISPRa, line identification, AMP benchmark).
- Epithelial CRISPRi search: no public epithelial CRISPRi Perturb-seq set covers more than 9/300 targets. The best is
  Jiang 2025 (Satija lab; A549/MCF7/HT29/BxPC3 plus HAP1/K562; cytokine-stimulated; Zenodo 14518762, CC BY 4.0; ELK1,
  FOXO4, IFNAR2, IFNGR2, MED15, MTF1, SMARCA5, STAT6, ZNF22). Others: Sunshine 2023 Calu-3 (6/300, SARS-CoV-2 infected),
  gastric dcPerturb-seq (Cas9 knockout, 9/300, raw reads only). None is worth pulling now.
- New direction (user's decision): survey published virtual-cell models and the datasets they train on, collect
  those datasets, train one general model that works on any unseen line from its control cells, and use the A/B/C
  leaderboard only as a check.
