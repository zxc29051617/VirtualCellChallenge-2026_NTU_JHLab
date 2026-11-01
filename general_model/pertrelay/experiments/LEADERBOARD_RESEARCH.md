# What the 2026 leaderboard says (2026-10-10, 1,419 teams)

Our first submission, `transfer_x1p5` (3-line mean transfer x1.5): **0.127, rank ~580**
(pds 0.55, mse 0, nmae 0.06, fid -0.01, reach 0.16, jac 0.00). Same method scored 0.267 on the
held-out-H1 practice set: H1 over-predicts; use the leaderboard (2/day) for decisions.

## Score components, medians
| ranks | pds | mse | nmae | fid | reach | jac |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (Illumina AI, 0.454) | 0.89 | 0.77 | 0.40 | 0.14 | 0.45 | 0.06 |
| 1-10 | 0.81 | 0.44 | 0.26 | 0.05 | 0.28 | 0.02 |
| 11-50 | 0.79 | 0.40 | 0.22 | 0.04 | 0.24 | 0.01 |
| 51-200 | 0.75 | 0.29 | 0.15 | 0.02 | 0.22 | 0.00 |
| ours | 0.55 | 0.00 | 0.06 | -0.01 | 0.16 | 0.00 |
The gap is pds (+0.24) and mse (+0.40); fid/jac are ~0 for nearly everyone.

## Keywords in model names (median / best overall score)
CD4 T cells 0.138/0.289 · "atlas" (multi-source) 0.140/0.289 · iPSC/KOLF 0.105/0.289 ·
X-Atlas 0.099/0.289 · K562/Replogle 0.037/0.289 · flow/diffusion -0.125/0.271 ·
STATE -0.099/0.159 · foundation-model embeddings -0.045/0.163.
Generative and foundation models do worst; multi-source statistical transfer does best.

## Techniques named in descriptions of high entries
- many sources fused: CD4 T-cell donors (Marson 2025), X-Atlas (calibrated), KOLF2.1J iPSC,
  Feng HipSci iPSC, H1, Jurkat; Wiener-filter / agreement shrinkage, reliability n/(n+100)
- **separate magnitudes for the bulk shift and the DE signal** ("dual-head ... log2FC/bulk
  amplitudes 1.5/0.50", "pseudobulk 0.5x", "bulk shifts shrunk entrywise toward the controls")
  -- the bulk shift drives mse/pds, the per-cell distribution drives the DE metrics
- remove the across-target shared component; tilt away from the cross-cell-type generic response
- per-context source weights, e.g. CD4 weight doubled for the T-lymphoid context
- exact integer group totals when emitting counts
- public code: github.com/kaipengm2/Virtual-Cell-Challenge-2026 ("AtlasShift", 0.180)

## Contexts (marker genes in the released controls, CPM)
- **A is a T-cell leukaemia line**: CD3E 759, CD3D 943, LCK 429, TRBC1 3,358, TAL1 145
  (TAL1+ T-ALL, Jurkat-like)
- B: KRT8 2,533, KRT18 848, VIM 6,132, COL1A1 369, no EPCAM/CDH1 -- epithelial origin, mesenchymal
- C: EPCAM 312, CDH1 128, KRT8/18 high, SOX2 143, VIM 2 -- classic epithelial
- whole-profile correlation with K562/HCT116/HEK293T/H1 controls is 0.35-0.64 and does not
  identify B or C (H1, same Flex chemistry, is closest for all three)

## What other teams use (survey, 2026-10-10; sources in the session's research report)
- 1,423 teams; 7 score >= 0.30, 92 >= 0.25, median 0.089. None of the top 17 has published a method. Illumina AI
  ("iCell", 0.4536; largest margins on mse 0.77 vs 0.46 and reach 0.45 vs 0.29) most likely trains on Illumina's private
  Billion Cell Atlas (CRISPR on/off of 20k genes x 250 contexts, alliance access only). Xaira "X-team" (0.317) owns
  X-Atlas/Pisces (not yet public).
- Almost every disclosed 2026 entry is measured-response transfer (fuse public CRISPRi fold changes per target,
  re-anchor on each context's controls, careful count generation). Best disclosed: "transfer-protocol-v37a", 0.2888
  (K562 GWPS, X-Atlas HCT116/HEK293T, CD4, KOLF2.1J PC-cleaned, HipSci iPSC, VIPerturb-seq, LINCS; down-weighting along
  control principal axes, removing the generic cross-line response). Pure deep models (State fine-tunes, PIE zero-shot,
  CellFlow, scGPT) score near or below 0.
- 2025 winners (BM_xTVC 0.340, XLearning 0.328, Outlier 0.326) all used responses of the same targets in other lines as
  pseudobulk / DE features, with learned models on top (hybrids beat pure deep nets). Altos PRiMeFlow (2025 Generalist
  Prize, arXiv 2604.13986, recommended by Arc) trained a flow-matching U-Net on ~7M cells from H1, Replogle K562/RPE1,
  Nadig HepG2/Jurkat, sci-Plex-GxE, Jiang, Feng iPSC plus internal screens.
- Lessons: knockdown depth differs a lot between screens (median own-gene log2FC: X-Atlas -0.39/-0.33, K562 -1.33,
  RPE1 -1.85, Nadig ~-1.7, H1 -2.7; VCC 2026 says all targets > 80 % knockdown), so rescale per screen; averaging sources
  shrinks magnitudes ("norm restoration" helps); cross-line agreement is capped (~0.47 noise-corrected HepG2 vs Jurkat);
  count generation bugs can cost more than biology.
- Target coverage: K562 GWPS 272/300, X-Atlas 300/300, CD4 291/300; essential-gene screens (Nadig HepG2/Jurkat,
  Replogle essential/RPE1) cover 0/300. Genome-scale sets we do not have yet: KOLF2.1J iPSC CRISPRi (figshare+ 27261219,
  46.7 GB strong-perturbation subset), VIPerturb-seq (Zenodo 18460279, 17.3 GB). Download list for 21 source contexts:
  github.com/mibohl/VCC2026.
- Rules (revised 2026-09-16): literature/lab results may not be submitted or used to revise model predictions; training
  on them is allowed; finalists disclose datasets and non-learned/blended components. Pure lookup transfer could be
  challenged; no Arc ruling found.
