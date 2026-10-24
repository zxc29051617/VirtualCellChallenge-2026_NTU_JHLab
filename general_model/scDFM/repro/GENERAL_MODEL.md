# General model for unseen cell lines (from 2026-10-10 evening)

Goal: one model that, given only a new line's control cells and a target gene, predicts the response; trained on
public perturbation screens; checked by leave-one-cell-type-out (LOCO); the A/B/C leaderboard only as a check.

## Data in the common table (`vcc/contexts.py`, scorer space log1p(50k x group fraction))
| context | perturbations | source | notes |
| --- | --- | --- | --- |
| K562, HCT116, HEK293T, H1 | VCC targets + auxiliary genes | `data/stage3/stats` | existing |
| CD4_Rest / Stim8hr / Stim48hr | 520 each | `data/stage3/stats` | Marson pseudobulk |
| K562ess | 2,057 (7,940 genes measured) | Replogle essential | `vcc/stats_ext.py` |
| HepG2 | 2,393 (9,024 genes) | Nadig 2025 | |
| Jurkat | 2,393 (8,284 genes) | Nadig 2025 | |
| FengNF / FengF | genome-wide iPSC, shallow | Feng 2025 | 'unassigned' cells as reference (12 non-targeting cells only) |
| KOLF | 11,739 (282 VCC targets) | Mali lab 2026 | pending: 189 GB CSC file, ~0.3 MB/s per connection |

## Cross-context agreement (`python -m vcc.agreement`)
Median Pearson r of common-mode-removed deltas over shared perturbations (all / responsive half):
- same line, different screens (K562 GWPS vs K562 essential): 0.13 / 0.34
- different lines, essential genes (K562ess vs HepG2): 0.13 / 0.23; H1 vs K562ess 0.15 / 0.25
- CD4 states vs each other: 0.29-0.38
- VCC-type (non-essential) targets between unrelated lines: 0.01-0.06. Little trans signal transfers for them.

## CRISPRi cis effects (`python -m vcc.cis`)
Genes whose TSS lies within 2 kb of the target's TSS (mostly head-to-head partners) drop by -0.4 to -0.7 (scorer
units) in every context checked (K562, K562ess, HepG2, Jurkat, H1, CD4, HCT116), about 0.4-0.65 of the target's own
drop; 2-5 kb -0.15 to -0.3, 5-10 kb -0.1 to -0.2, 10-20 kb -0.03 to -0.07, background ~0. This is a perturbation-specific
signal that does not depend on the cell line, and the discrimination metric excludes only the target genes.

## General model v1 (`vcc/gm.py`)
ln r = OWN (target) + OWN x beta(distance bin, head-to-head) (cis, within 50 kb) + ALPHA x s_p (trans).
beta is fitted by least squares across contexts; s_p is a ridge estimate across contexts of the common-mode-removed
log ratios, rescaled by each context's own knockdown (capped at KCAP), with the own and cis genes masked.
Prediction needs only the target name and the new context's controls (applied by the usual cell construction).

LOCO pseudobulk proxy (pds-like, 0.5 = chance, target genes excluded), all perturbations of the held-out context:
| held out | own only | + cis | + cis + trans |
| --- | --- | --- | --- |
| K562 / K562ess | 0.509 / 0.504 | 0.565 / 0.576 | 0.683 / 0.895 |
| HCT116 / HEK293T | 0.500 / 0.498 | 0.574 / 0.574 | 0.623 / 0.612 |
| H1 | 0.500 | 0.628 | 0.839 |
| CD4 Rest / 8h / 48h | 0.500 | 0.608-0.617 | 0.633-0.650 |
| HepG2 / Jurkat | 0.493 / 0.507 | 0.573 / 0.575 | 0.896 / 0.874 |
On the VCC targets only, weighting and shrinkage matter little (mean 0.66-0.69); KCAP 1 beats 3 and LAM 3 beats 0.3.

Official scorer, held-out H1 (fit without H1; 188 targets; previous best = 3-line transfer x1.5, 0.2667):
| model | pds | fid | reach | jac | nmae | mse | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| own + cis only | 0.106 | 0.001 | -0.057 | 0.013 | -0.181 | 0 | -0.020 |
| full, LAM 1, KCAP 3, ALPHA 1 | 0.738 | 0.645 | 0.073 | 0.192 | 0.037 | 0 | **0.2807** |
| LAM 3, KCAP 1, ALPHA 1 / 1.5 / 2 / 3 / 4 | | | | | | | 0.157 / 0.231 / 0.275 / 0.302 / 0.304 |
| LAM 1, KCAP 3, ALPHA 1.5 | | | | | | | 0.291 |
The scorer rewards strong predicted effects (mse is floored at 0, DE calls need large shifts); the pds-like proxy
does not see this. H1 shares many essential perturbations with K562ess/HepG2/Jurkat, so it over-states transfer for
the non-essential VCC targets; held-out HCT116 (VCC targets) and KOLF are the closer checks.

### Settings sweep with the official scorer (2026-10-10 night)
avg score (from replicate) on held-out lines; previous best method in brackets.
| setting (per_celltype=False) | H1 [0.267] | HCT116, VCC targets [0.218] |
| --- | --- | --- |
| LAM 3, KCAP 1, ALPHA 1 / 2 / 3 / 4 / 5 / 6 | 0.157 / 0.275 / **0.302** / 0.304 / - / - | 0.148 / 0.228 / **0.303** / 0.301 / 0.246 / 0.272 |
| LAM 1, KCAP 3, ALPHA 1 / 1.5 / 2 / 3 | 0.281 / 0.291 / 0.291 / - | - / - / - / 0.212 |
| + confidence gate z >= 2 (ALPHA 2 / 4) | 0.279 | 0.235 |
| + generic response x1 / x3 (ALPHA 3) | 0.302 / 0.299 | 0.258 / 0.264 |
Chosen: LAM 3, KCAP 1, ALPHA 3, no gate, no generic term. Larger amplitudes raise pds/reach but hurt lfc_nmae; the
official numbers move by ~0.03 between neighbouring settings, so only differences larger than that are read.
Per-gene responsiveness (excess variance across perturbations) correlates 0.2-0.7 between contexts (`vcc.responsiveness`):
a possible gene-level prior, not used yet.

### Held-out KOLF2.1J iPSC (279 VCC targets, 100 cells each; different lab and library from every training screen)
| method | pds | fid | reach | jac | nmae | avg |
| --- | --- | --- | --- | --- | --- | --- |
| old transfer (K562+HCT116+HEK293T mean x1.5) | 0.105 | 1.019 | 0.113 | -1.868 | 0.067 | -0.094 |
| gm LAM 3 KCAP 1, ALPHA 2 / 3 / 4 | 0.088 / 0.141 / 0.173 | 0.70 / 0.98 / 1.04 | 0.16 / 0.16 / 0.18 | -1.51 / -1.74 / -1.75 | 0.13 / 0.12 / 0.09 | -0.072 / -0.057 / -0.045 |
| gm ALPHA 3 + generic x1 | 0.132 | 0.983 | 0.170 | -1.702 | 0.128 | -0.048 |
Raw (unscaled) means, gm ALPHA 3 vs the context-mean baseline: fidelity 0.47 vs 0.06, reach 0.21 vs 0.14, nmae 1.02 vs
1.11, pds 0.56 vs 0.51 -- better on everything except sig_jaccard, 0.012 vs 0.093. The baseline's significant genes
are the context's generic response, which every real perturbation shares; ours are perturbation-specific (cis/trans
with each source's common mode removed). With 100 cells per target the split-half replicate barely beats the baseline
on jaccard, so the replicate scale multiplies that gap ~20x. A 200-cell version (KOLF200) is queued.

### Overnight 2026-10-11: KOLF/Feng in training, fair comparisons (fit on every context except the held-out one)
avg score (from replicate); KOLF200 = KOLF with 200 cells per target (the 100-cell version exaggerates jaccard).
| model (LAM 3, KCAP 1, ALPHA 3) | H1 | HCT116 | KOLF200 | mean |
| --- | --- | --- | --- | --- |
| old transfer x1.5 (3 lines) | 0.267 | 0.218 | 0.135 | 0.207 |
| gm, contexts weighted equally (`v2`) | 0.352 | 0.245 | 0.163 | 0.253 |
| gm + context similarity sim_tau 5 | 0.354 | 0.301 | 0.142 | **0.266** |
| gm + sim_tau 10 | 0.328 | 0.318 | 0.132 | 0.259 |
| gm + sim_tau 20 | 0.268 | 0.296 | (KOLF100 -0.076) | |
| gm + cell-type weights (all iPSC = 1 unit) | 0.316 | 0.260 | 0.104 | 0.227 |
| gm + depth weights n0 100 | 0.324 | 0.253 | (KOLF100 -0.045 vs -0.057) | |
| gm + generic x1 | - | - | 0.158 | |
Adding KOLF/Feng helped H1 (0.302 -> 0.352, a pluripotent line) and hurt HCT116 (0.303 -> 0.245); mild similarity
weighting (exp(5 (r - 1)) on the correlation of control profiles) keeps the H1 gain and recovers HCT116.
On the challenge contexts the weights come out sensibly: A -> Jurkat and the CD4 states (~54 % of the weight together),
B and C -> H1 first (same Flex chemistry), then fairly flat.
Submission candidates (fit on all contexts): `VCC_gm_v1_a3.0_t5_{A,B,C}.npz` (per context) and `VCC_gm_v1_a3.0.npz`.

## Leaderboard check 2026-10-11 (A/B/C, anchors r4)
| submission | avg | pds | mse | nmae | fid | reach | jac | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transfer_x1p5 (10-10) | 0.1274 | 0.552 | 0 | 0.064 | -0.007 | 0.156 | -0.001 | ~581 |
| cd4_ctxw (10-10, CD4 x3 for A) | 0.1485 | 0.555 | 0 | 0.130 | 0.016 | 0.184 | 0.005 | ~490 |
| gm_v1_sim5 (entry v9jl2r0nAxts0dRe50tU) | 0.1415 | 0.515 | 0 | 0.116 | 0.025 | 0.185 | 0.007 | 538 |
| gm_v1_plain (entry FXD3pdX9SuLnunpNkYp3) | 0.1455 | 0.525 | 0 | 0.143 | 0.021 | 0.178 | 0.006 | 523 |
The LOCO gains (+0.06 mean over H1/HCT116/KOLF200) did not carry over. Raw metrics of the two gm submissions give the
r4 anchors (score = (u - b)/(a - b)): pds b 0.467 a 0.982 (ours 0.732); fid b **0.513** a 0.803 (ours 0.520);
reach b 0.083 a 0.949 (ours 0.243); jac b 0.030 a 0.414 (ours 0.033); nmae b 1.00 a 0.385 (ours 0.93); mse raw 4.3x
the baseline (floored). The context-mean baseline already gets half of the DE directions right in A/B/C (a strong
shared response), and the split-half replicate shows reproducible perturbation-specific DE sets (jac 0.41) that we do
not predict. Packaging on Brev: n2d-highmem-8, ~1 h, ~US$0.37; `vcc prep` peaked at 19.4 GB for 2.09e9 nnz.

### Post-processing checks (2026-10-11): amplitude and projection onto control PCs (`vcc/pcproj.py`)
official avg (raw normalised mse, raw pds); base = gm LAM 3 KCAP 1 ALPHA 3, all contexts
| variant | H1 | KOLF200 |
| --- | --- | --- |
| base | 0.352 (1.431, 0.898) | 0.163 (2.317, 0.573) |
| trans x1/3 | 0.198 (**0.962**, 0.852) | 0.042 (1.289, 0.573) |
| project trans onto 50 control PCs | 0.145 (0.985, 0.746) | -0.016 (1.286, 0.482) |
| 50 PCs, trans x0.5 | 0.060 (0.972, 0.690) | 0.004 (1.198, 0.515) |
| 20 PCs | 0.082 (0.996, 0.677) | 0.009 (1.265, 0.509) |
Projection loses discrimination and DE signal; smaller trans amplitude brings mse below 1 (H1) but costs far more on
the DE metrics. Per-target correlation of predicted vs true pseudobulk deltas for the VCC targets in held-out KOLF and
CD4 is ~0.01 (source agreement predicts accuracy, r 0.24-0.38 across targets, but few targets agree). Noise-corrected
proxy (`unbiased normalised MSE`): control 1.03-1.05, own+cis the same, trans x0.25-0.5 1.01-1.07, x3 1.8-3.1.
The mse member subtracts the prediction's own sampling noise only up to the reference's and only as far as the
submission's across-perturbation spread pays for it (cell-eval2 #247/#348), so spread without signal is break-even.

### Why local checks over-state the score (2026-10-11, KOLFbig = KOLF with up to 400 cells/target and 20,000 controls)
| version | avg | pds | fid (raw) | reach | jac |
| --- | --- | --- | --- | --- | --- |
| own + cis only | 0.023 | 0.087 | 0.008 (0.009) | 0.024 | 0.001 |
| shuffled trans (each target gets another's signature) | 0.136 | 0.010 | 0.676 (0.462) | 0.029 | 0.109 |
| gm (real trans) | 0.187 | 0.203 | 0.691 (0.473) | 0.075 | 0.122 |
Most of the gain over own+cis comes from making MANY DE calls at chance-level direction agreement: locally the
context-mean baseline calls almost nothing (raw fid 0.004 even at 400 cells and 20k controls), so chance-level calls
score high; on A/B/C the baseline already sits at chance (0.513). Real (vs shuffled) trans adds ~0.05, mostly pds.
`tools/adjusted_score.py` re-scores local runs with the fidelity baseline set to max(local, 0.5): KOLFbig gm 0.047,
shuffled -0.011, own+cis -0.427; H1 best 0.27, HCT116 best 0.22 -- much closer to the leaderboard level.

## Leaderboard tests 2026-10-12 (A/B/C, anchors r4)
| submission | entry | avg | pds | mse | nmae | fid | reach | jac | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gm_v1_plain (10-11, all sources) | FXD3pdX9SuLnunpNkYp3 | 0.1455 | 0.525 | 0 | 0.143 | 0.021 | 0.178 | 0.006 | 523 |
| gm_v1_cd4trans (trans from CD4 only) | NlLq5uZaZbaQW4QqEzJl | 0.0835 | 0.375 | 0 | 0.058 | -0.049 | 0.118 | -0.001 | 771 |
| gm_v1_shuffled (trans of another target) | qeClzX9dEuPoF9iwyAdi | -0.0124 | 0.055 | 0 | -0.041 | -0.078 | 0.003 | -0.012 | 1039 |
On A/B/C correctly paired trans signatures clearly beat shuffled ones (+0.158 avg, pds 0.525 vs 0.055), unlike the
12-target KOLF pilot where no reliable advantage was detected. Trans from CD4 alone is much worse than from all sources,
so the Flex-chemistry-only hypothesis is not supported; pooling sources helps.

## VIPerturb-seq K562 (10x Flex) added 2026-10-12
`genome_wide_filtered.rds` (Zenodo 18460279, CC BY 4.0, 3.6 GB Seurat v5) read without R by streaming the RDS twice
(`tools/rds_stream.py`, `tools/viperturb_stats.py`): 326,247 cells, 6,724 targets + 7,949 'NO-TARGET' controls,
18,111 VCC genes; 124 VCC targets are in this filtered object (118 with >=20 cells; ~45 cells each); own knockdown
median -0.59 (VCC targets), clear cis effect (-0.32 at <1 kb). Context `VIPK562`, grouped with K562 for LOCO and
cell-type weights. Tests scheduled for 10-13: gm_v1_celltype (per_celltype=True, built before VIPK562 existed) and
gm_v1_vip (gm_v1_plain settings + VIPK562).

## Leaderboard tests 2026-10-13
| submission | entry | avg | pds | mse | nmae | fid | reach | jac | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gm_v1_plain (reference, 10-11) | FXD3pdX9SuLnunpNkYp3 | 0.1455 | 0.525 | 0 | 0.143 | 0.021 | 0.178 | 0.006 | 523 |
| gm_v1_celltype (per_celltype=True) | 86nXxDrEII0zNdn1kx7n | 0.1419 | 0.516 | 0 | 0.158 | 0.002 | 0.175 | 0.000 | 549 |
| gm_v1_vip (plain + VIPK562) | fJKqTlWxAmOqcWim5etg | 0.1482 | 0.530 | 0 | 0.143 | 0.025 | 0.184 | 0.007 | 529 |
Equal weight per cell type: no gain (-0.004). Adding VIPK562: +0.003 — within the size of differences seen between
near-identical variants, so at most a small gain. Best overall is still cd4_ctxw 0.1485 (10-10).

## SIGnature attributions for source weights (2026-10-19): negative result
Full write-up: `repro/signature_20261013/SUMMARY.md`. SCimilarity+IG attributions of control cells as extra features for a small
source-weight model (groups A uniform / B expression / C expression+attribution / D per-context shuffled attribution / A_shuf pairing
control), 3 outer hold-outs (H1 188, HCT116 263, KOLF200 279 targets), 3 seeds, 45 official scoring runs. C did not lower raw MSE vs A or B
(H1 1.61 vs 1.45/1.52; HCT116 11.3 vs 13.5/10.4; KOLF200 2.22 vs 2.19/1.74), kept pds only at the level of B, and did not beat shuffled attribution D
consistently. Pre-set rule not met in any set -> abandon. Learned expression-only weights (B) were also inconsistent vs uniform.

## Rendering: exact group totals (2026-10-19) -- first leaderboard gain since 10-10
Same model and effects (gm_v1_vip); only the cell construction changes (`vcc/render.py`, `repro/next_20261019/`).
Screens first (`next_20261019/`): co-expression in a context's own control cells carries no perturbation signal (median
cosine 0.006); with the current signal even the best amplitude reaches only 0.94-1.00 of "no change" on the hold-outs'
mse (so amplitude tuning alone cannot win that member); HPA cell-line matching identifies A Jurkat, B HeLa, C CAL-33-like
(background only, not used for tuning).
Local hold-outs (official scorer, 3 seeds; V0 = old renderer, V1 = exact totals, V2 = bulk trans x1 / cells x3):
| set | V0 avg | V1 avg | V2 avg | raw mse V0 / V1 / V2 | raw pds V0 / V1 / V2 |
| --- | --- | --- | --- | --- | --- |
| H1 | 0.347 | 0.357 | 0.339 | 1.45 / 1.19 / 0.86 | 0.888 / 0.901 / 0.907 |
| KOLF200 | 0.154 | 0.161 | 0.118 | 2.19 / 1.14 / 1.17 | 0.550 / 0.571 / 0.568 |
| HCT116 | 0.183 | 0.255 | 0.401* | 13.5 / 3.6 / 1.25 | 0.600 / 0.647 / 0.654 |
*HCT116's mse scale is unreliable (replicate 1.23 +- 0.68), which drives V2's jump there. V2 keeps only ~50-70 % of the
cell-level change (sparse genes: total and nonzero-cell count are tied), so nmae falls; not submitted.
Leaderboard (entry j8rlnzHbqPXDRVyWXg7b): gm_v1_vip_exact **0.1608**, rank 451 (pds 0.586, mse 0, nmae 0.155, fid 0.024,
reach 0.188, jac 0.012) vs gm_v1_vip 0.1482 (pds 0.530, nmae 0.143, fid 0.025, reach 0.184, jac 0.007). The pds gain
(+0.056) matches the local direction. `tools/final_pipeline.sh` now renders with exact totals (RENDER=current = old).
Evening 2026-10-19 (`next_20261019/SUMMARY.md`): removing the top-10 shared principal components of the panel's trans part
raises pds on every hold-out but costs the DE members (V5 fails); keeping cells as V1 and only the gene totals shared-removed
(V6) passes the pre-registered rule (avg H1 -0.012, KOLF200 +0.014, HCT116 +0.224 inflated). Not yet on the leaderboard.
Leaderboard 2026-10-19 21:25 (entry via `next_20261019/upload_v6.log`): **gm_v1_vip_v6 0.1808, rank 358** -- pds 0.681,
mse 0, nmae 0.176, fid 0.026, reach 0.194, jac 0.008 (V1 0.1608: pds 0.586, nmae 0.155, fid 0.024, reach 0.188, jac 0.012).
On A/B/C the DE members did not fall (nmae rose); `tools/final_pipeline.sh` now builds V6 by default (RENDER=v6).

## Leaderboard 2026-10-20 and the G1 totals (details: next_20261019/SUMMARY.md, run.log)
| submission | avg | pds | mse | nmae | fid | reach | jac | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gm_v1_vip_v7a (V6 totals x0.5) | 0.2032 | 0.685 | 0.168 | 0.159 | 0.009 | 0.189 | 0.008 | 269 |
| gm_v1_vip_f9 (V10 cells, V8 totals x0.5) | 0.2080 | 0.753 | 0.150 | 0.153 | 0.016 | 0.166 | 0.010 | 247 |
F9 raw (leaderboard API): pds 0.840, mse 0.846, nmae 0.908, fid 0.517, reach 0.227, jac 0.034; top-40 raw mse 0.47-0.60.
With exact totals the mse correction is bounded by the across-target spread (#348): the trans scale is now set per context
so that this spread equals the expected claim (computed from the context's controls), and the totals' trans part uses
depth-weighted sources and per-target sharpening (gamma 3), which raised the global noise-corrected cosine with the held-out
truth by 21-60 % on H1 / KOLF200 / HCT116 (`final_pipeline.sh` RENDER=g1). First leaderboard test 10-21 08:01.
