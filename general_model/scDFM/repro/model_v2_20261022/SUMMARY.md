# Trained response model (model_v2), started 2026-10-22

Goal (user, 10-22): a trained model that predicts the response of a cell type it has never seen perturbed, judged by D/E/F.

## Benchmark (bench.py, fixed before training)
Leave one cell type out (7 folds; same-screen siblings hidden together), up to 500 targets per hidden context, trans genes only.
dir = noise-corrected cosine, pds = cosine-rank discrimination (0.5 chance), skill = noise-corrected error vs no change.

| method | dir | pds | skill |
| --- | --- | --- | --- |
| zero | +0.0000 | 0.5000 | +0.0000 |
| consensus | +0.1013 | 0.7417 | -0.0849 |
| consensus_d200 | +0.1111 | 0.7382 | -0.0011 |
| adaptnet_s0 | +0.1488 | 0.7469 | +0.0202 |
| adaptnet_s1 | +0.1535 | 0.7435 | +0.0216 |

Per fold (consensus_d200 -> AdaptNet seed 0):

| fold | dir | pds | skill |
| --- | --- | --- | --- |
| K562 | +0.096 -> +0.153 | 0.711 -> 0.715 | -0.005 -> +0.029 |
| XAtlas | +0.035 -> +0.070 | 0.597 -> 0.636 | -0.047 -> -0.009 |
| H1 | +0.178 -> +0.202 | 0.833 -> 0.842 | +0.013 -> +0.036 |
| CD4 | +0.047 -> +0.070 | 0.635 -> 0.625 | -0.026 -> -0.004 |
| HepG2 | +0.186 -> +0.234 | 0.888 -> 0.903 | +0.033 -> +0.044 |
| Jurkat | +0.196 -> +0.266 | 0.870 -> 0.878 | +0.038 -> +0.064 |
| iPSC | +0.038 -> +0.048 | 0.634 -> 0.629 | -0.014 -> -0.019 |

## What did not work
- ResponseNet (free 18,533 x 64 gene embeddings, set encoder, decoder): training loss below the consensus, but on the hidden
  cell type it overfits (H1 pds 0.76 after 1k steps, 0.69 after 3k; contrastive term 0.74) vs consensus 0.83.
- Projecting the consensus on the top-k training programs: dir/skill up, pds down (k 200: 0.127 / 0.687 / +0.014); mixing a
  fixed share of the detail back moves along the same trade-off, never better on all three.

## AdaptNet (adaptnet.py)
Fixed top-200 programs per fold; learned per-source weights, per-target detail weight beta and size alpha, and
program gains from the new cell's control profile (~8.4k trainable weights, 1,500 steps, ~1 min per fold on the RTX 4060 Ti).
Loss: MSE relative to 0.4 x consensus plus a within-context contrastive term (pds). Two seeds agree.

## Next
1. Train on all 14 contexts, predict D/E/F (own + cis from the existing rules).
2. Mock exam: H1 2025 (400 cells per target) hidden, official scorer, consensus pipeline vs AdaptNet pipeline.
3. Replace gm_final_v1 only on a clear win (ask the user first).

## 10-22 late: the pipeline's processing changes the comparison

Mock exam subset s0 (official scorer, H1): AdaptNet (learned beta ~0.3) lost all 6 members to the current pipeline.
Cause: the totals processing (panel shared removal k 10, control axes, gamma 3, target weights) was tuned on the
consensus (H1 pds 0.833 -> 0.893); it removes AdaptNet's target signal, which sits in the top programs (pds 0.842 -> 0.862).
AdaptNet learned to shrink the per-gene detail (beta ~0.3) for MSE, but target identity (pds, DE) lives in that detail.
Training with the query context's shared components removed from the loss (orth 10) did not help.

| method (7 folds) | dir | pds | skill |
| --- | --- | --- | --- |
| consensus_d200 | +0.1111 | 0.7382 | -0.0011 |
| consensus_d200_shared10 | +0.0676 | 0.7459 | -0.0082 |
| adaptnet_s0 | +0.1488 | 0.7469 | +0.0202 |
| adapt_b1_raw | +0.1309 | 0.7623 | +0.0156 |
| adapt_b1_shared10 | +0.0658 | 0.7382 | +0.0029 |

AdaptNet with the full detail (beta fixed at 1) and NO shared removal beats the consensus with the pipeline's shared
removal on all three (dir 0.131 vs 0.068, pds 0.762 vs 0.746, skill +0.016 vs -0.008). On H1 alone (the set the processing
was tuned on) the processed consensus keeps the higher pds (0.890 vs 0.858).
Mock exam rerun (all 5 subsets): A = current pipeline; B = AdaptNet beta 1 with minimal processing (SHAREDK=0 GAMMA=1,
no target/entry weights, no cell agreement factor; control axes, budget = claim, top-2 % rendering kept).

## 10-23 00:07: mock exam result (official scorer, H1 hidden, 5 subsets / 298 targets, scaled from_replicate)
| version | mean | pds | mse | nmae | fid | reach | jac | subsets won |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A current pipeline (= gm_final_v1 config) | 0.2723 | 0.830 | 0.041 | -0.032 | 0.540 | 0.061 | 0.195 | - |
| B AdaptNet beta 1, minimal processing (totals + cells) | 0.2741 | 0.858 | 0.092 | -0.041 | 0.494 | 0.047 | 0.195 | 4/5 |
| B + cell agreement factor (stopped after 3 subsets) | worse DE than B | | | | | | | |
| **hybrid: AdaptNet totals + current cells** | **0.2910** | 0.858 | 0.092 | -0.026 | 0.548 | 0.073 | 0.201 | **5/5** |
The trained model's smooth responses over-call DE genes when used for the cells (fid down); used for the gene totals only
(mean responses: pds, mse) and rendered with the current cells, every member improves on average (+0.0186 mean). H1 is the
set the current processing was tuned on, so this is a conservative test for the trained model.
Next: AdaptNet (beta 1) trained on all 14 contexts, per-context totals for D/E/F (the model uses each context's controls),
cells as gm_final_v1; ask the user before replacing the final entry.

## 10-23: a model for ANY gene (user's goal: predict anything; D/E/F is one exam)
Leave-genes-out benchmark (bench_genes.py): 5 gene folds, all knockdown data of a fold's genes removed.
S1 = unseen gene in seen cell types, S2 = unseen gene in a hidden cell type (H1 / XAtlas / iPSC groups).
Gene knowledge (gene_knowledge.py): STRING v12 (777,843 links >= 400, 18,246 genes), GO (10,142 terms, 16,649 genes).

| method | S1 dir | S1 pds | S1 skill | S2 dir | S2 pds | S2 skill |
| --- | --- | --- | --- | --- | --- | --- |
| kNN co-expression | 0.040 | 0.550 | -0.084 | 0.008 | 0.532 | -0.027 |
| kNN response loadings | 0.049 | 0.558 | -0.076 | 0.007 | 0.522 | -0.028 |
| ridge (response + co-expression) | 0.063 | 0.550 | -0.033 | 0.009 | 0.523 | -0.009 |
| kNN GO | 0.099 | 0.587 | -0.051 | 0.017 | 0.541 | -0.030 |
| kNN STRING | 0.109 | 0.605 | -0.053 | 0.024 | 0.555 | -0.032 |
| kNN STRING + GO | 0.115 | 0.605 | -0.048 | 0.024 | 0.558 | -0.032 |
| ridge (all four representations) | 0.118 | 0.584 | -0.031 | 0.024 | 0.550 | -0.020 |
| blend (kNN STRING + ridge) | 0.130 | 0.600 | +0.004 | 0.027 | 0.557 | -0.016 |

GeneNet (genenet.py): one network, gated by whether the gene has measurements: measured genes = own data in other cell
types (AdaptNet path) + STRING/GO neighbours; never-perturbed genes = neighbours (learned attention over the 32 most
similar genes) + gene-description path. Lesson: the description path memorises training genes, so it must not serve
measured genes (it dragged H1 seen-gene pds 0.83 -> 0.77). Quick check on H1: seen genes dir 0.209 pds 0.832 skill
+0.043 (consensus 0.179 / 0.833 / +0.013); never-perturbed genes (fold 0) 0.043 / 0.610 (blend 0.022 / 0.587).
Full benchmarks running (run_genenet.py).

## 10-23 morning: GeneNet4 (the general model) and ELEM-1200RS
GeneNet4 = one model with two experts: genes with measurements in other cell types -> AdaptNet (beta 1, frozen after its
own training); never-perturbed genes -> knowledge branch (attention over the 32 STRING+GO neighbours' consensus responses +
gene-description path), trained only on episodes that hide the query gene's own data, with its size calibrated on 20 % of
the training genes held out. Bug fixed on the way: measured genes off the 18,533-gene axis got zero (now AdaptNet).

| model | hidden cell types (7 folds) dir / pds / skill | never-perturbed genes in hidden cell types (S2) dir / pds / skill |
| --- | --- | --- |
| consensus (current pipeline input) | 0.111 / 0.738 / -0.001 | - |
| blend (kNN STRING + ridge) | - | 0.027 / 0.557 / -0.016 |
| GeneNet (shared gates, no calibration) | 0.135 / 0.731 / +0.016 | 0.024 / 0.572 / -0.041 |
| GeneNet2 (own data only for measured genes, calibrated) | 0.122 / 0.750 / +0.012 | 0.025 / 0.572 / -0.006 |
| **GeneNet4** | **0.131 / 0.762 / +0.016** | **0.025 / 0.572 / -0.007** |
| GeneNet4 + ELEM-1200RS (same test sets) | 0.124 / 0.762 / +0.013 | 0.019 / 0.549 / -0.018 |

ELEM-1200RS hurts both benchmarks on identical test sets (bulk CRISPR-KO, weak references: batch medians / few wild types,
HAP1 near-haploid) -> not used. Licence attribution only applies if it is used.

## 10-23: more gene knowledge vs more experiments
- CollecTRI TF->target prior (720 TFs): direction ~0 on its own; on never-perturbed TFs in hidden cell types pds 0.512 -> 0.531,
  elsewhere no gain. UniProt ProtT5 protein embeddings (17,899 genes): kNN 0.062 / 0.573 (weaker than STRING / GO), adds
  nothing combined (kNN STRING+GO+protein 0.116 / 0.606; ridge + protein 0.120 / 0.588). The knowledge route is saturated.
- RPE1 (Replogle 2022 essential CRISPRi, from the kept bulk summary: 2,204 targets, 14,190 controls) as an extra
  training-only cell type, identical test sets:
  | | hidden cell types dir / pds / skill | never-perturbed genes (S2) |
  | --- | --- | --- |
  | GeneNet4 | 0.131 / 0.762 / +0.016 | 0.025 / 0.572 / -0.007 |
  | GeneNet4 + RPE1 | **0.152 / 0.766 / +0.025** | 0.025 / 0.571 / -0.010 |
  Gains concentrate where targets overlap essential genes (HepG2 dir 0.205 -> 0.303, Jurkat 0.238 -> 0.257, K562, H1);
  RPE1 covers 0 of the 300 D/E/F targets, so the exam is barely affected. Kept in the general model.
- Next: more single-cell perturbation data in new cell types (scPerturb v1.4, Zenodo 13350497): TianKampmann 2019/2021
  CRISPRi in iPSC-derived neurons, FrangiehIzar2021 melanoma (CRISPR KO).
- Neurons (scPerturb TianKampmann2021 CRISPRi, iPSC-induced neurons: 184 targets, 437 controls, knockdown median -0.30;
  the 2019 day-7 set has only 26 single-perturbation targets without visible knockdown, unused) on top of RPE1: hidden cell
  types 0.152 / 0.766 / +0.025 -> 0.152 / 0.768 / +0.023, i.e. no change on average (H1 dir +0.020 and iPSC pds +0.008, but
  HepG2 dir -0.016, XAtlas pds -0.010). Not in the default model.

## 10-23 midday: why are the DE members weak? (de_diag.py, de_sim.py, de_potential.py on mock H1 s0)
mock_exam.py now takes WRITE_DE=1 (scorer DE tables) and KEEP=1 (keep the predicted cells). de_sim.py scores a
hypothetical prediction DE table with the scorer's own metric functions (reproduces the real run exactly: fid 0.5527,
reach 0.1699, jac 0.1183, nmae 0.9395), so call strategies can be compared without building cells.
- Reference calls per target vary hugely (median 546; 0 to 7,518; quartile medians 8 / 214 / 1,349 / 4,083); we call
  ~3,000 for every target (corr of log counts 0.40). Sign precision of our calls 0.565 (weak targets 0.505, strong 0.66).
- Upper bounds with the current ranking: count = the reference count per target (oracle) fid +0.051 / jac +0.054 scaled;
  TRUE signs on our calls fid 0.98, reach 0.64 -> the sign accuracy of the gene-level prediction is the bottleneck, not
  the rendering. Calling fewer genes for predicted-weak targets loses (under-calling crashes fid); ~3,000 is robust.
- Sign precision at the top-k genes per target: our cells' p-values 0.69 @10, 0.65 @100, 0.60 @1000; AdaptNet totals
  0.61 @10. Strong targets: top 0.1 % 0.87, top 1 % 0.80; weak targets ~0.51-0.58 even at the top.
- Common response: 69 % of reference-significant genes share the sign of the real panel mean (the context-mean baseline
  knows it; our totals have the panel mean removed). Predicting it: training contexts' common modes correlate 0.22 with
  H1's panel mean (leave-one-group-out over 15 contexts mean 0.27; essential screens 0.55-0.59, non-essential panels
  0.0-0.28; D/E/F targets are non-essential). Adding it (or genes' habitual direction from training) gives fid/reach/jac
  +0.01-0.025 scaled each in the simulator.
- D/E/F control profiles: D myeloid (LYZ, SPI1, CEBPA, MPO, ELANE), E mesenchymal (VIM, COL1A1), F epithelial-mesenchymal
  (KRT8/18, VIM, COL1A1); nearest training contexts only CD4/K562 for D (corr 0.26-0.35).
- Oracle tests on mock s0 (official scorer, TRUE H1 panel mean added; COMMON/COMMONSCALE/CELLCOMMON options in
  tools/final_pipeline.sh + make_bulk_effects.py --common): in the totals it kills pds (0.812 -> 0.348) while helping
  mse/nmae/fid/reach (overall 0.2906 / 0.2589 vs 0.2952); in the cells only (x2) +0.030 overall (0.3247: nmae +0.046,
  fid +0.054, reach +0.056, jac +0.021, pds/mse unchanged). But the common response of a new cell type is not
  predictable from our data: leave-one-group-out corr 0.27 (non-essential panels -0.11..+0.22); projecting it on the
  new cell type's own control-cell axes or propagating it through their covariance does not help on average (0.27 ->
  0.15..0.28), although 10-65 % of each cell type's common response lies in its top-20 control axes.
- Cell-cycle composition (state_shift.py): reweighting H1 controls to each target's S / G2M scores explains almost
  nothing of the responses (mean corr 0.04, max 0.24) -> no composition-based renderer.
- Sign benchmark on hidden cell types (bench_sign.py; truth = full ln ratio, ranking |p| x sqrt(expression)):
  | method | all genes @10 / @100 / @1000 | significant genes @10 / @100 / @1000 |
  | --- | --- | --- |
  | consensus | 0.611 / 0.588 / 0.559 | 0.635 / 0.612 / 0.524 |
  | consensus x agreement (cells input) | 0.621 / 0.592 / 0.560 | 0.646 / 0.615 / 0.525 |
  | AdaptNet | 0.612 / 0.589 / 0.562 | 0.639 / 0.614 / 0.555 |
  | AdaptNet x agreement | 0.616 / 0.590 / 0.562 | 0.643 / 0.615 / 0.555 |
  | AdaptNet + sign loss (w 1 / 3) | 0.612 / 0.589 / 0.562 ; 0.615 / 0.588 / 0.561 | same as AdaptNet |
  Per fold: essential screens 0.70-0.80 at the top (HepG2, Jurkat), non-essential panels 0.53-0.58 (X-Atlas, CD4,
  iPSC), also against the truth without the common mode -> for non-essential targets in a new cell type the gene-level
  direction barely transfers; a direction loss cannot change it (the model only reweights the sources).
- Public leaderboard (A/B/C, api/leaderboard, 1,635 teams): Illumina AI 0.472 (raw mse 0.235, reach 0.495 - far ahead,
  likely own data); places 2-40 at 0.30-0.35 with raw pds ~0.87, mse 0.50-0.57, nmae 0.82-0.86, fid ~0.53, reach
  0.30-0.37, jac ~0.035; ~100 teams cluster at mse 0.56-0.64. Ours (G3v) 0.239: pds 0.855, mse 0.708, nmae 0.922,
  fid 0.519, reach 0.250, jac 0.034 -> gaps mse (-0.18 scaled), nmae (-0.12), reach (-0.09). Nobody's jac is high: the
  empty-set convention (N_conf = 0 -> calling nothing scores 1.0) is rare on these panels (the scorer docs say so too).
- mse_proxy.py: the mse member from gene totals alone (bulk_lognorm, jackknife corrections, #348 budget); tracks the
  official raw mse with a constant offset (+0.05) and reproduces differences between variants within 0.004. With it on
  s0: the budget = claim global scale is optimal (x0.75 0.903, x1 0.876, x1.25 0.923); ORACLE per-target trans scales
  only reach 0.833 (-0.043 raw, ~+0.008 overall), 11-13 of 60 targets best at 0; the common part is 6.6 % of our
  squared error. -> neither the common response nor per-target amplitude explains the mse gap to other teams.
- Blend AdaptNet + w x consensus (same size) on the hidden-cell-type benchmark: w 0 0.152 / 0.766, w 0.25 0.154 /
  0.761, w 0.5 0.153 / 0.757 -> no gain, AdaptNet alone stays.
- Propagating predictions through the hidden cell type's own control co-variation (run_prop_ctrl.py, 9 contexts with
  saved controls): consensus dir 0.134 -> 0.142 but pds 0.743 -> 0.709 (a 0.2) -> rejected (smooths targets together).
- Natural variation as a knockdown predictor (natvar.py: partial correlation with the target across 20,001 H1 control
  cells after library size + 10 state axes): dir -0.010, pds 0.473 (chance); blends only lower AdaptNet
  (0.185 / 0.867) -> no signal (targets are lowly expressed, single-cell values mostly zero).
- Totals post-processing sweep for the trained-model totals (totals_sweep.py, proxy raw mse, mean of 5 subsets; current
  = shared-k 0, 50 control axes, gamma 1): no control-axis removal -0.031 (5/5 subsets), 20 axes -0.009, 100 axes
  +0.005; target weights (agreement x n_sources)^0.5 -0.024 (5/5); gamma 1.5 -0.007; entry agreement ~0; shared-k 3
  +0.050. The control axes were adopted for the noisier consensus totals (10-20); for AdaptNet they remove signal (each
  cell type's common response lies largely in its top control axes). Official mock check of CTRLAXES=0 (+ TWPOW=0.5)
  running (final_pipeline.sh now takes CTRLAXES, default 50).
- Official mock (5 subsets, H1 hidden) of the totals choices:
  | variant | overall | pds | mse | nmae | fid | reach | jac | subsets better than hybrid_rpe1 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | hybrid_rpe1 (50 control axes) | 0.2934 | 0.859 | 0.102 | -0.023 | 0.549 | 0.073 | 0.202 | - |
  | CTRLAXES=0 | 0.2987 | 0.846 | 0.133 | -0.015 | 0.551 | 0.074 | 0.202 | 4/5 |
  | CTRLAXES=0 TWPOW=0.5 | **0.3020** | 0.852 | 0.159 | -0.023 | 0.548 | 0.074 | 0.202 | 4/5 (s2 -0.003) |
  Cross-check of the totals choices on HCT116 / KOLF200 hold-outs (totals_xcheck.py, proxy mse) running.
- Cross-check on other hidden cell types (totals_xcheck.py; AdaptNet without the hold-out's group, trans only, proxy mse):
  | hold-out | 50 axes (current) | 0 axes | 50 axes + target weights | 0 axes + target weights |
  | --- | --- | --- | --- | --- |
  | HCT116 | 1.010 | 0.946 | 0.994 | **0.922** |
  | KOLF200 | 1.073 | 1.076 | 1.055 | **1.056** |
  -> CTRLAXES=0 + TWPOW=0.5 adopted for the trained-model totals (better on H1 mock, HCT116 and KOLF200).
- 3-seed AdaptNet ensemble on top (mock_exam.py SEEDS=0,1,2): **0.3055** (pds 0.859, mse 0.172, nmae -0.020, fid 0.548,
  reach 0.072, jac 0.202); vs single seed 0.3020 better in 4/5 subsets (1 tie), vs hybrid_rpe1 5/5.
  -> gm_final_v4 = AdaptNet (beta 1, RPE1, seeds 0-2 averaged) totals per context with CTRLAXES=0 TWPOW=0.5, cells as
  v1-v3 (consensus x agreement x4.5, top 2 %). Mock: v2 config 0.2910, v3 0.2934, v4 0.3055.
- Round-2 proxy sweep around v4 on the ensemble inputs (5 subsets): v4 (0 axes, target weights ^0.5) -0.056 vs the
  ensemble without them; weights ^0.25 -0.049, ^0.75 -0.057, ^1 -0.055; + gamma 1.5 -0.045; 10 axes -0.040 -> v4's
  settings sit at the optimum.
- 10-23 18:40: gm_final_v4 uploaded (user OK) and scored without error: entry jmkScAMG7a7BbaXxIfrz, now the counting
  final entry (vccsub/out/gm_final_v4.vcc, build log vccsub/out/build_gm_final_v4.log).
- Scorer-space training (adaptnet.py ew: gene errors weighted by (x/(1+x))^(2 ew), x = query control CP50k) on the
  hidden-cell-type benchmark in the scorer's weighting (bench_w.py): ew 0 weighted dir 0.179 / pds 0.768 / skill 0.033,
  ew 1 0.179 / 0.773 / 0.032, ew 0.5 0.180 / 0.771 / 0.032 -> within seed noise; not adopted (the model only reweights
  sources and programs, so the loss weighting barely changes it).
- v4 totals, proxy (5 subsets): overall trans scale x0.9 -0.010, x1.1 +0.026 (budget = claim stays, near optimal);
  cis neighbours x2 / x3 -0.001 (no effect).
- Ordering the cells' DE signal by source consistency (t = mean / (sd / sqrt n) over the signature-cache sources) instead
  of |cells| x sqrt(expression): sign precision @10 0.52 (x expression) / 0.43 (t alone) vs 0.69; reach 0.099 vs 0.170
  -> genes that are consistent across the centred sources have the WRONG sign in H1 more often than not (centring
  artefact: a target's weaker-than-average generic response turns into a consistent anti-generic signal). Not used.

## 10-23 evening: three mock sets (H1, HCT116, KOLF200)
- mock_data_loco.py builds HCT116 (4 x ~66 targets, ~160 truth cells / target) and KOLF200 (4 x ~70, 200 cells) mock sets
  from the 10-11 hold-outs; mock_exam.py MOCK_SET=hct116 / kolf200 (VCC_EXCLUDE and the model's training contexts follow).
  Their replicate anchors are unstable (few cells, many targets without significant genes: HCT116 baseline jac 0.26 from
  empty reference sets, nmae gated on 7 / 66 targets) -> compare RAW members there (mock_raw.py).
- mse_proxy.Proxy.pds reproduces the official pds_cosine from the totals within 0.001 (H1 and HCT116); the mse proxy's
  offset is constant on H1 (+0.05) but varies on HCT116 (+0.07 / +0.14), so mse differences there are approximate.
- Totals choices with both proxies (totals_sweep3.py; v4 inputs):
  | set | old pipeline | v4 (0 axes, tw 0.5) | 50 axes, tw 0.5 | 20 axes, tw 0.5 | 0 axes, no tw | 0 axes, shared-k 3 / 10 |
  | --- | --- | --- | --- | --- | --- | --- |
  | H1 mse / pds | 0.963 / 0.892 | 0.832 / 0.906 | 0.868 / 0.913 | 0.858 / 0.908 | 0.855 / 0.903 | 0.921 / 0.888 ; 0.986 / 0.855 |
  | HCT116 mse / pds | 1.088 / 0.757 | 0.997 / 0.705 | 1.069 / 0.768 | 1.063 / 0.763 | 1.018 / 0.703 | 1.091 / 0.695 ; 1.191 / 0.691 |
  On HCT116 the control-axis removal matters for pds (+0.063) while on H1 it mostly costs mse -> cell-type dependent.
- Cells + training common response (COMMON=common_train_<set>.npz COMMONSCALE=0 CELLCOMMON=3, official mocks):
  H1 0.3055 -> 0.3117 (nmae +0.010, fid +0.018, jac +0.014, reach -0.005 scaled); HCT116 raw nmae 0.805 -> 0.808,
  reach 0.645 -> 0.642; KOLF200 raw nmae 0.995 -> 0.988, reach 0.148 -> 0.141, fid -0.001 -> reach drops on all three,
  gains only on H1 -> not adopted.
- KOLF200 proxies: old pipeline 1.152 / 0.652; v4 1.109 / 0.598; 50 axes 1.107 / 0.654; 20 axes 1.106 / 0.656 (mse / pds).
  Depth (UMI per control cell): H1 24k, HCT116 12k, KOLF200 3.9k, A 20k, D/E/F 20k -> H1 is the closest to the exam.
- Number of control axes removed from the v4 totals (proxies, mse / pds):
  | axes | H1 | HCT116 | KOLF200 |
  | --- | --- | --- | --- |
  | 0 (v4) | 0.832 / 0.906 | 0.997 / 0.705 | 1.109 / 0.598 |
  | 5 | 0.840 / 0.911 | 1.028 / 0.747 | 1.104 / 0.656 |
  | 10 | 0.848 / 0.909 | 1.048 / 0.755 | 1.105 / 0.657 |
  | 20 | 0.858 / 0.908 | 1.063 / 0.763 | 1.106 / 0.656 |
  | 30 | 0.863 / 0.910 | 1.066 / 0.766 | 1.107 / 0.655 |
  The first 5 axes carry most of the pds loss on the shallow sets, at little mse cost -> CTRLAXES=5 candidate (official
  mocks on the three sets queued).
- Heterogeneous down genes in the cells (DOWN_HET=1, render.py): H1 0.3050 vs 0.3055 (2/5 subsets better) -> neutral.
- DOWN_HET=1 on the other sets (raw): HCT116 reach 0.645 -> 0.678, fid / nmae / jac flat; KOLF200 reach 0.148 -> 0.150,
  nmae 0.995 -> 0.997, jac 0.0176 -> 0.0159. With H1 neutral (0.3050 vs 0.3055) -> kept as an option, not adopted.
- Official mocks, CTRLAXES=5 vs v4 (0 axes): H1 0.3063 vs 0.3055 (pds raw 0.905 -> 0.910, mse 0.788 -> 0.795); HCT116 raw
  pds 0.704 -> 0.747, mse 0.861 -> 0.915; KOLF200 pds 0.597 -> 0.654, mse 1.054 -> 1.046; DE members flat. Adopted ->
  gm_final_v5 = v4 with CTRLAXES=5 (same gm_final_v4pre inputs).
- 10-24 00:24: gm_final_v5 uploaded (user OK) and scored without error: entry 9uhqF2Pvw5jKXcjLlpkB, now the counting
  final entry (vccsub/out/gm_final_v5.vcc). Progress deck for the professor: deck/project (artifact 1oFtaxuSjLczEm8McQjDkm).
- 10-24 afternoon: professor deck redone in oral-defence style (outline, section dividers, claim titles with figure evidence,
  references, backup slides, speaker notes on all 34 slides): deck/make_defense_pptx.py + deck/make_figures.py ->
  docs/VCC_defense_style_20261024.pptx (+ PDF). Checked slide by slide in LibreOffice renders; replaces the card-style deck.
- 10-24 evening: reproduction test of general_model/README.md section 11 in a fresh clone from GitHub (raw downloads replaced by
  the processed stats of this machine; HCT116/KOLF200 hold-outs not rebuildable here). Training data, H1 hold-out, signature
  cache, H1 mock sets, v1 exports, AdaptNet totals (kappa 6.251 / 6.138 / 6.263) all bit-identical; the rebuilt
  gm_final_v5.vcc has the same SHA-256 as the uploaded file (9a174d6e1c24...); H1 mock s0 0.3096 (v5) / 0.2751 (v1) and the
  consensus_d200 benchmark identical. Measured: model env ~27 min (PyTorch download), first mock subset ~28 min.

## 10-24 night: validation clean-up after the Codex audit (workspace `CODEX_AUDIT_20261024.md`)
- Robustness fixes, no change to results (commit 1f62bd4): downloads only reuse `.done`/`.part` with a matching `.meta`
  (url/size/chunk), fetch scripts stop on the first failure, `make_effect_submission.py --require-all` (STRICT=1 in
  final_pipeline.sh) stops on missing targets or non-finite effects, the 13-source cap prints a note, mock_exam.overall
  reads `from_replicate`, credential patterns in .gitignore. Fresh clone of that commit: AdaptNet totals inputs
  bit-identical and gm_final_v5.vcc rebuilt with STRICT=1 has the uploaded SHA-256 (9a174d6e1c24...).
- Noise term of the 7-fold metrics (audit M2): bench.score's variance is the delta-method variance of log1p(CP50k), but
  y = ln((x_p + 0.5)/(x_c + 0.5)) - common mode; on that scale the perturbed side's variance is ((1 + x_p)/(0.5 + x_p))^2
  larger and the control side cancels with the common mode. var_scale_check.py re-scores every saved result exactly
  from its JSON (only the noise term moves; pds and comparisons within a context do not): noise share of sum y^2 per
  context 0.13-0.54 -> 0.21-0.83; dir +19 % for every method (consensus_d200 0.111 -> 0.131, AdaptNet b1 0.131 -> 0.156,
  GeneNet4+RPE1 0.152 -> 0.181, AdaptNet+RPE1 ens3 0.154 -> 0.182; gains +18 % -> +19 %, +37 % -> +38 %). On shallow
  data the ln-scale delta method overstates the noise (TianNeuron2021: 0.71 -> 1.20 of sum y^2, impossible), so the two
  scales bracket the truth; tables keep the old scale (results/var_scale.json).
- How much could transfer at all (ceiling_check.py): noise-corrected cosine between the hidden response and the plain
  mean of the sources' true responses, same targets and genes as bench. With RPE1: 0.193 (ln-scale noise) / 0.135
  (log1p scale) vs AdaptNet+RPE1 ens3 0.182 / 0.154 = 94 % / 113 %; without RPE1 0.169 / 0.113 vs AdaptNet b1 0.156 /
  0.131 (consensus_d200 0.131 / 0.111). Per fold (ln scale, RPE1): K562 0.185, XAtlas 0.068, H1 0.250, CD4 0.067, HepG2
  0.381, Jurkat 0.336, iPSC 0.065; AdaptNet is at or above it except where the signal is strong (H1 0.210, HepG2 0.334,
  Jurkat 0.320). The single best source picked with the answer: 0.201 / 0.139. -> with these public experiments the
  transferable part of the direction is used up; non-essential panels carry ~0.07 even without noise.
- Outer test data: TianNeuron2021 rebuilt from a fresh Zenodo 13350497 download (TianKampmann2021_CRISPRi.h5ad) with
  sc_build.py is identical to the stored npz (README 11.7 recipe). Feng licence: figshare 27989294 v2 says MIT (API,
  10-24), README table fixed. Old `vcc.gm loco` and DATA_AND_METHODS.md marked as historical.
- Context PCA from the training contexts only (audit M1; STRICT_PCA=1, responsenet._ctx_pca_strict: no held-out, A-F or
  D/E/F controls), AdaptNet seed 0, beta 1, run_clean_checks.py: 0.131 / 0.762 -> 0.136 / 0.765 (dir / pds); with RPE1
  0.152 / 0.766 -> 0.156 / 0.769 (every fold equal or better except iPSC -0.001 dir). The default-PCA RPE1 run reproduces
  genenet4_rpe1.json to 4e-8. -> the transductive basis did not inflate the benchmark; strict is equal or better
  everywhere, so v5 + STRICT_PCA=1 is checked on the three official mocks (below) as the one candidate change.
- Untouched outer test: TianNeuron2021 (166 targets, never used to select anything) as a scored context (NEURON=outer,
  ONLY_FOLDS=Neuron), training on all 14 + RPE1: consensus_d200 0.012 / 0.543, AdaptNet 0.011 / 0.551, AdaptNet strict
  PCA 0.012 / 0.558; plain-mean reference (log1p-scale noise; ln-scale noise exceeds the signal here) 0.007, best single
  source 0.014. -> nothing transfers to these neurons for any method; the mock exams (H1 reference 0.25) are an easy case.
- Leave-genes-out without the centring leak (audit H1; CLEAN_CENTRING=1: every fold re-centres the training contexts on
  the common mode of their training rows only, bench.centred_on; GeneNet's inner calibration likewise; randomized SVDs
  seeded without touching the caller's RNG): knn_combo S1 0.115 / 0.605 -> 0.118 / 0.606, S2 0.024 / 0.558 -> 0.024 /
  0.558; blend S1 0.130 / 0.600 -> 0.135 / 0.603, S2 0.027 / 0.557 -> 0.027 / 0.557 (S1 truth now uses the
  training-only common mode). -> the leak did not inflate the gene benchmark; GeneNet4 S2 re-running.
- adapt_b1_check (run_clean_checks.py adapt, default PCA, no RPE1) equals adapt_b1_raw exactly, so both strict-PCA
  comparisons above are matched pairs. On the ln scale the strict runs read 0.162 (vs 0.156) and 0.185 (vs 0.181).
- GeneNet4 + RPE1 S2 without the leak (genes_genenet4_rpe1_clean.json): 0.025 / 0.571 -> 0.025 / 0.565.
- v5 + STRICT_PCA=1 on the official mocks (v5_strictpca vs v4_ax5, 10-25 morning; a first try at 01:15 was OOM-killed
  while the user's own job filled the memory, the queue then waited for >= 12 GB free): H1 0.3063 -> 0.3077 (subsets
  -0.0027 / +0.0037 / +0.0038 / +0.0024 / -0.0005), HCT116 raw pds 0.747 -> 0.741, mse 0.915 -> 0.918, KOLF200 pds
  0.654 -> 0.653, mse 1.046 -> 1.041, DE members flat. -> within noise, fails the adoption rule; v5 stays the counting
  entry (nothing uploaded). Context PCA with or without the test contexts' controls makes no difference.
