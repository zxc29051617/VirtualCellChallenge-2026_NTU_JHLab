# Next direction after SIGnature (2026-10-19): rendering the predictions into cells

## Screens (minutes each, hold-out data only)
- `coexp_screen.py`: co-expression with the target in a context's own control cells (20k cells, library-size partialled)
  carries no perturbation signal: median cosine with the real delta 0.006 (54 % positive) on H1; adding it lowers a
  pds-like rank score. Dropped.
- `structure_screen.py`: with the current model's signal, even the best amplitude (oracle, per target) reaches only
  0.94 (H1), 0.998 (KOLF200), 0.995 (HCT116) of the "no change" error -> amplitude tuning alone cannot earn the mse member
  on these hold-outs; the real effects are not low-rank (top PC 7-13 %, top 10 PCs 17-36 % of the variance).
- `identify_lines.py` (Human Protein Atlas cell-line RNA, 1,206 lines, 206 MB download): validated on known contexts
  (K562, HepG2, HCT116, HEK293, Jurkat all ranked first); A ~ Jurkat (0.91), B ~ HeLa (0.71), C ~ CAL-33 (0.72).
  Background only; nothing is tuned to A/B/C (user's instruction: no A/B/C-specific work).

## Rendering experiment (`render_eval.py`, `analyze_render.py`, renderer `scDFM/vcc/render.py`)
Same model and effects as signature_20261013 group A (V0 = those runs, current renderer); same control draws;
200 cells/target; official cell-eval2 0.18.0 vcc2026 + pdex; fixed bundles; 3 seeds; 27 new runs.
- V1 exact gene totals: pseudobulk = prediction exactly (no Monte-Carlo noise), per-cell noise kept.
- V2 dual: bulk trans x1, cells trans x3 (top-tail adjustment). V2s: V2 with shuffled trans pairing.

| set | avg V0 / V1 / V2 / V2s | raw mse V0 / V1 / V2 | raw pds V0 / V1 / V2 | pds per target V1-V0 [95% CI] |
| --- | --- | --- | --- | --- |
| H1 | 0.347 / 0.357 / 0.339 / 0.094 | 1.45 / 1.19 / 0.86 | 0.888 / 0.901 / 0.907 | +0.013 [+0.007, +0.019] |
| KOLF200 | 0.154 / 0.161 / 0.118 / 0.071 | 2.19 / 1.14 / 1.17 | 0.550 / 0.571 / 0.568 | +0.021 [+0.003, +0.039] |
| HCT116 | 0.183 / 0.255 / 0.401* / -0.110 | 13.5 / 3.6 / 1.25 | 0.600 / 0.647 / 0.654 | +0.047 [+0.024, +0.071] |
*HCT116's mse scale is unreliable (split-half replicate 1.23 +- 0.68), which drives V2's avg there.
Pseudobulk metrics of V1/V2 are identical across seeds (totals are deterministic); only the DE members vary.

Pre-set rule (V2): raw mse below V0 in all seeds, pds not clearly lower, pairing beats shuffle -> met on all three
sets. But V2's overall avg is below V0 on H1 (-0.008) and KOLF200 (-0.036): it keeps only ~50-70 % of the cell-level
change (for sparse genes the total and the number of nonzero cells are tied), so nmae and fid fall
(`calibrate_cells.py`: a stronger cell-level input does not restore it: 0.49 at x3, 0.69 at x8). Not submitted.
V1 improves avg and pds on all three sets and was submitted (user approved).

## Leaderboard (A/B/C, anchors r4), entry j8rlnzHbqPXDRVyWXg7b
gm_v1_vip_exact: **0.1608**, rank 451 -- pds 0.586, mse 0, nmae 0.155, fid 0.024, reach 0.188, jac 0.012.
Same effects with the earlier renderer (gm_v1_vip, 10-13): 0.1482, pds 0.530, nmae 0.143, fid 0.025, reach 0.184, jac 0.007.
`tools/final_pipeline.sh` now uses `--render exact` (RENDER=current restores the old construction).

## Notes
- During the upload the system ran out of memory and killed one local scoring run (HCT116_V2_seed2); it was re-run.
- Files: run.log, status.json, results_render_runs.csv, bootstrap_render.csv, verdict_render.json, coexp_screen.json,
  upload_v1.sh/.log, scores/.

## Follow-ups the same evening (all vs V1; rules pre-registered in run.log before any score)
- Dense-gene-only dual: truth-free feasibility check (`retention_by_bin.py`) failed -- even genes with > 10 counts/cell keep
  only 0.67-0.73 of the cell-level change under a x1 bulk; not scored.
- Source transfer screen (`source_transfer_screen.py`): same-chemistry (Flex) sources do not transfer better (lowest for H1);
  biologically closer sources (iPSC for H1) and the plain average do best. Platform weighting dropped.
- Similar-target smoothing (`smoothing_screen.py`, similarity from training signatures): lowers the pds-like score on all
  three sets. Dropped.
- pds screen (`pds_screen.py`, `pds_screen_k.py`; the noise-free proxy equals the official raw pds of V1): removing the mean
  and top-k principal components of the panel's trans part raises raw pds on all sets, plateau k = 5-20 (k = 10: H1 0.901 ->
  0.927, KOLF200 0.571 -> 0.629, HCT116 0.647 -> 0.697). Cis scaling: negligible.
- V5 = V1 + top-10 shared components removed (18 runs): raw pds and raw mse better on all sets, but DE members fall
  (H1 nmae 0.198 -> 0.103, reach 0.187 -> 0.118); avg H1 0.357 -> 0.327, KOLF200 0.162 -> 0.156 -> FAILS the rule.
- V6 = cells as V1, gene totals as V5 (render mode dual; truth-free check: keeps 75-81 % of V1's cell-level change on the genes
  most changed by the removal, V5 51-62 %) (18 runs): avg H1 0.345 (-0.012), KOLF200 0.175 (+0.014), HCT116 0.479 (+0.224,
  inflated by HCT116's unreliable mse and pds scales); raw pds = V5's; pairing beats shuffle everywhere -> PASSES the rule
  (`verdict_v6.json`). Challenge file: effects VCC_gm_vip.npz (cells) + VCC_gm_vip_pc10.npz (totals), `build_v6.sh`.

## Leaderboard V6 (2026-10-19 21:25): 0.1808, rank 358
pds 0.681, mse 0, nmae 0.176, fid 0.026, reach 0.194, jac 0.008 (V1: 0.1608 with pds 0.586, nmae 0.155, fid 0.024,
reach 0.188, jac 0.012). The day: 0.1482 -> 0.1608 -> 0.1808 with the same model.

## 2026-10-20: leaderboard V7a / F9, scorer mechanics, G1
- V7a (V6 totals x0.5): **0.2032**, rank 269 (mse 0.168 from 0). F9 (V10 cells + V8 totals x0.5): **0.2080**, rank 247
  (pds 0.685 -> 0.753 from removing the context's control-cell axes in the totals; mse 0.150; reach 0.189 -> 0.166, so the
  V10 cells are not confirmed on A/B/C). V10 strength check: keep x2 (x3 better on 1 of 3 sets only).
- The public leaderboard API returns RAW members. F9 raw: pds 0.840, mse 0.846, nmae 0.908, fid 0.517, reach 0.227,
  jac 0.034. Top-40: pds 0.84-0.88, mse 0.47-0.60, nmae 0.82-0.87, reach 0.29-0.37. Entries with our pds (0.83-0.85) have
  mse median 0.76: the largest gap is mse, then reach.
- #348 on A/B/C (`budget_abc.py`, controls only): claim ~4,300-4,800; F9 budget ~3,000 (ratio 0.64-0.70). Below ratio 1
  the correction refunds the spread's plug-in cost, so raw mse falls linearly with the trans scale up to budget = claim;
  above it the error grows. Optimum ~ budget = claim; the planned x0.3 test would have been worse and was cancelled.
  At that point raw mse ~ 1 - 2 cos sqrt(claim / distance) with cos = the global noise-corrected cosine (`bulk_eval.py`).
- Cell-free screens on the hold-outs (`cos_screen.py`, `combo_check*.py`, `eb_screen.py`, `pcclean_screen.py`,
  `weights_screen.py`, `wiener_screen.py`, `generic_check.py`, `target_energy.py`, `source_quality.py`): the real energy is
  concentrated in few targets (H1 top-20 = 42 %) while our per-target magnitudes were nearly flat; per-target sharpening
  (|t_p|^(gamma-1), gamma 3) plus depth weights n/(n+200) raise cos H1 0.147 -> 0.189, KOLF200 0.025 -> 0.040, HCT116
  0.069 -> 0.093, and pds slightly. No gain: per-gene / per-target transfer coefficients, Wiener filter, generic response
  (unpredictable from sources, and even the true one costs pds heavily), knockdown weights, target expression.
- G1 = cells V1 + totals own + cis + s_c x (depth-weighted trans, shared + control axes removed, gamma 3), s_c per context at
  budget = claim (`make_vcc_g1.py`; `tools/make_bulk_effects.py --gamma 3 --match-claim` reproduces it; `final_pipeline.sh`
  RENDER=g1). Scheduled upload 10-21 08:01.
- DE metric structure (cell-eval2 source): reach ranks the reference's significant genes by OUR p-values; fid_yield_raw =
  k / max(n_pred, N_conf). Offline on the H1/KOLF200 DE tables (recomputation equals the scorer): ordering by agreement x |t|
  gives reach H1 0.24 -> 0.28, KOLF 0.14 -> 0.16; calling ~N_conf confident genes gives fid H1 0.555 -> 0.616, KOLF 0.342 ->
  0.546. On A/B/C our fid 0.517 = chance (over-calling). `de_eval.py` tests cell variants with the official scorer.

### DE rendering test (`de_eval.py`, official scorer, totals fixed, seed 0; raw members)
| cells | H1 fid / reach / jac / nmae (n_pred vs N_conf 150) | KOLF200 fid / reach / jac / nmae (n_pred vs N_conf 26) |
| --- | --- | --- |
| D0 V1 (trans x3) | 0.539 / 0.230 / 0.067 / 0.923 (539) | 0.261 / 0.138 / 0.021 / 0.992 (10) |
| D0f + added counts in top 2 % of cells | 0.549 / 0.241 / 0.066 / 0.919 (805) | 0.316 / 0.139 / 0.022 / 0.992 (16) |
| D8 trans x4.5 | 0.561 / 0.239 / 0.067 / 0.917 (1493) | 0.394 / 0.141 / 0.028 / 0.992 (37) |
| **D8f** x4.5 + top 2 % | **0.561 / 0.242 / 0.065 / 0.909** (1982) | **0.432 / 0.151 / 0.027 / 0.992** (64) |
| D7 x2.25 | 0.463 / 0.232 / 0.064 / 0.928 (198) | - |
| D1 depth-weighted x3 / D2 top-300 / D5 top-300 eq. | 0.436 / 0.281 / 0.302 fid (156 / 39 / 47 calls) | - |
Under-calling of strong targets (large N_conf) dominates: more and cleaner calls raise fid and reach; restricting the
cells to the most confident genes fails because those are mostly low-expression genes that cannot reach significance.
Decision: G2 = G1 totals + D8f cells, uploaded instead of G1 on 10-21 08:01 (pds/mse depend only on the totals).
Further rows: D9f (x6 + top 2 %) H1 0.555 / 0.252 / 0.063 / 0.903 (3104 calls), KOLF200 0.478 / 0.156 / 0.026 / 0.995 (169);
D8g (x4.5 + top 0.5 %) H1 0.559 / 0.240 / 0.065 / 0.907, KOLF200 0.439 / 0.153 / 0.027 / 0.995 -> top 2 % kept; x6 vs x4.5
neutral on H1, better on KOLF200 (fid +0.046).
Plan 10-21: 08:01 G2 (scheduled; cells x4.5). Slot 2 candidate G3 = gamma-4 totals + cells x6 + top 2 % (pds/mse change from
the totals, DE change from the cells); final choice after G2's result. D/E/F: `RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5
tools/final_pipeline.sh <dir> <name> D E F` (rehearsed on A/B/C: 15 min, matches G2).

## 2026-10-20 afternoon/evening (more screens; details run.log)
- Per-target magnitude: (source agreement x number of sources)^0.5 on top of |t|^3 (totals concentration ~20-30 of 300)
  improves the approx value on all three hold-outs (+0.016 / +0.012 / +0.015); power 1 is too concentrated on A/B/C (7 of 300).
  DepMap essentiality: correlates with response strength (0.16-0.34) but adds little. Oracle per-target magnitudes would
  raise cos only to 0.23 / 0.06 / 0.13 -> directions are the main limit.
- No gain for directions: context-similarity weights, expression-match weights, leave-one-out agreement weights, per-gene
  median / trimmed mean, quality / kcap / lam changes.
- Cells: amplifying the shared part helps locally (D8f > D10f, D9f > D11f) while A/B/C (F9 vs V7a) suggested the opposite;
  kept D8f for G2; G3w (cells x6) will show the amplification effect on A/B/C.
- Built: gm_vip_g3w.vcc = totals gamma 3 x (a n)^0.5 at budget = claim + cells x6 + top 2 % (preferred slot-2 candidate);
  gm_vip_g3.vcc = gamma-4 totals + cells x6 (alternative). tools/watch_final.sh waits for the D/E/F dataset listing.

## 2026-10-20 evening: agreement-weighted cells, refined totals, disk incident
- Cells x4.5 x per-gene source sign agreement (/ median) + top 2 % (D12f): H1 reach 0.242 -> 0.265 (fid / nmae equal), KOLF200
  fid 0.432 -> 0.472 (reach -0.007); agreement^2 no better; per-target sharpening of the cells worse. Adopted.
- Totals: per-gene agreement (k 1) before the removals + (a n)^0.5 per-target factor: approx value +0.018 / +0.016 / +0.021
  over G1 (sum +0.055); per-entry inverse-variance weights mixed (not adopted).
- Our cells' pseudobulk jackknife is 1.48x a control draw's (min 1.19) -> the claim estimate (and the budget = claim scale)
  holds.
- 16:34 the disk filled during a pack (~37 GB peak per build); superseded files deleted, builds now sequential with a 40 GB
  check (also in final_pipeline.sh).
- 10-21 08:01: G3c = g3a totals + D12f cells + top 2 %. Slot 2 chosen after G3c's members: G3v (same totals, cells x3) if the
  DE members disappoint, G2c (G1 totals, D12f cells) if pds/mse disappoint.
- D/E/F config: RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5 CELLAGREE=1 TWPOW=0.5 ENTRYAGREE=1 tools/final_pipeline.sh <dir> <name> D E F
- Night 10-20: two-seed check on H1 confirms the agreement cells (fid 0.561 vs 0.537, reach 0.256 vs 0.235, nmae 0.909 vs
  0.924); HCT116 (reference DE sets of ~2 genes) goes the other way (-0.08 approx) -> G3c vs G3v on the leaderboard decides.
  Our cells call ~726 (x3) / ~2,064 (x4.5 x agreement) genes per target on context A.
- Totals grid: cis x3 on top of G3c +0.033 (summed approx value); gamma 4 slightly more but too concentrated. Next totals
  candidate: G3c + CISSCALE=3.
- Learned combiners (gradient boosting on per-entry features; binned transfer coefficients) trained on pseudo-new contexts
  do not transfer to the hold-outs (-0.13 / -0.125): dropped.
- Full-config D/E/F rehearsal reproduces G3c exactly (18 min).
