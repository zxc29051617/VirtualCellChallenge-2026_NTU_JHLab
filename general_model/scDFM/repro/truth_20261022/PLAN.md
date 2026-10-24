# A/B/C ground truth: plan (written 2026-10-21, before the release)

Expected 10-22: A/B/C truth (perturbed cells, maybe with NTC cells); D/E/F controls (`vcc datasets download controls-final`
when it becomes available). The val leaderboard may close once the truth is public -> all later checks are local.

1. Convert with tools/vcc_truth.py (tested 10-21 on hold-out data: plain file, file without controls, shuffled gene order,
   multi-context file; sums exact):
   - `stats <truth> X`  -> data/stage3/stats/VCC_X.npz (vcc/stats.py format; controls = truth NTC or context_X.h5ad)
   - `loco <truth> X`   -> data/stage3/loco/VCC_X/ (200 cells per target + 4,000 controls) for bulk_eval / de_eval;
     `--cells 0 --n-ctrl 0` for full-size sets (official-scale scoring on the Brev box, 15 GB RAM is not enough here).
2. Check that the official scorer run locally reproduces the leaderboard raw values of G3c / G3v on A/B/C; then A/B/C is
   an unlimited exam.
3. Register the sources: vcc/contexts.py CONTEXTS += VCC_A/B/C ("stats/VCC_X", "vcc"); vcc/gm.py GROUPS += ["VCC_A", "A"],
   ["VCC_B", "B"], ["VCC_C", "C"] (exporting context A then leaves A's own truth out; D/E/F use all three); sigexp
   CONTEXTS/GROUPS/OUTER, cache_contexts.py, weights_VCC_X.json (train = the 14 sources + the other two).
   Regression check: exporting A without the new files must reproduce VCC_gm_vip.npz exactly.
4. Questions, each answered leave-one-context-out (learn on two of A/B/C, score the third; accept a change only when all
   three agree in sign, and it does not hurt the H1 / KOLF200 / HCT116 hold-outs):
   a. Do the other two contexts as sources improve the third? (likely large if D/E/F share the 300 targets)
   b. Their weight relative to the 14 public sources.
   c. Platform calibration: totals scale (budget = claim), cell amplitude, top fraction, cis scale, DE call count and LFC
      size (nmae dropped with the x4.5 x agreement cells on the leaderboard).
5. Check the overlap of the D/E/F targets with A/B/C (pert_counts.csv of controls-final).
6. Final: rebuild with the chosen settings through tools/final_pipeline.sh (rehearse on A/B/C with each context's own truth
   left out first).
