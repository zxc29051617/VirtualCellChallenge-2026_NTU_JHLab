# Stage 1 — reproducing scDFM on Norman (additive split)

Branch `vcc-repro`. Harness: `experiments/stage1.py` (reuses `src/script/run.py`'s own
`train_step`, `test`, `generate_sample`). Env: `requirements-repro.txt` (Python 3.11,
torch 2.7.1+cu126, cell-eval 0.5.42), RTX 4060 Ti 16 GB, 15 GB RAM.

## Checklist

| check | result |
| --- | --- |
| Norman loads | yes — 84,986 cells x 19,264 genes, already log1p-normalized; 5,029 genes after HVG + forced perturbation genes, identical to the shipped vocab (ids 4..5032, alphabetical, which the graph mask relies on) |
| split | `data/norman/split_results.pkl` from the authors: 5 folds, 88 train / 37 test double perturbations; fold 1 gives 189 train conditions, 37 test |
| forward | yes — released fold-1 checkpoint, loss 0.18 on one training batch |
| loss decreases | yes — from scratch, d_model 512, batch 8, 400 steps: 1.73 (first 10%) -> 0.57 (last 10%) |
| evaluator | rescoring the authors' saved `pred.h5ad`/`real.h5ad` reproduces their `agg_results.csv` (pearson_delta 0.8749, cosine discrimination 0.9869, mse 0.0029); only overlap_at_N differs (0.1309 vs 0.1348, tie-breaking) |
| generation + eval from checkpoint | **partial** — runs end to end (8,327 s on the 4060 Ti), but scores below the authors' saved predictions; see below |

## Config vs implementation

1. **`--d_model` never reached the model.** `instantiate_model` did not pass it, so run.sh's
   `--d_model=128` built a 512-wide, 60.4M-parameter model. The released checkpoints are
   512-wide (80k iterations, not run.sh's 200k). Fixed: d_model is now passed through.
2. **`ntoken` in the config (512) is unusable** — the Norman vocab has 5,033 entries. The
   model's own default (6,000) is what actually runs; left as is.
3. **`split_method == 'additive' or 'combinations'` is always true** (`data.py`), so the
   `unseen` split branch is dead code. Not hit by run.sh (additive).
4. **Poisson noise is broken with the default config**: `poisson_target_sum=-1` makes
   `per_cell_L is not None` true, the scale negative, and every rate clamps to 1e-8 —
   i.e. zero noise. run.sh uses Gaussian, so it does not affect the reproduction.
5. **Data layout**: the code reads `data/norman.h5ad`, the authors' zip ships
   `data/norman/norman.h5ad` (fixed with a symlink).
6. **Checkpoint selection uses the test set** — `test()` runs on `valid_sampler`, which is
   the test split; there is no separate validation split.
7. **Evaluation genes are picked on the test set**: the 1,000 genes scored are the top HVGs
   of the test cells + controls (`infer_top_gene`), not the training HVGs.
8. **`pick_eval_score` reads the wrong row**: `df[scheme].iloc[0]` is cell-eval's `count`
   row, so the printed "Current evaluation score" is the number of perturbations (37), not a metric.

## Hardware findings

- **Batch 48 (run.sh) does not fit 16 GB.** Training memory is ~0.9 GiB per cell at
  d_model 128 and ~1.35 GiB per cell at 512 — it is the 1000x1000 attention maps kept for
  backprop, so shrinking d_model barely helps. Batch 48 needs ~45-65 GB (an 80 GB card).
  Max here: batch 8 at d_model 512 (11.6 GiB, 0.72 s/step).
- **Generation is slow**: 1.3 s per forward (48 cells x 1000 genes), fp32 sgemm + explicit
  softmax attention; rk4 with 20 grid points = 76 forwards per batch, so ~90 s per 48 cells.
  Memoizing the GeneEncoder is exact (max |diff| 0) but only 1.2x faster.

## Original evaluation, and how it differs from VCC 2026

- cell-eval 0.5.42 `MetricsEvaluator`, log-normalized values, 1,000 genes, 128 generated
  cells per perturbation vs ~239 real; control cells are the real ones on both sides.
- The released model calls ~952 of 1,000 genes significant per perturbation against ~200 in
  the real data (`de_nsig_counts_pred` vs `de_nsig_counts_real`): recall 0.90, but heavy
  over-calling. VCC 2026's DE metrics (significance Jaccard, direction fidelity) punish that.
- VCC 2026 scores raw counts on all 18,533 genes with cell-eval2's `vcc2026` profile, and
  zero-shot in unseen cell lines rather than unseen combinations in a seen one.

## Regenerating from the released fold-1 checkpoint

Same checkpoint, same 37 test perturbations, same real cells and genes (`de_nsig_counts_real`
identical), original `test()`:

| metric | authors' pred.h5ad | regenerated |
| --- | --- | --- |
| pearson_delta | 0.8749 | 0.8053 |
| mse | 0.0029 | 0.0052 |
| discrimination_score_cosine | 0.9869 | 0.9744 |
| discrimination_score_l1 | 0.9730 | 0.9781 |
| de_direction_match | 0.8987 | 0.9200 |
| pr_auc | 0.3225 | 0.3326 |

The gap is systematic, not sampling noise: pearson_delta is lower on 36 of 37 perturbations
(median -0.055). The regenerated cells have fewer exact zeros (45.3% vs 50.5%) and a larger
predicted effect (|pred delta| / |true delta| 1.12 vs 1.02), while the two predictions still
agree in direction (median correlation of deltas 0.93). So the released code's sampler does not
produce the released predictions — they were most likely generated with different sampling
settings (ODE solver/steps, noise, or a different iteration than the 80k checkpoint).

### Sampler sweep (`experiments/sampler_sweep.py`, two worst perturbations, 128 cells, same controls)

| setting | BCL2L11+BAK1 pd | zeros | effect ratio | MAP2K3+MAP2K6 pd | zeros | effect ratio | time / 128 cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| authors' pred.h5ad | 0.358 | 0.508 | 1.71 | 0.663 | 0.473 | 1.47 | — |
| rk4, 20 points (run.py default) | 0.266 | 0.461 | 2.59 | 0.401 | 0.461 | 2.40 | 154 s |
| midpoint, 20 | 0.267 | 0.460 | 2.59 | 0.404 | 0.460 | 2.39 | 77 s |
| dopri5 (adaptive) | 0.267 | 0.462 | 2.60 | 0.400 | 0.465 | 2.40 | 192 s |
| euler, 100 | 0.269 | 0.461 | 2.58 | 0.406 | 0.464 | 2.36 | 200 s |
| euler, 20 | 0.306 | 0.464 | 2.39 | 0.443 | 0.465 | 2.19 | 38 s |
| one step, x0 + v(x0, t=0) | **0.502** | 0.445 | 1.65 | **0.838** | 0.440 | 1.76 | **2 s** |

(effect ratio = |pred mean - ctrl mean| / |real mean - ctrl mean|.)

- **The solver is not the cause.** rk4, midpoint, dopri5 and euler-100 agree to the third
  decimal: the ODE is converged at run.py's settings, and no solver choice reaches the authors'
  numbers or their zero fraction. What produced the released predictions stays unidentified
  (most likely a different checkpoint iteration).
- **Control-cell draw matters on weak perturbations**: BCL2L11+BAK1 scored 0.133 in the full
  run and 0.266 here, differing only in which 128 controls were drawn.
- **One-step generation** (the line commented out in `generate_sample`, here with factor 1.0)
  gives the best mean prediction of all settings, close to the authors' effect size, at ~75x
  less compute. With predict_y flow matching, the velocity at t=0 is E[x1 - x0 | x0], so one
  step lands near the conditional mean: good for mean-based metrics, but it should under-
  disperse cells, which VCC's Wilcoxon-based DE metrics will see. A cost/quality knob worth
  testing in stage 4, not a free win.
