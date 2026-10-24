# KOLF hold-out pilot (12 targets) — 2026-10-11

**This is a 12-target pilot, not a benchmark.** Pilot-specific real bundle (cell-eval2 0.18.0, vcc2026 preset, de.backend=pdex); scores are on that bundle's scale, not comparable with leaderboard numbers.

## Design (as run)
- Targets: 12 drawn with seed 20261011 from loco/KOLFbig/real.json. **Deviation:** only 7 targets have ≥400 real cells in kolf_KOLF.h5ad, so the draw used the 22 targets with ≥350 and the truth set has 350 cells/target (not 400). Targets: ABCD1, ALG12, BCL6, CAPRIN2, ELK1, FZD2, MBD4, REV1, SLC39A1, SNN, TRIM35, USP28.
- Truth: 350 cells/target + 4,000 fixed NTC (seeded). Predictions: 400 cells/target sampled without replacement from those 4,000 NTC, indices identical across groups (per-target seed [gen_seed, target_index, 0]; counts seed [gen_seed, target_index, 1]); original-library up-regulation formula, lfc clipped ±4; generation seeds 0/1/2.
- Effects: existing hold-out-KOLF exports (runq.log ~955-971 show fit lists without KOLF, each followed by the file write; sha256 in manifest). Own/cis terms identical across plain/shuffled/own+cis; shuffled = plain with the trans signature of another target (fixed derangement over the 279 targets).
- Strict donor control: one model fitted without KOLF, FengNF, FengF (Feng metadata lists HipSci donors kolf_2/kolf_3; KOLF2.1J derives from the KOLF2 donor); strict plain and strict shuffled come from that same fit with the same derangement (checked).

## Panel results (official scale on the pilot bundle; mean over 3 seeds, sd in brackets)
| group | avg | pds | mse (score) | panel raw mse_norm | nmae | fid | reach | jac |
|---|---|---|---|---|---|---|---|---|
| control_resample | -0.123 (0.068) | -0.030 | 0.000 | 1.224 (0.039) | -0.158 | -0.008 | -0.517 | -0.027 |
| own_cis | -0.121 (0.068) | -0.015 | 0.000 | 1.223 (0.039) | -0.157 | -0.008 | -0.517 | -0.027 |
| gm_plain | -0.030 (0.056) | 0.030 | 0.000 | 1.820 (0.027) | -0.180 | 0.364 | -0.490 | 0.095 |
| shuffled | 0.007 (0.069) | 0.222 | 0.000 | 2.157 (0.040) | -0.204 | 0.452 | -0.573 | 0.142 |
| strict_plain | -0.022 (0.058) | 0.104 | 0.000 | 1.750 (0.027) | -0.176 | 0.347 | -0.490 | 0.086 |
| strict_shuffled | -0.023 (0.064) | 0.104 | 0.000 | 2.072 (0.041) | -0.216 | 0.422 | -0.578 | 0.133 |

Per-seed avg (seed effects are shared across groups because control indices are shared):

| group | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| control_resample | -0.192 | -0.122 | -0.056 |
| own_cis | -0.192 | -0.114 | -0.056 |
| gm_plain | -0.082 | -0.039 | 0.030 |
| shuffled | -0.048 | -0.016 | 0.084 |
| strict_plain | -0.079 | -0.023 | 0.037 |
| strict_shuffled | -0.074 | -0.044 | 0.049 |

The official mse member exists only as a panel ratio of sums; the per-target table carries only diagnostics (`diag_mse_capped`, `diag_ctrl_distance`). The 0.5 fidelity column in results_per_seed.csv (`heuristic_fid_scaled_b05_NOT_OFFICIAL`) is a heuristic and is not part of any score.

## Plain vs shuffled, per target (seed-averaged first, bootstrap over the 12 targets, 10,000 draws)
| comparison | metric | mean diff (plain − shuffled) | 95% CI | targets where plain better |
|---|---|---|---|---|
| full | diag_mse_capped (lower better) | -0.0010 | [-0.0022, +0.0002] | 7/12 |
| full | fid | -0.0714 | [-0.2083, +0.0610] | 5/12 |
| full | jac | -0.0106 | [-0.0252, +0.0025] | 6/12 |
| full | nmae (lower better) | -0.0111 | [-0.0720, +0.0428] | 6/12 |
| full | pds | -0.0657 | [-0.1540, +0.0253] | 2/12 |
| full | reach | +0.0497 | [-0.0051, +0.1273] | 6/12 |
| strict | diag_mse_capped (lower better) | -0.0009 | [-0.0023, +0.0003] | 7/12 |
| strict | fid | -0.0607 | [-0.2155, +0.0898] | 5/12 |
| strict | jac | -0.0107 | [-0.0259, +0.0033] | 5/12 |
| strict | nmae (lower better) | -0.0185 | [-0.0760, +0.0291] | 6/12 |
| strict | pds | -0.0000 | [-0.0581, +0.0657] | 4/12 |
| strict | reach | +0.0531 | [-0.0020, +0.1296] | 6/12 |

## Split-half repeatability (Codex-authored `split_repeatability.py`)
Median over targets of the mean (10 random splits) Pearson between half-cell effects: 0.291 (cosine 0.293), 11314 genes, target genes excluded. Within-screen split of cells with independent NTC halves; not biological replication and not an official ceiling.

## Current reading (12 targets, so weak evidence)
- No reliable advantage of correctly paired trans signatures over shuffled ones was detected on this panel: pds leans to
  shuffled (−0.066, CI crosses 0, 2/12 targets), fid/jac/nmae differences straddle 0, reach leans to plain (+0.05, CI
  touching 0). With 12 targets and a single fixed derangement this cannot show that target-specific signal is absent.
- Removing FengNF/FengF (HipSci donors kolf_2/kolf_3) changed little in this pilot (strict pds diff 0.000; strict plain avg
  −0.022 vs full plain −0.030); this does not fully rule out effects of the related genetic background.
- Panel raw mse_norm: control 1.22, own+cis 1.22, plain 1.82, shuffled 2.16 — every trans version is further from truth than
  unperturbed controls on this panel (score floored at 0).
- Generation-seed variation of the panel avg (sd ≈ 0.06) is as large as most group differences; comparisons are only
  meaningful as within-seed pairs.
- The gain of plain and shuffled over own+cis coincides with producing many DE calls; this pilot does not establish whether
  any part of it reflects target-specific information.

## Files
manifest.json, status.json, run.log, pilot.py (pilot_v1 backup in the job tmp dir), data/ (real.h5ad, bundle), pred/ (predictions + control_indices_seed*.json), effects/ (strict), scores/<group>_seed<s>/, results_per_seed.csv, results_per_target.csv, paired_plain_vs_shuffled_per_seed.csv, paired_{full,strict}_seedmean.csv, bootstrap_plain_vs_shuffled.csv, anchors.json, split_half_repeatability*.csv/json.
