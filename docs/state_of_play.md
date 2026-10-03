# State of play — 2026-09-21

Where this stands, what the next move is, and what has already been ruled out. Written so
work can resume without the conversation that produced it. `docs/findings.md` holds the
evidence; this file holds the decisions.

Deadline: **2026-11-05 23:59 UTC**. Test contexts release 2026-10-22. Two submissions per day.

## Submissions so far

| | file | overall | rank |
| --- | --- | --- | --- |
| v1 | `artifacts/transfer.vcc` | −0.9934 | — |
| v2 | `artifacts/transfer_v2.vcc` | 0.0434 | — |
| **v5** | `artifacts/transfer_v5_blend.vcc` | **0.1040** | **377** |

v5's card, against the leader's (each metric is worth exactly one sixth — the overall is an
unweighted mean, measured to six figures in findings section 14):

| metric | v5 | leader | gap |
| --- | --- | --- | --- |
| `pds` discrimination | 0.5674 | 0.853 | 0.286 |
| `mse` expression accuracy | **0.0000** | 0.423 | **0.423** |
| `reach` DE direction reach | 0.0629 | 0.373 | 0.310 |
| `nmae` DE log-FC accuracy | 0.0313 | 0.293 | 0.262 |
| `fid` DE direction fidelity | **−0.0316** | 0.111 | 0.143 |
| `jac` DE significance overlap | **−0.0058** | 0.016 | 0.022 |

## What v5 is

`transfer_v5_blend.h5ad` / `.vcc`, built by:

```powershell
python -m vcc2026.cli build-transfer `
  --source external/profiles/K562gw_perturbation_means.npz `
  --secondary external/profiles/CD4_target_lfc.npz --secondary-weight 0.10 `
  --scale 0.3 --profile-scale 0.15 --coverage all_expressed `
  --cell-patterns 200 --cells-per-pert 400 --seed 0 `
  --output artifacts/transfer_v5_blend.h5ad --run-id transfer-v5-blend

.\scripts\package_submission.ps1 -InputH5ad artifacts\transfer_v5_blend.h5ad `
  -Genes dataset\gene_names.csv -Perts dataset\pert_counts.csv `
  -OutputVcc artifacts\transfer_v5_blend.vcc
```

Each target's control profile moved by its measured fold change, carried from Replogle's
genome-wide K562 screen and averaged with Marson 2025's CD4 T-cell screen at weight 0.10.
286 of 300 targets carry a target-specific prediction; 14 fall back to a generic response.
16,135 genes at profile scale 0.15 — 0.95e9 stored entries, which `vcc prep` can package.

**The 0.0434 → 0.1040 jump is not attributed.** v5 changed two things against v2 at once: the
second source, and the full gene axis in place of v2's 9,923 selected genes. Local folds
predicted +0.006 raw `pds` from the blend; the move was +0.11. The axis is the likelier cause
(findings section 14). `artifacts/transfer_v3_geneaxis.vcc` is built, unsubmitted, and would
separate them for one slot.

## Next move: attack `mse`

The largest gap on the card, at 0.423, and **nothing in this repo has ever targeted it**.

`mse` came back exactly 0.0000 — not small, but precisely the baseline. The prediction is the
control profile moved by a fold change shrunk by `--scale 0.3`, which for this panel is close
enough to the context mean that the metric cannot separate the two. `scale` has always been a
stated assumption rather than a fitted quantity (findings section 1); least squares put it at
0.27 for K562→RPE1, but shrinking a noisy predictor is right for squared error and wrong for
ranking, and section 3 measured that applying it per component collapses discrimination.

So the experiment is a sweep of `--scale` at v5's settings, scored locally on both folds,
watching `pds` and `expr_error` together — a value that buys `mse` at the cost of `pds` is a
loss, since both are worth one sixth. It costs no stored entries and packages unchanged.

**Read local numbers as direction only.** They have now mispredicted a change by more than an
order of magnitude in both directions: 0.82 → 0.501 unfavourably, +0.006 → +0.11 favourably.

## Do not submit v4 as built

`artifacts/transfer_v4.h5ad` exists and **cannot be packaged on this machine**. It is also
not "v5 with better cells" — it predates the CD4 work, so it trades v5's advantages away:

| | v5 | v4 |
| --- | --- | --- |
| source | gwps + CD4 blend | gwps only |
| targets with a specific prediction | 286 | 267 |
| cell construction | floor/ceil — 7,581 false DE | conditioned NB — 8 false DE |
| profile depth | 0.15 | 1.0 |
| stored entries | 0.95e9 — packages | 2.08e9 — does not |

Upside if its clean DE table reached leader level on `fid`+`jac`+`reach`: +0.079 overall.
Downside from losing 19 targets: about −0.005. Positive in expectation, but "no manufactured
DE calls" is not the same as "the right DE calls", and the combination actually wanted —
v5's blend with v4's cells — needs 2.08e9 entries and cannot be packaged here.

The packaging wall is `vcc prep`'s `_all_values_integer`, which materializes the whole value
array and copies it for `abs`. Measured: 0.94e9 packages, 2.08e9 does not, at 24/30/40 GB of
free commit. `--no-require-counts --allow-discrete` skips that check and fails later anyway.
Raising the Windows page file to ~120 GB on D: (583 GB free, commit limit is currently 68.7 GB)
is the untried fix; it needs an administrator and a reboot.

NB construction is only clean at full depth — at 0.5 it produces 6,638 false DE and at 0.15
it produces 9,884, worse than the floor/ceil it replaces (findings section 12). So there is no
density compromise that keeps the benefit.

## Ruled out, with the measurement

- **Targeting `jac` for its headroom.** The top three score 0.011–0.030 across 20–61
  submissions. Headroom is not reachability (section 10).
- **Predicting that the knocked-down gene drops.** Every metric excludes each perturbation's
  own target gene; `pds` removes all 300 panel targets from the feature space.
- **Uniform profile scaling being free.** True for cosine, false for the three rank-based DE
  metrics: below full depth our counts sit on a coarser grid than the reference's (section 12).
- **Partial gene coverage.** An unstored gene in a sparse matrix reads as predicted zero, not
  as an abstention. `nmae`'s −6 floor binds below 93% coverage (section 9).
- **Replacing transfer with a generative model.** Flow and diffusion on identical data and
  splits win slightly on expression error and lose on discrimination, both directions.
- **A multi-context test on public CRISPRi data.** No shared perturbation panel exists across
  Replogle and Tian; the one overlap that does is 51 perturbations, below the noise floor.
- **Filling unmeasured genes from scGPT neighbours.** Strong inside one cell line (PDS 0.83 to
  0.89 against 0.60 to 0.63 with shuffled embeddings) and nothing across cell types, on the
  3,578 genes CD4 measured and K562 did not (r 0.002 to 0.004, PDS 0.51 to 0.53). Not wired in
  (section 15).

## Assets

- `external/profiles/CD4_target_lfc.npz` — 645 rows, 234 of the Challenge's targets.
- `external/profiles/CD4_essential_lfc.npz` — 1,894 rows, 744 targets, for the scorable fold.
- `src/vcc2026/remote.py` — reads a 16.8 GB remote HDF5 over HTTP range requests, with retry.
- `scripts/compare_cd4_transfer.py` — held-out blend evaluation with a shuffled control.
- `scripts/cd4_gwps_agreement.py` — matched-vs-mismatched agreement between two screens.
- `src/vcc2026/embeddings.py`, `src/vcc2026/neighbors.py`, `scripts/extract_scgpt_embeddings.py` —
  scGPT gene embeddings and FAISS retrieval, behind `build-transfer --neighbor-embeddings` and
  off by default. Borrowing a target's effect from its neighbours scores PDS 0.538 against a
  shuffled-embedding null of 0.514 (best of 20 settings, p = 0.078) and touches 14 of 300
  targets under v5's sources, so the expected gain is under +0.001. **Not submitted**; v5 is
  still the best score and reproduces with the same command (section 15).

Push target is `friend` (`scnpg/VCC_2026_NTU_JHLab`); `origin` is the personal mirror.
