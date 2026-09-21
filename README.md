# Virtual Cell Challenge 2026

This repository is set up for the 2026 Virtual Cell Challenge with a deliberately simple first-stage baseline.

The 2026 task is **zero-shot**: predict CRISPRi knockdown responses in cell lines never seen
perturbed, given only their unperturbed control cells and a list of genes to knock down.
No challenge-specific training set is released. See `docs/challenge_brief.md` for the full
rules, the six scored metrics, and the timeline.

Current goals:

1. Understand the challenge rules and submission format.
2. Build a reproducible local environment.
3. Validate the data layout and gene alignment.
4. Produce a correct, low-complexity baseline.
5. Obtain an external perturbation dataset so that local evaluation becomes possible.

## Current status

- **Start here: [docs/state_of_play.md](docs/state_of_play.md)** — the submitted scores, the
  next experiment and why, and what has already been ruled out with the measurement that
  ruled it out. Best score so far is **0.1040 (rank 377)** with `transfer_v5_blend.vcc`.

- **2026-09-21 offline validation:** output/readback ablations, independent-control null
  tests, new-data target coverage, and a source-only flow/diffusion pilot are documented in
  [the experiment report](docs/offline_validation_20260921.md). These are local diagnostics,
  not official leaderboard scores. The existing `transfer_v2.vcc` is unchanged.

- Challenge controls are present under `dataset/`: validation controls for contexts `A`, `B`, `C`.
- This repo contains a baseline predictor that uses the control-cell mean per context.
- Official scoring is server-side: six metrics normalized against a baseline model and
  averaged into an overall score. Matching the baseline scores 0.
- **There is no local ground truth**, for any context. Scoring a prediction locally against
  challenge data is not possible; the organizers hold the perturbed cells. Local evaluation
  requires an external public perturbation dataset — see `參考文獻/README.md`.
- The `vcc` profile in `cell-eval` v0.8.2 computes the 2025 three-metric panel, not the 2026
  six-metric panel, so it does not reproduce the leaderboard score.

## Suggested setup

Install the project dependencies:

```powershell
uv sync
```

Install the official evaluation tooling:

```powershell
uv tool install -U git+https://github.com/ArcInstitute/cell-eval
uv tool install -U --python 3.11 vcc-cli
```

On this machine, invoke the official `vcc` binary directly:

```powershell
& "$env:USERPROFILE\.local\bin\vcc.exe" --version
```

## Main scripts

- `uv run vcc2026 inspect` prints a compact challenge/data inventory.
- `uv run vcc2026 build-baseline` builds the simple context-mean baseline prediction.
- `scripts/explore_data.py` walks through one context file in plain language.
- `scripts/download_external_data.ps1` fetches the Replogle 2022 data used for local evaluation.
- `scripts/build_external_profiles.py` reduces it to per-perturbation mean profiles.
- `scripts/evaluate_with_cell_eval.ps1` shows how to run the official evaluator once ground truth is available.
- `scripts/package_submission.ps1` runs the official `vcc prep` packaging step.

Example packaging call:

```powershell
.\scripts\package_submission.ps1 -InputH5ad artifacts\baseline_prediction.h5ad -Genes dataset\gene_names.csv -Perts dataset\pert_counts.csv -OutputVcc artifacts\baseline_prediction.vcc -DryRun
```

## Notes

- The local `dataset/gene_names.csv` includes a header row; the official evaluation tooling expects a headerless gene list. The helper code strips that header when needed.
- The challenge submission must satisfy the official rules, including exact context labels, exact perturbation panel, raw integer counts, and no control cells.
