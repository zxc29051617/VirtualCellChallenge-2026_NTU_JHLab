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

- Challenge controls are present under `dataset/`: validation controls for contexts `A`, `B`, `C`.
- This repo contains a baseline predictor that uses the control-cell mean per context.
- Official scoring is server-side: six metrics normalized against a baseline model and
  averaged into an overall score. Matching the baseline scores 0.
- **There is no local ground truth**, for any context. Scoring a prediction locally against
  challenge data is not possible; the organizers hold the perturbed cells. Local evaluation
  requires an external public perturbation dataset — see `notes/research_public_datasets.md`.
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

- `scripts/inspect_challenge.py` prints a compact challenge/data inventory.
- `scripts/build_baseline.py` builds the simple context-mean baseline prediction.
- `scripts/evaluate_with_cell_eval.ps1` shows how to run the official evaluator once ground truth is available.
- `scripts/package_submission.ps1` runs the official `vcc prep` packaging step.

Example packaging call:

```powershell
.\scripts\package_submission.ps1 -InputH5ad artifacts\baseline_prediction.h5ad -Genes dataset\gene_names.csv -Perts dataset\pert_counts.csv -OutputVcc artifacts\baseline_prediction.vcc -DryRun
```

## Notes

- The local `dataset/gene_names.csv` includes a header row; the official evaluation tooling expects a headerless gene list. The helper code strips that header when needed.
- The challenge submission must satisfy the official rules, including exact context labels, exact perturbation panel, raw integer counts, and no control cells.
