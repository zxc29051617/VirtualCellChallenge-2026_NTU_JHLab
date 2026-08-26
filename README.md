# Virtual Cell Challenge 2026

This repository is set up for the 2026 Virtual Cell Challenge with a deliberately simple first-stage baseline.

Current goals:

1. Understand the challenge rules and submission format.
2. Build a reproducible local environment.
3. Validate the data layout and gene alignment.
4. Produce a correct, low-complexity baseline.
5. Define validation splits that reflect context generalization.

## Current status

- Challenge controls are present under `dataset/`.
- The workspace currently contains only validation controls for contexts `A`, `B`, and `C`.
- Official scoring uses `cell-eval` / `vcc`-style evaluation on held-out perturbation data.
- This repo now contains a baseline predictor that uses the control-cell mean per context.

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
