# Virtual Cell Challenge 2026

Baseline pipeline for the 2026 Virtual Cell Challenge: predict single-cell
perturbation responses across contexts, scored with `cell-eval` / `vcc`.

## Shell

Use the PowerShell tool, not Bash. (Machine-specific reason and the workaround
live in the developer's own global notes.)

## Commands

`uv` is not reliably on PATH in the PowerShell tool — call it by full path:
`$uv = "D:\anaconda3\Scripts\uv.exe"`.

```powershell
.\scripts\evaluate_with_cell_eval.ps1 -Prediction <p> -Reference <r>
.\scripts\package_submission.ps1       # package for submission
```

`cell-eval` is not a project dependency; it runs via `uv tool run cell-eval`.

## Notes

- `dataset/` holds the challenge controls: `context_{A,B,C}.h5ad` (~200 MB
  each), currently validation controls only.
- The baseline predicts the per-context control-cell mean. Validation splits
  must reflect *context* generalization, not random cell splits.