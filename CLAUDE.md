# Virtual Cell Challenge 2026

Baseline pipeline for the 2026 Virtual Cell Challenge: predict single-cell
perturbation responses across contexts, scored with `cell-eval` / `vcc`.

## Shell: use PowerShell

**Always use the PowerShell tool. Do not use the Bash tool.**

The Bash tool is broken on this machine and cannot be fixed from user config:
Claude Code's shell snapshot re-exports a Windows-format `PATH`
(`D:\anaconda3;C:\...`) into MSYS bash, which splits on `:` instead of `;`.
Every entry becomes a garbage fragment and `/usr/bin` is never on the path, so
`ls`, `head`, `cat`, `grep` all fail with `command not found`. This happens
after `BASH_ENV` and after `settings.json` `env`, so neither can override it.

If bash is genuinely unavoidable, prepend this to the command:

```sh
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"
```

## Environment

- Windows 11, Python 3.13, `uv` for dependency management.
- `python` / `uv` resolve to `D:\anaconda3` — not a project venv.
- Git for Windows lives at `D:\Git` (not the default location).

## Commands

`uv` is not reliably on PATH in the PowerShell tool — call it by full path:

```powershell
$uv = "D:\anaconda3\Scripts\uv.exe"

& $uv sync --extra dev                     # install dependencies
& $uv run pytest                           # run tests
& $uv run ruff check .                     # lint (line-length 100)
& $uv run vcc2026 --help                   # CLI entry point
& $uv run vcc2026 inspect                  # challenge manifest + data summary
& $uv run vcc2026 build-baseline           # build context-mean prediction

.\scripts\build_baseline.py                # build the baseline predictor
.\scripts\evaluate_with_cell_eval.ps1 -Prediction <p> -Reference <r>
.\scripts\package_submission.ps1           # package for submission
```

`cell-eval` is not a project dependency; it runs via `uv tool run cell-eval`.

## Layout

- `src/vcc2026/` — package. `baseline.py`, `eval.py`, `splits.py`, `io.py`,
  `submission.py`, `challenge.py`, `registry.py`, `cli.py`.
- `dataset/` — challenge controls: `context_{A,B,C}.h5ad` (~200 MB each),
  currently validation controls only.
- `scripts/` — driver scripts (PowerShell + Python).
- `docs/` — `challenge_brief.md`, `validation_strategy.md`,
  `research_public_datasets.md`.
- `artifacts/` — evaluation output, not committed.

## Notes

- Reading PDFs needs no extra tooling: the Read tool takes a `pages` argument
  (max 20 pages per call; required for PDFs over 10 pages).
- The baseline predicts the per-context control-cell mean. Validation splits
  must reflect *context* generalization, not random cell splits.
