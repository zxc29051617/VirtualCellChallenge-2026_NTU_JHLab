# VCC 2026 work on top of scDFM

Branch `vcc-repro` of a clone of AI4Science-WestlakeU/scDFM (MIT). Upstream code is unchanged except
`--d_model` now reaching the model and an opt-in fused attention (`SCDFM_SDPA=1`).

| path | what |
| --- | --- |
| `experiments/STAGE1.md` | reproducing scDFM on Norman; 8 config/code inconsistencies found; sampler sweep |
| `experiments/A100_BENCH.md` | training/generation speed and memory on an A100 80GB |
| `experiments/STAGE3.md` | cross-cell-line experiments, held-out-H1 validation, what failed |
| `experiments/LEADERBOARD_RESEARCH.md` | what the 2026 leaderboard shows; context A is a T-cell line |
| `experiments/DATA_AND_METHODS.md` | datasets, licences, cell construction, submissions, compute (for the finalist description) |
| `vcc/` | multi-cell-line scDFM (`train`, `predict`, `prepare`), pseudobulk stats, effect predictors (`effects.py`), diagnostics |
| `tools_snapshot/` | data extraction, LOCO scoring (`loco.py`), queues, `vccsub/` submission writer + builder |
| `remote/` | scripts for training on a rented GPU |

Not included: data, checkpoints, logs, credentials.
Environment: `requirements-repro.txt` (Python 3.11); scoring uses `cell-eval2` in a separate venv.
