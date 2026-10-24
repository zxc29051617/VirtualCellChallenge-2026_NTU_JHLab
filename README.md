# Virtual Cell Challenge 2026

> ## 從這裡開始
>
> 這個 repo 有兩部分：
>
> | 位置 | 是什麼 | 要不要看 |
> | --- | --- | --- |
> | [`general_model/`](general_model/) | 期末使用的通用模型 AdaptNet（v5，2026/10/24 上傳，H1 模擬考 0.306）；驗證期用其中的共識模型拿到公開排行榜 0.239（第 165 名／1,635 隊） | **先看這個**：[general_model/README.md](general_model/README.md) 寫給初學者，從背景知識、評分方式、模型、結果到怎麼從零重跑都有 |
> | 其他所有檔案 | 隊友最早的基準流程（最佳 0.109） | 保留作紀錄，可以先跳過 |
>
> 本頁以下是舊流程的說明（英文）。


---

## 舊流程（最早的基準，保留作紀錄）

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

- Old-pipeline notes: [docs/state_of_play.md](docs/state_of_play.md) — the submitted scores, the
  next experiment and why, and what has already been ruled out with the measurement that
  ruled it out. Best score so far is **0.1086** with `transfer_v9_hct116.vcc` (v5 scored 0.1040).

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
  requires an external public perturbation dataset — see `papers/README.md`.
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
- `scripts/fetch_xatlas_effects.py` and `scripts/fetch_hpsc_effects.py` pull per-target effects for
  the Challenge's targets from the HCT116 (X-Atlas/Orion) and stem-cell screens. Pass them to
  `build-transfer --secondary <a.npz> <b.npz> --secondary-weight 0.10 0.25`. v9 does this.
- `scripts/eval_source_pairs.py` and `scripts/eval_third_source.py` measure how well screens agree
  and whether one improves a blend (findings section 16).
- `scripts/build_bulk_profiles.py` rebuilds the K562/RPE1/K562gw profiles from figshare's
  pseudobulk files (80-375 MB) instead of the 1-9 GB per-cell files.
- `scripts/run_state_probe.py` runs Arc's STATE on a Challenge context (findings section 17).
- `scripts/extract_scgpt_embeddings.py` caches scGPT's gene embeddings (see below).
- `scripts/eval_neighbors.py` and `scripts/eval_gene_axis_fill.py` measure whether those
  embeddings carry usable signal. Read `docs/findings.md` section 15 before enabling them.

### Optional: scGPT features and FAISS retrieval

`build-transfer` can borrow effects from the genes nearest a target in scGPT's embedding space,
searched with FAISS. It is **off by default**, and the default recipe is unchanged. The
measurements so far show little cross-cell-type signal; section 15 of `docs/findings.md` has
the numbers and what they do and do not support.

```powershell
# Deliberately not in pyproject.toml: torch would add every platform's CUDA packages to uv.lock
# for a feature that is off by default. torch is only needed once, to cache the embeddings.
uv pip install faiss-cpu torch

# scGPT "whole-human" checkpoint: public bucket, no credentials (~205 MB)
#   https://czi-scgenept-public.s3.amazonaws.com/models/pretrained/scgpt/{best_model.pt,vocab.json,args.json}
# Put the three files in external/scgpt/, then cache the gene embeddings once:
uv run python scripts/extract_scgpt_embeddings.py

uv run vcc2026 build-transfer --neighbor-embeddings external/profiles/scgpt_gene_embeddings.npz `
  --neighbor-k 20 [--neighbor-weight 0.1] ...   # plus the usual v5 flags
```

Without `--neighbor-weight`, neighbours only fill targets no screen measured (14 of 300 under
v5's sources); with it, they are also mixed into measured targets.

Example packaging call:

```powershell
.\scripts\package_submission.ps1 -InputH5ad artifacts\baseline_prediction.h5ad -Genes dataset\gene_names.csv -Perts dataset\pert_counts.csv -OutputVcc artifacts\baseline_prediction.vcc -DryRun
```

## Notes

- The local `dataset/gene_names.csv` includes a header row; the official evaluation tooling expects a headerless gene list. The helper code strips that header when needed.
- The challenge submission must satisfy the official rules, including exact context labels, exact perturbation panel, raw integer counts, and no control cells.
