# Handoff 2026-10-11 16:06 Taiwan (08:06 UTC)

Repo `/home/zxc29051617/VCC/scDFM` (branch `vcc-repro`, local only). Detailed notes: `repro/GENERAL_MODEL.md`
(everything below is summarised from it), `repro/DATASET_SURVEY.md`, `repro/LEADERBOARD_RESEARCH.md`.
Credentials: the VCC token is stored by `vcc login` in `~/.config/vcc/credentials.json` (0600); Brev is logged out.
Do not print either.

## 1. Latest leaderboard submissions (val A/B/C, anchors `vcc2026-val{A,B,C}-r4`)
| model name | entry | avg | pds | mse | nmae | fid | reach | jac | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gm_v1_sim5 | v9jl2r0nAxts0dRe50tU | 0.1415 | 0.515 | 0 | 0.116 | 0.025 | 0.185 | 0.007 | 538 |
| gm_v1_plain | FXD3pdX9SuLnunpNkYp3 | 0.1455 | 0.525 | 0 | 0.143 | 0.021 | 0.178 | 0.006 | 523 |
| cd4_ctxw (10-10, best so far) | – | 0.1485 | 0.555 | 0 | 0.130 | 0.016 | 0.184 | 0.005 | ~490 |
| transfer_x1p5 (10-10) | – | 0.1274 | 0.552 | 0 | 0.064 | -0.007 | 0.156 | -0.001 | ~581 |
Raw values of gm_v1_plain: pds 0.737, mse_norm 4.26, nmae 0.914, fid 0.519, reach 0.237, jac 0.0328.
Anchors solved from the two gm submissions (score = (u-b)/(a-b), assumes linear per-metric scaling):
pds b 0.467 a 0.982; fid b 0.513 a 0.803; reach b 0.083 a 0.949; jac b 0.030 a 0.414; nmae b 1.00 a 0.385;
mse (from other teams' entries) b 0.986 a 0.038, floored at 0.
Files: effects `data/stage3/effects/VCC_gm_v1_a3.0_t5_{A,B,C}.npz` (sim5), `VCC_gm_v1_a3.0.npz` (plain);
.vcc files were built on Brev and deleted with the instance. Public leaderboard dump: `~/.claude/jobs/3cf1662b/tmp/lb.json`.

## 2. General model (`vcc/gm.py`), evaluation, commands
Method: ln r = OWN (target, -1.9) + OWN*beta(TSS distance bin, head-to-head) for genes within 50 kb (cis, beta by least
squares across contexts) + ALPHA * s_p (trans: ridge average across contexts of common-mode-removed log ratios,
knockdown-rescaled, cap KCAP). Submitted config: per_celltype=False, lam=3, kcap=1, alpha=3 (+ sim_tau=5 for sim5).
Contexts (`vcc/contexts.py`, stats in `data/stage3/stats{,_ext}/`): K562, HCT116, HEK293T, H1, CD4 x3, K562ess, HepG2,
Jurkat, FengNF, FengF, KOLF. Cells built by `vccsub/make_effect_submission.py` (thin/Poisson from context controls).

LOCO, official scorer (`tools/loco.py` prep/pred/score; scorer cell-eval2 0.18.0 in `evalenv`, `de.backend=pdex`):
| held out | old transfer | gm plain | gm sim5 |
| --- | --- | --- | --- |
| H1 | 0.267 | 0.352 | 0.354 |
| HCT116 (VCC targets) | 0.218 | 0.245 | 0.301 |
| KOLF200 (200 cells/target) | 0.135 | 0.163 | 0.142 |
| KOLFbig (400 cells, 20k controls) | – | 0.187 | – |
KOLFbig decomposition: own+cis only 0.023, shuffled trans 0.136, real trans 0.187. Leaderboard gains did NOT follow
LOCO (sim5 < plain on A/B/C).

Commands (run from `/home/zxc29051617/VCC`):
- export effects for a held-out set: `cd scDFM && .venv/bin/python -m vcc.gm export <H> data/stage3/loco/<H>/real.json ../data/stage3/effects/<H>_<name>.npz per_celltype=False lam=3.0 kcap=1.0 alpha=3.0`
  (use absolute paths; for challenge contexts pass `A vcc <out.npz>`)
- score: `tools/queue_gm_loco.sh <H> <name>` (KOLFbig needs `LOCO_PRED_ARGS="--n-pred 400"`); prints RESULT avg, raw mse, raw pds
- adjusted re-score: `evalenv/bin/python tools/adjusted_score.py data/stage3/loco/<H>`
- build + pack locally (low memory): `cd vccsub && ../scDFM/.venv/bin/python -I make_effect_submission.py --data ../data/vcc2026 --effects <npz | A=.. B=.. C=..> --out out/x.h5ad`
  then `~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/x.h5ad out/x.vcc --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv`
- heavy jobs go through the serial runner: append a line to `data/stage3/jobs.txt` (`tools/runq.sh`, log `data/stage3/runq.log`).

**`tools/adjusted_score.py` is a heuristic calibration, not the official A/B/C anchors.** It replaces only the local
fidelity baseline with max(local, 0.5) — 0.5 approximates chance-level sign agreement when n_pred >= N_conf, chosen
because the derived A/B/C fid baseline is 0.513. All other anchors stay local (bundle baseline + split-half replicate).

## 3. Background work and schedule
- `tools/runq.sh` (pid 210031): idle, 60/60 jobs done.
- `tools/submit_queue.sh` (pid 417071), log `data/stage3/submit_1012.log`: waits until 2026-10-12 00:02 UTC, then
  `vcc submit --wait` of `vccsub/out/gm_cd4only.vcc` as gm_v1_cd4trans (trans from CD4 only), then
  `vccsub/out/gm_shuf.vcc` as gm_v1_shuffled (negative control). Both validated by vcc's `validate_vcc`.
  `vccsub/out/gm_owncis.vcc` exists but is no longer scheduled. Daily limit 2, reset 00:00 UTC.
- No training, downloads or Brev instances running.

## 4. Verified vs hypothesis
Verified:
- Bugs fixed: KOLF h5ad had explicit zeros (cell-eval2 refused; `tools/fix_zeros.py`, fixed in `kolf_process.py`);
  `-I` broke module imports in queues; chained `cd scDFM` twice in runner lines; OOM kills when jobs overlapped (runner
  introduced); a wait loop matched the wrong log line and validated a half-written .vcc (file itself was fine).
- `pack_vcc.py` output equals `vcc prep` (same inner h5ad content/dtypes/obs/var, tar headers, meta.json) on a 3-target
  test; full-size packs pass `validate_vcc`. Not yet submitted once — tomorrow's uploads are its first server test.
- Scorer facts (cell-eval2 source): mse and pds exclude the target gene; fid_yield_raw = k/max(n_pred, N_conf) where k
  counts sign agreement with any non-zero real lfc; mse member floored at 0 with noise correction capped and budgeted.
- Local context-mean baselines have raw fid 0.004-0.065 vs 0.513 on A/B/C, even at 400 cells/20k controls.
- Per-target correlation of predicted vs true pseudobulk deltas for VCC targets in held-out KOLF/CD4 ≈ 0.01;
  projection onto control PCs and lower trans amplitude reduce raw mse but lower the total.
Hypotheses (not verified):
- A/B/C carry a consistent perturbed-vs-NTC shift that our public data lack (cause unknown).
- Top teams' mse ~0.45 comes from real predictive signal unavailable to us; their descriptions suggest bulk/DE decoupling,
  agreement-weighted amplitude and control-PCA cleaning.
- Flex-chemistry sources (CD4) transfer better than 3'/5' sources (tested tomorrow by gm_v1_cd4trans).
- The anchors derived above assume linear scaling identical across submissions.

## 5. Addendum: loco.py fix by Codex audit (2026-10-11)
- `tools/loco.py` (local check-set prediction step) was corrected by Codex. Up-regulated genes now get Poisson counts
  scaled by each sampled cell's ORIGINAL library size, the same as `vccsub/make_effect_submission.py`. Previously it used
  the total left after down-regulation thinning. When the control pool is large enough, cells are now drawn without
  replacement. Backup of the original, plus the loop tests, are in `repro/codex_audit_20261011/`. In a mixed down/up test, the mean of an
  up-regulated gene was 169.86 before the fix and 200.25 after (expected 200).
- Limitation: every LOCO score in this handoff and in `repro/GENERAL_MODEL.md` (H1, HCT116, KOLF, KOLF200, KOLFbig,
  including the adjusted re-scores) was produced with the OLD loco.py. That means up-regulation was understated, and
  cells may have been sampled with replacement. So local predictions were rendered differently from the submitted
  files built with make_effect_submission.py. These numbers have not been recomputed, and comparisons between
  local and leaderboard results inherit this difference. Leaderboard results and the files prepared for 10-12
  (built with make_effect_submission.py) are unaffected; the upload schedule is unchanged.
