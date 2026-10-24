# Final phase (D/E/F), started 2026-10-22

## What changed on 10-22
- vcc-cli 0.3.0 (required for final submissions). `vcc whoami`: phase final, contexts D/E/F, panel vcc2026-test-1,
  deadline 2026-11-05 23:59 UTC. Scores and ranks hidden until the winners are announced; 2 final entries per UTC day;
  only the last one counts; never submit a sample.
- A .vcc made without the 0.3.0 meta (contexts) uploads and fails at scoring. vccsub/pack_vcc.py now writes contexts,
  perts_fingerprint, panel_id and cli_version from the manifest next to --perts; on a tiny D/E/F file its meta and
  content equal `vcc prep` output exactly.
- controls-final -> data/vcc2026_final (context_D/E/F.h5ad, 18,400 controls each, same 18,533 genes). 300 targets,
  129 shared with A/B/C, 171 new.
- The A/B/C ground truth is not published (virtualcellchallenge.org, the CLI wiki and cell-eval2 GitHub checked).

## Source coverage for the final panel
The 'vcc'-kind sources had been extracted for the A/B/C targets (+1,000 auxiliary genes) only, so most new targets had
1-3 sources (A/B/C targets: 299/300 in >= 3). Filled (backups of the old stats: data/stage3/stats_abc_backup_20261022):
- K562 gwps: from the kept bulk file (mean UMI x filtered cells = the single-cell sums exactly): 148 -> 277/300.
- X-Atlas HEK293T / HCT116: re-extracted the 157 missing targets (all guide-passing cells) from Hugging Face.
  HEK293T 158 -> 298/300.
- CD4 (Marson pseudobulk, 44.6 GB on S3): full download with 128 parallel range requests (~11 MB/s; one connection
  is ~0.1 MB/s here and h5py chunk lookups over HTTP take minutes each), then tools/cd4_add_remote.py --local.
Groups that existed before are unchanged (max diff 0); aux genes extracted with 64 cells were replaced by all cells.

## Next
1. Rebuild the signature cache (cache_contexts.py now adds the final targets to T).
2. Export check: A/B/C effects before vs after the stats additions (should stay very close).
3. D/E/F file with the rehearsed config (= G3c, 0.2386 on A/B/C):
   RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5 CELLAGREE=1 TWPOW=0.5 ENTRYAGREE=1 tools/final_pipeline.sh data/vcc2026_final <name> D E F
   CISSCALE=3 stays out (never checked on a leaderboard; small expected gain).

## 10-22 evening: first D/E/F file
- Signature cache rebuilt (T 633 incl. the final targets): unchanged sources identical, filled sources corr >= 0.9997 on
  old rows; the A/B/C export after the fill-in matches VCC_gm_vip (per-target corr median 0.9999, min 0.9997).
- vccsub/out/gm_final_v1.vcc = the G3c configuration on D/E/F: 300 targets, every one with cache rows (median 8 sources),
  totals claim 4312 / 4562 / 4533 and s_c 0.0104-0.0109 (A/B/C: 4311 / 4373 / 4751, 0.0106-0.0111), nnz 2.01e9;
  D/E/F controls are as deep as A/B/C (~21k UMI, ~5.5-5.7k genes per cell). Validated with contexts D/E/F recorded.
