# Data, methods and compute record (Virtual Cell Challenge 2026)

For the finalist description the rules require: datasets used in training, processing, models, **non-learned
components including anything blended with model outputs**, training/inference approach, compute.
Updated 2026-10-13 for the general model (`vcc/gm.py`), which every submission since 10-11 uses.

**Historical record (2026-10-13).** It describes the consensus model of the A/B/C phase (50 control axes, no trained
network). The final D/E/F entry gm_final_v5 adds AdaptNet gene totals (3 seeds, RPE1 as an extra source), 5 control
axes and the cell rendering; the current description is `general_model/README.md` (sections 5-7 and 11.8). The data
table below still holds (Feng counts: figshare 27989294 v2, MIT, checked 2026-10-24).

## 1. Training data (public CRISPRi screens; no challenge perturbation data exists or is used)
| context(s) | dataset | perturbations used | source | licence |
| --- | --- | --- | --- | --- |
| K562 | Replogle et al. 2022, genome-wide (GWPS) | VCC targets + 1,000 auxiliary genes (selected cells) | figshare+ 20029387 | CC BY 4.0 |
| K562ess | Replogle et al. 2022, essential-gene screen | all | figshare+ 20029387 | CC BY 4.0 |
| HCT116, HEK293T | X-Atlas/Orion (Xaira) | VCC targets + auxiliary genes | figshare+ 29190726 / HF `Xaira-Therapeutics/X-Atlas-Orion` | **CC BY-NC-SA 4.0** |
| CD4_Rest/Stim8hr/Stim48hr | Zhu, Dann et al. (Marson lab) primary CD4+ T-cell CRISPRi, pseudobulk | VCC targets + others in the pseudobulk file | s3://genome-scale-tcell-perturb-seq | MIT (per dataset metadata) |
| H1 | Arc Virtual Cell Challenge 2025, H1 hESC (Training/Validation/Test) | all 300 | gs://arc-institute-virtual-cell-atlas | CC0 |
| HepG2, Jurkat | Nadig et al. 2025 | all (essential genes) | GEO GSE264667 | none stated |
| FengNF, FengF | Feng et al. 2025 HipSci iPSC (34 donor lines; non-fitness and fitness gene sets) | all | figshare 27989294 | MIT |
| KOLF | KOLF2.1J iPSC genome-scale CRISPRi (Mali lab, Nat Biotech 2026), raw-count layer | all 11,687 | figshare+ 27261219 | CC BY 4.0 |
| VIPK562 | VIPerturb-seq (Satija lab 2026), K562, 10x Flex, filtered object | all 6,724 | Zenodo 18460279 | CC BY 4.0 |
Processing: per context, raw counts summed per target gene (pseudobulk) and over non-targeting controls (Feng: cells
without a guide call, as few NTCs were captured), on the challenge's 18,533-gene axis matched by symbol
(`vcc/stats.py`, `vcc/stats_ext.py`, `tools/kolf_process.py`, `tools/viperturb_stats.py`). Gene positions: GENCODE v46.

## 2. Model (learned components)
General model v1 (`vcc/gm.py`). For target p in a context with control cells, the predicted log fold change is
own-gene knockdown + CRISPRi cis effect on genes within 50 kb of p's TSS + ALPHA x a trans signature of p:
- **cis kernel** beta (per TSS-distance bin and for head-to-head partners): least squares of neighbour log ratios on
  the target's own log ratio, pooled over all training contexts;
- **trans signature** s_p: ridge estimate (lambda 3) across training contexts of each context's log ratios with that
  context's mean response over perturbations removed, rescaled by the observed own knockdown (capped at 1),
  own and cis genes masked; contexts weighted equally (per_celltype=False).
At prediction time the model uses only the target name; signatures are learned parameters (no test-context
perturbation data). Settings chosen on held-out-cell-line checks and the validation leaderboard.

## 3. Non-learned components
- Own-gene knockdown fixed at ln r = -1.9 (~85 %; the challenge states every target has > 80 % knockdown).
- ALPHA = 3 amplitude on the trans part (chosen by validation, not learned by gradient descent).
- Cell construction (`vccsub/make_effect_submission.py`): per context and target, 400 of that context's control cells
  drawn without replacement; log2 fold changes clipped to ±4; decreased genes binomially thinned with p = fc;
  increased genes get Poisson counts with mean = (cell's original library) x (gene rate in that context's controls) x
  (fc - 1). Integer counts, no controls submitted.
- Exact group totals (from 2026-10-19, `scDFM/vcc/render.py` mode "exact", `--render exact`): after the step above, each
  gene's total over a target's 400 cells is set to round(400 x mean count per control cell x fc) by adding the few
  residual counts in proportion to library size or removing them uniformly from the existing counts, so the
  pseudobulk equals the prediction (no Monte-Carlo noise from the control draw or the thinning). The model is unchanged.
- Shared-component removal in the gene totals (from 2026-10-19 evening, `vcc/gm.py export bulk_out=...`, `--render dual`):
  across the panel's targets, the mean and the top-10 principal components of the trans part are removed, and each
  gene's total over a target's cells is set to own + cis + that reduced trans part; the cells themselves are generated
  from the full prediction and the residual counts are added to / removed from the upper tail of the target's cells
  (`vcc/render.py` mode "dual"). Computed from the predictions only (no test data).
- Control-axis removal (from 2026-10-20, `tools/make_bulk_effects.py --ctrl-axes 50`): the top-50 principal axes of each
  context's own released control cells (log1p CP10k, 2,000 cells) are also removed from the totals' trans part.
- G1 totals (from 2026-10-21, `final_pipeline.sh` RENDER=g1): the totals' trans part comes from a second fit in which each
  (training context, target) row is weighted ncell / (ncell + 200) (GeneralModel n0=200); after the shared-component and
  control-axis removal each target's trans row is multiplied by |t_p|^2 (per-target sharpening, gamma 3: the measured
  responses are concentrated in few targets); totals = own + cis + s_c x trans with s_c chosen per context so that the
  across-target spread of the predicted pseudobulk means equals the expected sampling correction of a 400-cell
  pseudobulk of that context's controls (scorer cell-eval2 #348 budget = claim). Computed from predictions and the
  released controls only.
- Packaging (`vccsub/pack_vcc.py`): byte-equivalent content to `vcc prep` (raw counts untouched), streamed.

## 4. Validation
Leave-one-cell-type-out with the official cell-eval2 0.18.0 vcc2026 scorer (de.backend=pdex) on H1, HCT116 and
KOLF; local scores over-state the leaderboard (local context-mean baselines call few DE genes; A/B/C baseline
fidelity sits at chance). Leaderboard (val A/B/C, anchors r4): gm_v1_plain 0.1455, gm_v1_vip 0.1482; shuffled-signature
control -0.0124 (so the learned signatures carry target-specific information on A/B/C); gm_v1_vip with exact group
totals 0.1608 and with the shared-component totals (V6) 0.1808 (2026-10-19); V6 totals x0.5 0.2032 and with control axes
removed (F9) 0.2080 (2026-10-20). Details: `repro/GENERAL_MODEL.md`,
`repro/next_20261019/`.

## 5. Compute
Local 12-core PC, 15 GB RAM, RTX 4060 Ti (not needed by the general model). NVIDIA Brev (course credits): two A100
sessions and CPU sessions on 10-09/10, one n2d-highmem-8 session on 10-11 (~1 h) for packaging; about US$4.4 in total
(estimated from hourly prices; check the Brev billing page). Packaging is local since 10-11.

## 6. To check before a final entry
- X-Atlas/Orion is non-commercial (CC BY-NC-SA); confirm it is acceptable for a prize entry.
- Rules (revised 2026-09-16): literature results may not be submitted or used to revise predictions; training on them
  is allowed. Here public screens are training data for learned signatures; confirm with the organisers if unsure.
- The final `pert_counts.csv` / contexts for D/E/F may differ; `tools/final_pipeline.sh` checks the files and stops
  before uploading.
- Credentials used during the work must be revoked by the user when the competition ends (the VCC token is stored
  locally with the user's permission).
