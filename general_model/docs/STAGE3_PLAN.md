# Stage 3 — cross-cell-line CRISPRi, leave-one-cell-line-out (LOCO)

## Facts measured so far (2026-10-09)

- The 300 VCC validation targets are **not essential genes**: Replogle K562-essential and
  RPE1-essential cover 0/300; Replogle K562 genome-wide covers 272/300 (258 with >= 50 cells).
  Per-target counts: `data/coverage/replogle_cells_per_target.csv`.
- Replogle's released matrices keep only ~8,000 expressed genes (K562 gwps 8,248; RPE1 8,749),
  i.e. 41-45% of the VCC 18,533-gene axis.
- VCC control cells: 18,400 per context, median ~20k UMI, ~5,800-6,100 nonzero genes per cell.

## Consequence for the design

The only public data that perturbs the VCC targets is one cell line (K562 gwps). What
transfers across cell lines therefore has to be learned on perturbations that *are* shared
across lines — essential genes (K562, RPE1, HepG2, Jurkat; H1 if the VCC 2025 data is used) —
and then applied to the VCC targets, whose effects are known only in K562.

LOCO mirrors the VCC task: hold out cell line L entirely (its perturbed cells), give the model
L's control cells, and predict L's response to perturbations seen in other lines.

## Pipeline

1. Download raw single-cell files (running: `tools/fetch_stage3.sh`, log in
   `data/external/download.log`): Replogle K562-essential, RPE1; Nadig HepG2, Jurkat.
   K562 gwps (66 GB) and anything larger should be fetched straight into the GPUtw Vault.
2. Per dataset, write a compact h5ad: raw counts, VCC gene axis (genes the dataset did not
   measure flagged, not zero-filled), obs = dataset, cell_line, target_gene, is_control;
   at most N cells per perturbation and M controls, so all lines fit in memory together.
3. Model gene set = genes measured in every training line ∩ VCC axis, expressed in VCC
   controls. Genes outside it are filled from (resampled) control cells at output time.
4. Context-aware sampler: each batch = (controls of line L, perturbed cells of line L for
   gene g); the perturbation embedding must not be learned per line.
5. Baselines on the same LOCO splits, scored with cell-eval2 `vcc2026` (same six metrics):
   control resample; mean-shift transfer (held-out controls + the target's log-fold-change
   averaged over the other lines).
6. scDFM on the same splits. The claim to establish: it beats the transfer baseline on
   held-out lines.
