# Notes on the two papers (read in full, 2026-10-10)

## scDFM (Yu et al., ICLR 2026, arXiv 2602.07103; the README's first badge, 2601.01829, is a different paper)
- Task: **one cell line** (Norman, K562, CRISPR *activation*, ~100 singles + 124 doubles, ~400 cells each);
  predict unseen double combinations (additive split) or unseen singles + their combinations (holdout split).
- Training (A.4.3): Adam 5e-5 cosine to 1e-6, **batch 96, 100,000 steps**, H800 GPUs, MMD weight 0.5, kNN k=30 signed
  co-expression mask, d=512, 4 layers, 8 heads, **Euler with 100 steps at inference**. For CRISPR perturbations the
  perturbation embedding of gene g *is* the gene-identity embedding E_g(g) (no separate table).
- Each step samples one perturbation: ~100k/189 = ~530 exposures per training perturbation. Our token run: 20k steps / 4,565
  (line, perturbation) pairs = ~4. Our batch 16 x 20k = 0.32M cells vs 9.6M (30x less).
- Baselines: the Additive baseline (sum of single effects) is hard to beat: additive split DS 0.9686 (Additive) vs
  0.9737 (scDFM); MSE 0.00448 vs 0.00315. The intro cites Systema/Ahlmann-Eltze: predicting the population mean often beats
  sophisticated models. Foundation models (scGPT, Geneformer) and STATE (from scratch) do worse.
- Ablation (holdout, Pearson delta 0.773): dropping MMD hurts most (-7.2%); gene mask and differential attention -2.4% each.
- Limitations (A.6) say it was **not tested on multi-context / multi-cell-line data** (ARC-State was released during their
  work) -- the cross-cell-line zero-shot task of VCC 2026 is outside what the paper shows.

## PerturbCellRL (Wu, Li, ... Fox, arXiv 2606.27752)
- Post-trains a pretrained scDFM with RL (DiffusionNFT): 200 optimizer updates, lr 5e-7, rollouts RK4-20.
- Rewards: energy-distance witness (population alignment), expression-support (calibrated radial deviation), pathway
  discrimination (Hallmark activities + L1 logistic classifier per condition). Calibrated on real cells of *training*
  perturbations; test perturbations need no real cells.
- Results on Norman (additive / holdout) and ComboSciPlex vs scDFM: energy -6.8..-13.1%, MMD -14.7..-20.8%, pseudobulk MAE
  and Pearson slightly better; **discrimination score worse in 3 of 4 settings** (e.g. additive 0.9743 -> 0.9558).
- Same single-cell-line setting as scDFM; no cross-cell-line evaluation. Not useful for our pds/mse gap; revisit only
  after a base model learns perturbation-specific effects.
