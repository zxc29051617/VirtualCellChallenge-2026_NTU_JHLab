# Survey: virtual-cell models and their training data (2026-10-10)

Two research passes (models + datasets; other VCC teams). Other teams' methods are summarised in
`LEADERBOARD_RESEARCH.md`. Gene lists behind the overlap numbers: `data/external/research_genelists/`
(overlap script `tools/research_overlap.py`, exact HGNC match against the 300 VCC targets).

## Models (what matters for us)
- No published model shows strong zero-shot transfer of CRISPRi responses to an unseen line. Best numbers:
  PIE (Arc, Oct 2026; Perceiver-IO over gene knowledge + control means, predicts pseudobulk LFC; held-out line AUPRC
  0.424 vs perturbation mean 0.088, zero-shot H1 0.395 vs context mean 0.289), X-Cell (Xaira, private X-Atlas/Pisces),
  TxPert (Pearson delta 0.41-0.51 vs 0.37-0.48 general baseline), Rhaister-O (additive + ridge on summary statistics),
  COMPASS (shared response + line-specific residual). An independent benchmark found GEARS/CPA/scGPT/scFoundation etc. do
  not beat linear or mean baselines (PMC12328236).
- Predicting pseudobulk LFC from (target knowledge, control profile) is the common structure of the models that do
  transfer across lines; single-cell generation is a separate step.
- STATE / PIE filter training perturbations to target mRNA < 30 % of control; VCC 2026 kept targets with >= 80 % median
  knockdown, so a similar filter matches the test.

## Datasets and coverage of the 300 targets
| dataset | context | overlap /300 | size | licence | status |
| --- | --- | --- | --- | --- | --- |
| X-Atlas/Orion | HCT116, HEK293T | 300 / 300 (knockdown 75 % / 52 %) | parquet mirror | CC BY-NC-SA | have |
| Marson CD4 T (Cell 2026) | CD4 T, 4 donors x 3 states, Flex | 297 (267 significant knockdown) | pseudobulk 44.6 GB | MIT (VCP) | have (pseudobulk stats) |
| Replogle 2022 K562 GWPS | K562 | 272 | 65.8 GB | CC BY 4.0 | have |
| KOLF2.1J iPSC atlas (Nat Biotech 2026) | iPSC | 282, median 236 cells | 189 GB pan-genome h5ad (CSC, uncompressed, X + layers/counts) | CC BY 4.0 | new; see access note |
| Feng 2025 hiPSC | 34 iPSC lines | 166 non-fitness + 31 fitness | 1.75 GB (have) + 1.45 GB | MIT | partly have |
| IGVF TF Perturb-seq (Hon lab) | iPSC cardiomyocytes | 91 | 4.6 GB UMI h5 | not stated | new |
| Schnitzler 2024 | TeloHAEC endothelial | 66 | 1.5 GB IGVF h5ad | not stated | new |
| VIPerturb-seq (Satija, Flex v2) | K562 | 300, median 47 cells | 17.3 GB Seurat | CC BY 4.0 | new (assay-matched replicate) |
| VCC 2025 H1 | H1 hESC, Flex | 25 (300 perts) | 33 GB | CC0 | have |
| McFaline-Figueroa 2024 | 3 glioblastoma lines | 23 | 12 GB | none stated | optional |
| Jiang 2025 | 6 lines x 5 cytokines | 9 | 20.5 GB | CC BY 4.0 | optional (cross-line learning) |
| Replogle essential K562 / RPE1, Nadig HepG2 / Jurkat | 4 lines | 0 | have | CC BY / none | have (cross-line learning only) |
| X-Atlas/Pisces, Illumina Billion Cell Atlas | many | ? | - | - | not public |

KOLF access note: the pan-genome file is CSC (gene-major), so a subset of cells cannot be range-read; it has to be
streamed in full (~94 GB for `layers/counts`). figshare+ serves it from S3 eu-west-1 at ~0.3 MB/s per connection from this
machine (measured 2026-10-10), so a single-connection stream would take days; many parallel connections or a cloud
machine would be needed.
