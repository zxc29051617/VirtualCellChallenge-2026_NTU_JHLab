# Public Perturbation Data Notes

> **Superseded.** This content has been merged into `參考文獻/README.md`, which sits next to
> the PDFs and carries the verified figures for the dataset we actually selected
> (Replogle 2022 via scPerturb). The `待確認` items below were resolved there on 2026-09-14.
> Kept only so you can confirm nothing was lost before deleting this file.

This is a working shortlist for pretraining or representation learning.
Anything marked `待確認` still needs exact counts, licensing, or download-path verification before use.

## 1. Virtual Cell Challenge 2025 atlas

- Source: Arc Virtual Cell Atlas / official challenge repository.
- Cell type: H1 human embryonic stem cell.
- Perturbations: CRISPRi gene perturbations.
- Scale: ~300,000 cells in the challenge atlas.
- Assay: single-cell RNA-seq.
- Suitability: direct perturbation-response pretraining, but highest leakage risk if we accidentally reuse challenge-specific validation or held-out data.
- License / usage: public/open according to Arc materials, but exact redistribution terms should be rechecked before use.

## 2. Replogle et al. 2022 genome-scale Perturb-seq

- Source: Cell 2022 / genome-scale Perturb-seq resource.
- Cell types: K562 and RPE1.
- Perturbations: CRISPRi gene perturbations.
- Scale: genome-scale perturbation sets; exact cell counts `待確認`.
- Assay: droplet-based scRNA-seq / Perturb-seq.
- Suitability: strong candidate for perturbation-response pretraining and gene-perturbation priors.
- Leakage: different cell types from the challenge, so direct leakage is low.
- License / usage: public resource, but the exact usage terms on the hosting portal should be confirmed.

## 3. Norman et al. 2019 Perturb-seq

- Source: Science 2019 / associated public dataset mirrors.
- Cell type: K562.
- Perturbations: CRISPR-based genetic interaction / perturbation screens.
- Scale: `待確認`.
- Assay: Perturb-seq / scRNA-seq.
- Suitability: useful for general perturbation priors and response manifolds.
- Leakage: low direct leakage, but older chemistry and different screen design.
- License / usage: public mirror exists, exact terms `待確認`.

## 4. Tahoe-100M

- Source: Arc / Tahoe public dataset.
- Cell types: 50 cancer cell lines.
- Perturbations: 1,100 small-molecule perturbations.
- Scale: 100 million transcriptomic profiles.
- Assay: single-cell transcriptomics.
- Suitability: very good for context-aware representation learning, weaker as a CRISPRi-specific pretraining set.
- Leakage: no direct challenge leakage if we avoid challenge test data, but the perturbation modality is mismatched.
- License / usage: public dataset on multiple hosts; exact access terms should be checked before download.

## 5. scBaseCount

- Source: Arc Virtual Cell Atlas.
- Scope: curated open single-cell data for atlas-scale learning.
- Scale: Arc states the atlas exceeds 602 million cells overall; exact perturbation subset for scBaseCount `待確認`.
- Assay: mixed observational and perturbational single-cell data.
- Suitability: good for representation learning and broad cell-state priors.
- Leakage: depends on which subset is selected.
- License / usage: public/open in Arc materials, but exact reuse conditions should be rechecked.

## Decision rule

Prefer datasets in this order for the first external-data pass:

1. Challenge-safe perturbation-response data with explicit open terms.
2. Replogle-style CRISPRi perturbation data.
3. Public atlas-scale representation data such as Tahoe-100M or scBaseCount.
4. Anything else only after we verify licensing and modality fit.

