# Survey: public perturbation data like the final test lines (2026-10-25)

Question (user, 10-25): is there public perturbation data in cells like D/E/F, and how many of the 300 final targets does
it cover? Gene lists behind every number: `data/research_genelists_def/` (one HGNC symbol per line; recompute with
`python3 -I tools/target_overlap.py data/research_genelists_def`). Three parallel literature/data searches plus checks
here; all overlaps below were recomputed from the downloaded lists.

## 1. What the test lines are (DepMap 24Q4 expression, `pertrelay/model/depmap_match.py`)
Control pseudobulks against 1,673 CCLE RNA-seq profiles, Pearson correlation on the 3,000 most variable genes, both centred
on the CCLE mean.

| context | best match | r | runner-up |
| --- | --- | --- | --- |
| A | Jurkat (T-ALL) | 0.87 | PF-382 0.75 |
| B | HeLa | 0.70 | YD-8 0.38 |
| C | CAL-33 (head and neck squamous carcinoma) | 0.76 | CAL-27 0.64 |
| D | **MOLM-13** (AML, MLL-AF9, FLT3-ITD) | 0.86 | MONO-MAC-1 / OCI-AML2 0.78 |
| E | **U-87 MG** (glioblastoma) | 0.81 | U-118 MG 0.58 |
| F | no single line; non-cancerous mesenchymal stromal / fibroblast lines | 0.66 | many at 0.61-0.65 |

## 2. Does the line's own CRISPR dependency predict response size? (`pertrelay/model/dep_magnitude.py`)
DepMap Chronos gene effects vs the noise-corrected trans-response energy of each knockdown in the training lines that are
DepMap lines: Spearman with the line's own effect 0.08-0.42, with the mean effect over all lines 0.11-0.43, and the line's
own effect beyond the mean -0.03 to +0.11 (K562 -0.03, K562ess +0.07, VIPK562 +0.03, HepG2 +0.11, Jurkat -0.02, HCT116
-0.02). The pan-line version was already worth little (10-20 run.log), so knowing MOLM-13 / U-87 MG adds no usable
magnitude signal. For the record: 15 targets are MOLM-13-specific dependencies (MED25, MTF1, UROS, SUPT7L, PLAGL2,
SLC52A2, BRPF1, VEZF1, SMARCD1, PAPOLA, HDAC8, COQ10B, LDLR, RBM5, S100A11) and 5 are U-87 MG-specific (E2F1, TIMM50,
EIF4ENIF1, SMARCC2, RBM18), all moderate (gene effect -0.5 to -1.0).

## 3. Datasets (overlap with the 300 D/E/F targets)
| dataset | cells | modality | overlap /300 | size of the data | note |
| --- | --- | --- | --- | --- | --- |
| IGVF Engreitz TeloHAEC genome-wide CC-Perturb-seq (released 2026-10-12, IGVFDS8166LJMC) | hTERT aortic endothelial; resting, IL-1b, oxLDL x 2 reps | CRISPRi, high MOI (~4.8 guides per cell) | **300** | 6.11 M cells; ~530 cells per target per condition; merged h5ad 199 GB, per condition 35-87 GB; public S3, ~13 MB/s here with 32 connections | the only new resource with every target; endothelial, not like D/E/F |
| IGVF Gersbach WTC-11 TF screen | iPSC | CRISPRi | 78 | IGVF portal | not test-like |
| LINCS 2020 L1000 CRISPR-KO, U-251 MG | glioblastoma (like E) | KO, bulk L1000 | 75 | ~2 signatures per target | LINCS sources gave nothing on 10-19 |
| Norman Hs27 TF atlas (Nat Genet 2025) | foreskin fibroblast (like F) | **CRISPRa** | 72 | 2 GB | activation, wrong direction |
| Pan et al. 2026 (GSE295214) | H1 hESC | CRISPRi | 56 | 7.8 GB | not test-like |
| PerturbPair (Kudo/Regev 2026, GSE312107) | mouse bone-marrow macrophages | KO | 36 | 3.5 GB | mouse |
| IGVF Quertermous HCASMC | coronary smooth muscle | CRISPRi | 33 (24 with confirmed knockdown) | 87 GB | |
| LINCS shRNA, ASC52telo | adipose MSC (like F) | shRNA, bulk L1000 | 32 | ~3 signatures per target | |
| McFaline-Figueroa 2024 (GSE225775) | A172, T98G, **U-87 MG** | CRISPRi (+ drugs) | 21 kinases | ~70 cells per gene and condition | the only single-cell data in E's line |
| Jung 2026 (GSE327644) | primary human macrophages | KO | 20 | 6.1 GB | |
| Yao 2024 (GSE221321) | THP-1 | KO | 18 | | |
| Li 2026 ADAM12 CAF screen (GSE278990) | colorectal fibroblasts | CRISPRi | 13 | 2 GB | |
| Aryal 2026 (GSE294098), Otto 2023 (GSE200201) | **MOLM-13** | KO | 3 and 2 | 0.6 / 1.2 GB | the only Perturb-seq in D's line |
| PerturbAtlas bulk RNA-seq aggregator | MOLM-13 1, U-87 MG 1, AML lines 5, GBM lines 4, fibroblast/MSC 8 | KD/KO bulk | small | API | single-gene experiments |

Not available: Xaira X-Atlas/Pisces (Hugging Face repo empty), Illumina Billion Cell Atlas (pharma partners only), Arc's
VCC 2026 screens, CZI Billion Cells (only the CD4 data so far). Union of all human myeloid single-cell perturbation data:
26/300; with mouse myeloid data through orthologs 96/300.

## 4. Verdict
No public data covers 50 or more of the 300 targets with knockdown or knockout in cells like the test lines. In the test
lines themselves there are 5 targets for MOLM-13 (knockout), 21 kinases for U-87 MG (CRISPRi, shallow) and nothing for a
fibroblast line. The one large new resource is the IGVF TeloHAEC screen (all 300 targets, CRISPRi, three cell states), in
an endothelial line; adding it would mean a 35-87 GB download per condition and high-MOI deconvolution, about two days
with validation, for a gain the transfer ceiling (`ceiling_check.py`) suggests is small on non-essential targets.
