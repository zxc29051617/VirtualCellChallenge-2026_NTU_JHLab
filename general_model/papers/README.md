# 論文與資料來源

這裡放這個專案**真的讀過、或實際用到**的論文。每一篇都寫了我們怎麼用它、結論是什麼，以及詳細紀錄在哪裡。
PDF 只供隊內閱讀，引用時請用下面的正式出處。
repo 根目錄的 [`papers/`](../../papers/) 是隊友早期整理的版本（同樣有其中 4 份 PDF，另外有他當時的資料選擇紀錄）。

## 1. 讀過的論文

| 檔案 | 論文 | 我們怎麼用 | 結論 | 紀錄 |
| --- | --- | --- | --- | --- |
| [`scDFM_Yu2026_ICLR.pdf`](scDFM_Yu2026_ICLR.pdf) | Yu C, Wang C, Liao B, Wu T. **scDFM: Distributional Flow Matching Model for Robust Single-Cell Perturbation Prediction.** *ICLR* (2026). arXiv:2602.07103 | 專案的起點：先在原論文的資料（Norman, K562）重現，再改成多細胞種類版本，拿來預測沒看過的細胞 | 在沒看過的細胞上學不到「關掉哪個基因」的訊號（區分度 0.50，等於亂猜），所以改走小模型 AdaptNet | `pertrelay/experiments/STAGE1.md`、`STAGE3.md`、`PAPERS.md` |
| [`PerturbCellRL_Wu2026_arXiv.pdf`](PerturbCellRL_Wu2026_arXiv.pdf) | Wu D, Li M, Lundberg E, Zhang Y, Yeung-Levy S, Kumar A, Fox EB. **PerturbCellRL: Aligning Distributions and Grounding Biology via Post-Training Perturbation Generators.** arXiv:2606.27752 (2026) | 評估能不能用強化學習進一步微調 scDFM | 這篇和 scDFM 一樣只測單一細胞種類，沒有跨細胞的結果，而且區分度在 4 個設定中有 3 個變差；在基礎模型學到擾動訊號之前用不上，沒有採用 | `pertrelay/experiments/PAPERS.md` |
| [`SIGnature_Gold2026_NatBiotechnol.pdf`](SIGnature_Gold2026_NatBiotechnol.pdf) | Gold MP, Reyes M, Diamant N, …, Heimberg G, Biancalani T. **Scoring gene importance by interpreting single-cell foundation models.** *Nature Biotechnology* (2026). doi:10.1038/s41587-026-03112-5 | 用它的「基因重要性分數」當作判斷來源可信度的特徵，在 3 個留出細胞 × 3 個種子 × 5 組設定上做了 45 次官方評分 | 沒有比原本的表現量特徵更好，也沒有贏過打亂分數的對照組，放棄 | `pertrelay/experiments/signature_20261013/SUMMARY.md` |

## 2. 比賽的官方文件

| 檔案 | 文件 | 用途 |
| --- | --- | --- |
| [`VCC2026_Cell2026_commentary.pdf`](VCC2026_Cell2026_commentary.pdf) | Arc Virtual Cell Initiative Team & Goodarzi H. **Virtual Cell Challenge 2026: Benchmarking zero-shot generalization across cellular contexts.** *Cell* 189, 5827–5830 (2026). doi:10.1016/j.cell.2026.08.004 | 比賽的設計和規則。其中寫明「Participants are encouraged to use any appropriate data and modeling strategy」，也就是可以用公開資料 |
| [`vcc2026_metric_suite_official_spec.pdf`](vcc2026_metric_suite_official_spec.pdf) | **The vcc2026 Metric Suite**（官方評分規格，2026/08/19；來源 github.com/ArcInstitute/cell-eval2） | 六項分數怎麼算、怎麼換算成 0～1；本機評分和快速試算都以它為準 |

## 3. 用到資料的論文

| 檔案 | 論文 | 用到的資料 |
| --- | --- | --- |
| [`Replogle2022_Cell_genome-scale_Perturb-seq.pdf`](Replogle2022_Cell_genome-scale_Perturb-seq.pdf) | Replogle JM, et al. **Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq.** *Cell* 185, 2559–2575 (2022). doi:10.1016/j.cell.2022.05.013 | K562（全基因組、必需基因）和 RPE1 的 CRISPRi 實驗 |
| [`scPerturb_Peidli2024_NatMethods.pdf`](scPerturb_Peidli2024_NatMethods.pdf) | Peidli S, Green TD, Shen C, et al. **scPerturb: harmonized single-cell perturbation data.** *Nature Methods* 21, 531–540 (2024). doi:10.1038/s41592-023-02144-y | 從它整理好的資料庫取得神經細胞的 CRISPRi 實驗（TianKampmann 2021）做測試；沒有幫助，期末模型沒有用 |

其他資料沒有附 PDF，出處如下（用法見主說明文件第 5 節和第 11.5 節）：

| 資料 | 出處 |
| --- | --- |
| HepG2、Jurkat | Nadig et al., *Nature Genetics* (2025)；GEO GSE264667 |
| HCT116、HEK293T | X-Atlas/Orion（Xaira Therapeutics），<https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion>（CC BY-NC-SA 4.0，不能商業使用） |
| H1 | Virtual Cell Challenge 2025 資料，Arc Virtual Cell Atlas |
| CD4 T 細胞 | Marson lab 全基因組 CRISPRi，<https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/> |
| FengNF、FengF | Feng et al. (2025)，figshare 27989294 |
| KOLF | KOLF2.1J 全基因組 CRISPRi（*Nature Biotechnology* 2026），figshare+ 27261219 |
| VIPK562 | VIPerturb-seq（Satija lab, 2026），Zenodo 18460279 |
| 基因知識 | STRING v12（Szklarczyk et al., *Nucleic Acids Research* 2023）、Gene Ontology、GENCODE v46 |
