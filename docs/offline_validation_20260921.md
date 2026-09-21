# 2026-09-21 離線驗證與資料採用決策

四項離線工作已完成。這份紀錄只呈現本地實測，不是官方排行榜分數。
`artifacts/transfer_v2.vcc` 保留原檔；本次沒有提交、產生完整新版 VCC 或上傳資料。

## 1. 輸出基因與整數化

新的 transfer 預設保留所有控制組有表現的基因，不以 DE 的 5 CPM 門檻決定輸出。
預設 profile_scale=1、cell_patterns=400；指定容量時，先算擾動後密度，超出便報錯。
`--coverage legacy_budget` 才使用舊的截斷策略；縮放後再整數化會改變細胞分布。
400 個整數細胞的平均只能落在 1/400 的格點，floor/ceil 的每基因平均誤差上限為 0.00125。
極低表現仍可能量化為零；保留基因軸不等於保證每個基因都有非零計數。

固定抽取 32 個兩邊至少 50 細胞的共同目標；共同輸出軸 7,226 基因；效果縮放 0.3。
各條件均用 production writer 寫出 400 細胞/目標，再從 H5AD 讀回平均後評分。
只有 continuous 列是寫檔前的對照。這不是與舊版 914 目標完全相同的實驗。

| 輸出方法 | K562 → RPE1 PDS | RPE1 → K562 PDS |
|---|---:|---:|
| all_continuous | 0.7762 | 0.7681 |
| all_integer400 | 0.7732 | 0.7681 |
| all_depth015_integer400 | 0.7802 | 0.7651 |
| all_depth015_integer200 | 0.7742 | 0.7742 |
| top3000_integer400 | 0.5171 | 0.5161 |
| de_gate_integer400 | 0.7812 | 0.7681 |

**判讀：** 截掉大量基因會破壞正規化及擾動排名；這是實際輸出流程的問題。
本地結果不能證明官方差距全由此造成，也不能把全基因結果當成官方分數上限。

## 2. 沒有生物擾動時的 DE 檢查

A/B/C 各做 seeds 0、1、2：同一切分中 2,000 個 fit、800 個 reference、400 個真實 null 細胞互不重疊。
不同 seeds 之間可能重疊，不是九組獨立生物複本。每次所有方法使用同一切分與完整 18,533 基因軸。
先做每細胞全基因 CPM，再用 reference 平均 CPM>5 篩基因；tie-corrected rank-sum、BH<0.05。
實作為 asymptotic Mann–Whitney U、無 continuity correction；是秩和診斷，未宣稱逐位元重現伺服器。

| 細胞建構 | 深度倍率 | 九次 DE 數範圍 | DE 數中位數 |
|---|---:|---:|---:|
| real_null | 1 | 0–0 | 0 |
| resample | 1 | 0–1 | 0 |
| floorceil | 1 | 7035–8018 | 7442 |
| floorceil | 0.15 | 8757–9186 | 8923 |
| nb | 1 | 14–157 | 37 |
| conditioned_nb | 1 | 1–66 | 7 |

**判讀：** floor/ceil 保留量化後平均，但不保留真實細胞分布，會產生大量人工差異。
這是生物學 null 下的分布建構失真；統計檢定本身確實看見兩個不同分布，不宜稱成其 FDR 失控的證明。
固定總量的 conditioned NB 明顯改善，但 context C 仍有 41–66 個 DE；不可稱為完全修好。
逐基因 NB/Dirichlet-multinomial 不保留基因間相關，不能等同 diffusion/flow。
bootstrap 控制組 null 最接近真實對照，但加入擾動後還要評估平均誤差與轉移效果。

## 3. Notion 新資料：可補到什麼

以作者原始 metadata 做 exact-symbol matching；沒有把同義基因名默默合併。
原 K562gw 在每目標至少 25 細胞下覆蓋 267/300；CD4 guide library 名單涵蓋 297/300。

| CD4 狀態 | 有 DE 摘要的目標 | 比 K562gw 新增 | 候選 QC 通過 | QC 後新增 |
|---|---:|---:|---:|---:|
| Rest | 283 | 28 | 210 | 15 |
| Stim48hr | 284 | 28 | 216 | 19 |
| Stim8hr | 287 | 29 | 219 | 16 |

候選 QC 是本專案的預先篩選：至少 100 細胞、on-target 顯著、非 low_target_gex，
且沒有作者標出的 distal_offtarget 或 neighboring_gene_KD。不是官方品質認證，也沒有證明跨細胞有效。
三狀態聯集可新增 29 個已測目標；QC 後新增 19 個，原本 267 個加上它們最多成為 286 個候選。

QC 後新目標：DLAT、HEATR6、MAPK7、MLKL、PDK3、PHF11、PHF19、PSMB9、RAB11FIP5、SOCS5、SPATA20、STAT6、TBC1D19、TDRD7、TESK2、TMEM79、TTBK2、TXNDC16、ZC2HC1A。

CD4 是四位 donor × 三種 culture condition 的設計，不能說是十二種細胞株。
實際遠端 DE H5AD metadata 有 10,282 個 response genes，與比賽軸交集 10,113；這與 pseudobulk 文件的 18,129 基因層次不同。
DE H5AD 的 X 是空值，效果與統計量存於其他 layers；不能把它當 raw-count 訓練矩陣。
已取得作者摘要與遠端檔案 metadata，尚未下載、調和或訓練 CD4 全部細胞。

- CRISPRa HS27：mean-population 檔的 target categories 與比賽重疊 80 個，比 K562gw 新增 2 個（STAT6, ZNF174）；輸出軸 4,914 基因。
- CRISPRa RPE1：mean-population 檔的 target categories 與比賽重疊 80 個，比 K562gw 新增 2 個（STAT6, ZNF174）；輸出軸 5,424 基因。

CRISPRa 類別交集尚未代表每個 guide 的有效細胞數或活化品質；不能倒轉 LFC 就當作 CRISPRi。
**採用決策：** 先採 CD4 的直接 CRISPRi 證據作為新增來源，按狀態及 donor 保留層級；
功能先驗只作為未覆蓋目標的正則化或訊息共享候選，必須在外部 context 留出評估增益。
CRISPRa 資料另列為轉錄因子功能或調控方向的輔助資料，不直接混入 CRISPRi 訓練標籤。
Notion 的 133、舊關鍵字分類的 120 與 TF library 交集 80 定義不同，不能互相替代或當同一比例的重複驗證。

## 4. 相同資料與流程的生成模型 pilot

兩個方向，各 16 個相同的共同目標、兩個 seeds、7,226 基因；每個方法輸出 128 細胞/目標，真實評分亦為 128。
每目標 source 96 training + 32 validation；target 擾動細胞只作最後評分，
target 控制組分成 400 prediction controls 與 400 evaluation controls。樣本 row IDs 已保存在 cache NPZ。
PCA 16 維只 fit source；flow/diffusion 同一 128-wide MLP、同一資料、各 1,500 updates、batch 128、100 sampling steps。
checkpoint 只看固定 source validation objective；兩個 objective 不在同一數值尺度，不能比較其 loss 大小。
來源控制組條件用來校正解碼平均偏移；效果倍率固定 0.3。DDPM 用 cosine noise schedule 與 source-bound x0 clipping；
兩個生成家族的最後 latent 都限制在 source training 各成分的 min/max。模型 fit 與 checkpoint 不讀 target 擾動。
實作調試時曾看過本地結果，並修正 decoder、noise schedule 與 sampler；因此本地是開發集 pilot，
不是預先鎖定、從未查看過的最終檢验，也不能把此結果當成無偏的全新 context 泛化估計。
先前未校正 decoder、未達純噪音的 schedule 與未限制反推範圍的診斷另存 *_diagnostic.json，不列入正式表格。

| 方向 | 方法 | PDS 平均 | expr_error 平均 ↓ | DE Jaccard 平均 ↑ |
|---|---|---:|---:|---:|
| K562 → RPE1 | control | 0.5312 | 1.2051 | 0.0000 |
| K562 → RPE1 | transfer | 0.7750 | 1.0668 | 0.0629 |
| K562 → RPE1 | transfer_resample | 0.7083 | 1.2148 | 0.0252 |
| K562 → RPE1 | latent_mean | 0.6896 | 1.0431 | 0.0605 |
| K562 → RPE1 | flow | 0.6771 | 1.0391 | 0.0338 |
| K562 → RPE1 | diffusion | 0.6646 | 1.0351 | 0.0350 |
| RPE1 → K562 | control | 0.5250 | 1.4847 | 0.0005 |
| RPE1 → K562 | transfer | 0.8208 | 1.3162 | 0.0298 |
| RPE1 → K562 | transfer_resample | 0.8125 | 1.5874 | 0.0515 |
| RPE1 → K562 | latent_mean | 0.8000 | 1.3295 | 0.0283 |
| RPE1 → K562 | flow | 0.7854 | 1.3254 | 0.0285 |
| RPE1 → K562 | diffusion | 0.7896 | 1.3100 | 0.0294 |

`expr_error` 沒有官方 jackknife/replicate normalization；DE Jaccard 是此處 rank-sum 診斷，
不是原 local_eval 的 top-100 proxy。這兩種 Jaccard 在 JSON 以不同欄位保存，不可混稱。
pilot 使用共同基因軸，而非比賽完整軸；兩個 seeds 不構成生物複本的信賴區間。
這是小型 PCA latent 模型，不是原論文 PRiMeFlow、不含其預訓練、全部資料或大型 U-Net。
**判讀：** 依各指標及方向比較，不宣稱某一模型全面勝出。此 pilot 不支持因為模型較大就直接取代簡單轉移；
也不能以這個小實驗否定整個 diffusion/flow 家族。完整原模型復現與未見 context 的正式分數仍未取得。

## 重現與產物

```powershell
& '.venv/Scripts/python.exe' scripts/offline_validation.py roundtrip
& '.venv/Scripts/python.exe' scripts/offline_validation.py null
& '.venv/Scripts/python.exe' scripts/audit_external_coverage.py
& 'D:/anaconda3/python.exe' scripts/compare_generative_pilot.py
& '.venv/Scripts/python.exe' scripts/summarize_offline_validation.py
& 'D:/anaconda3/Scripts/uv.exe' run pytest -q -p no:cacheprovider
```

模型 pilot 使用已安裝的 CPU PyTorch；其 torch/sklearn 不屬於專案 uv 依賴。
原始新資料摘要抓取位置及 SHA256 在 external_coverage.json；遠端大型 H5 只透過有大小限制的 HTTP Range 檢查。
所有實驗檔在 `artifacts/offline_20260921/`：roundtrip.json、null.json、generative_pilot.json、
external_coverage.json、target_coverage.csv、H5AD、NPZ split/cache、PT source-only checkpoints。
完整新版提交仍需另行建構及確認壓縮容量；本次小型 roundtrip 不能保證全量檔符合上傳上限。

## 原始來源

- [CD4 作者程式與資料入口](https://github.com/emdann/GWT_perturbseq_analysis_2025)
- [CD4 作者資料欄位說明](https://github.com/emdann/GWT_perturbseq_analysis_2025/blob/master/metadata/data_sharing_readme.md)
- [CD4 GEO GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342)
- [Hs27 CRISPRa processed data](https://zenodo.org/records/15200179)
- [RPE1 CRISPRa processed data](https://zenodo.org/records/15213619)
- [TFs CRISPRa 作者程式](https://github.com/norman-lab-msk/TFs_CRISPRa)
- [PRiMeFlow 原作者實作](https://github.com/altoslabs/primeflow)
- [MIT Flow Matching and Diffusion Models 2026](https://diffusion.csail.mit.edu/2026/index.html)
