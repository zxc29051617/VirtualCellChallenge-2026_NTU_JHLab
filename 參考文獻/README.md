# 參考文獻

這個資料夾放專題用到的論文 PDF，以及「為什麼用這份資料」的決策紀錄。

---

## A. 賽事文件

### Virtual Cell Challenge 2026.pdf ✅ 已下載

> Arc Virtual Cell Initiative Team and Goodarzi, H. (2026). Virtual Cell Challenge 2026:
> Benchmarking zero-shot generalization across cellular contexts. *Cell* 189, 5827–5830.
> doi:10.1016/j.cell.2026.08.004（開放取用）

**今年賽制的唯一權威來源。** 四頁的 Commentary。重點：今年是 zero-shot，不發放訓練集；
六個細胞株（3 驗證 + 3 測試）；最終計分是「對 baseline 正規化後的六指標平均」。

詳細整理見 `docs/challenge_brief.md`。

### vcc2026 Metric Suite (official spec).pdf ✅ 已下載

> The vcc2026 Metric Suite（2026/08/19，19 頁）

**六個指標的完整數學定義與提交格式規格。** 這是最重要的技術文件，比 Commentary 詳細得多。
內容包含：

- 計分公式 `s = (u − b) / (r − b)`，其中 b 是 baseline、**r 是實驗重複（不是滿分）**
- 每個指標實測的 b 與 r 數值，以及「完全複製參考答案」會得幾分
- **每個干擾的目標基因本身被排除在全部六個指標之外**（`pds_cosine` 更是移除全部 300 個）
- 上傳格式（`vcc prep`）與計分格式（`cell_eval2`）是**兩份不同的契約**
- 顯著性檢定細節：Wilcoxon 秩和檢定、低表現過濾（對照組 > 5 CPM）、BH 校正 α = 0.05

原始檔在 `tmp/pdfs/vcc_official/vcc2026-metrics.pdf`，此處為副本。

### 虛擬細胞挑戰：邁向虛擬細胞的圖靈測試.pdf ⬜ 目前不在資料夾

> Roohani, Y.H., Hua, T.J., Tung, P.-Y., Bounds, L.R., et al. (2025). Virtual Cell Challenge:
> Toward a Turing test for the virtual cell. *Cell* 188, 3370–3374.
> doi:10.1016/j.cell.2025.06.008（開放取用）

**2025 年第一屆**的賽事說明。內容已被上面那篇取代，但可用來說明賽制如何演變
（去年有訓練集、只有 H1 幹細胞一種細胞、只計三個指標）。可留可不留。

---

## B. 方法參考

### STATE 論文 ⬜ 目前不在資料夾，建議補回

> Adduri, A.K., Gautam, D., Bevilacqua, B., Imran, A., Shah, R., Naghipourfar, M.,
> Teyssier, N., Ilango, R., Nagaraj, S., Dong, M., et al. (2025). Predicting cellular
> responses to perturbation across diverse contexts with State. bioRxiv.
> doi:10.1101/2025.06.26.661135

Arc 官方目前表現最好的模型，2026 賽事論文有引用。兩個模組：State Transition（transformer，
在細胞集合層級建模干擾效應）+ State Embedding（在 1.67 億個細胞上預訓練）。
訓練資料超過 1 億個受干擾細胞、涵蓋 70 種細胞情境。

**用途：** 報告的「相關研究」章節。不需要重現，用來對比「業界頂尖做法 vs. 本專題在資源
限制下的取捨」。

### 值得一讀但尚未下載

> Wei, Z., Wang, Y., Gao, Y., Wang, S., Li, P., Si, D., Gao, Y., Wu, S., Li, D., Dong, K.,
> et al. (2026). Benchmarking algorithms for generalizable single-cell perturbation response
> prediction. *Nat. Methods* 23, 451–464. doi:10.1038/s41592-025-02980-0

獨立的第三方基準評測，2026 賽事論文引用。適合用來佐證「現有方法的極限在哪」。

---

## C. 本地評估用資料

### 為什麼需要外部資料

2026 挑戰**完全不提供正確答案**，連驗證用的 A/B/C 也沒有。手上每一顆細胞都是對照組，
所以無法在本地替任何預測打分數。要建立「改模型 → 看分數 → 決定保留或回退」的迭代迴圈，
只能靠外部的公開擾動資料。

官方論文明確允許：「Participants are encouraged to use any appropriate data and modeling
strategy」。

### 選用：Replogle 2022（經 scPerturb 統一化）

需要引用**兩篇**，因為原始實驗和實際下載的檔案來源不同。

**原始實驗：**

> Replogle, J.M., Saunders, R.A., Pogson, A.N., Hussmann, J.A., Lenail, A., Guna, A.,
> Mascibroda, L., Wagner, E.J., Adelman, K., Lithwick-Yanai, G., et al. (2022). Mapping
> information-rich genotype-phenotype landscapes with genome-scale Perturb-seq.
> *Cell* 185, 2559–2575.e28. doi:10.1016/j.cell.2022.05.013
>
> 📄 `Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq.pdf`

**實際下載的檔案版本：**

> Peidli, S., Green, T.D., Shen, C., Gross, T., Min, J., Garda, S., Yuan, B., Schumacher, L.J.,
> Taylor-King, J.P., Marks, D.S., et al. (2024). scPerturb: harmonized single-cell
> perturbation data. *Nature Methods* 21, 531–540. doi:10.1038/s41592-023-02144-y
>
> 📄 `scPerturb harmonized single-cell perturbation data.pdf`
>
> 資料位置：Zenodo record 13350497（`scripts/download_external_data.ps1` 會自動下載）

### 選它的理由

不只是「有答案」，而是**結構上跟比賽任務同形**：

| | K562 | RPE1 |
| --- | --- | --- |
| 細胞數 | 310,385 | 247,914 |
| 基因數 | 8,563 | 8,749 |
| 干擾數 | 2,058 | 2,394 |

**2,056 個干擾在兩個細胞株都有測量。** 同一批干擾施加在兩種不同的細胞情境上，
所以可以「在 K562 上擬合、在 RPE1 上評分」，模擬比賽的跨情境 zero-shot 任務。
兩者都是 CRISPRi，擾動方式與比賽一致。

### 查證紀錄（2026-09-14）

兩篇論文都讀過、檔案也直接檢查過：

- **三方數字完全吻合** — scPerturb Table 1 列出的 310,385 / 2,058 與 247,914 / 2,394，
  與實測完全一致；Replogle 原文也載明 K562 essential 篩選了 2,057 個基因（+1 對照 = 2,058）。
- **scPerturb 是重新封裝而非重新處理**（他們自述：多數來源只提供處理後的矩陣）。
- **`.X` 確認是原始整數計數**，符合比賽的原始計數要求。`core_scale_factor` 等欄位是
  Replogle 附帶的校正參數，並未套用到主矩陣上。
- **原始實驗方法嚴謹** — 每個基因兩條 sgRNA、排除導引 RNA 不一致的細胞、
  敲低效率中位數 85.5%（K562）/ 91.6%（RPE1）。

⚠️ **最重要的但書：** RPE1 的基因清單是「DepMap 必需基因 + 依據 K562 全基因組實驗結果
人工挑選的基因」。兩組資料並非互相獨立，所以 K562→RPE1 的分數會**偏樂觀**。
本地分數只能用於「改動前後的相對比較」，不可用來推估比賽成績。

完整的注意事項排序見 `docs/validation_strategy.md`。

### 考慮過但未採用

| 資料集 | 未採用的原因 |
| --- | --- |
| VCC 2025 atlas（H1 幹細胞，約 30 萬細胞）| 與比賽高度相關，誤用去年的驗證/測試資料會有洩漏風險 |
| Norman et al. 2019（K562）| 只有單一細胞株，無法做跨情境評估；晶片化學較舊 |
| Tahoe-100M（50 種癌症細胞株，1 億筆）| 小分子擾動而非 CRISPRi，擾動方式與比賽不符 |
| scBaseCount（Arc atlas，逾 6 億細胞）| 以觀察性資料為主，適合表徵學習而非擾動反應評估 |

> 本節取代原先的 `notes/research_public_datasets.md`（該檔內容已併入此處，
> 其中標註「待確認」的項目已由上方查證紀錄取代）。
