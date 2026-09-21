# transfer_v3_geneaxis：以 v2 為基準的單一變因實驗

## 實驗定義

唯一改動是輸出 gene strategy：取消 v2 的 5 CPM、最低平均計數與容量選基因條件，
保留完整基因軸。正式提交是官方 A/B/C 共同的 18,533 基因，順序與
`dataset/gene_names.csv` 完全一致。外部沒有量到的基因沿用 v2 的無額外效果處理，
不能把它們從官方輸出軸刪除。

**版本校正：** `transfer.vcc` 才是 top-3000；`transfer_v2.vcc` 實際使用
`entries_per_cell=8000`、5 CPM 與 `min_mean=1/200`，A/B/C 分別選取
9,923／9,610／10,124 個基因。本次的主比較是 v2 gene strategy → full axis；
top-3000 另外列為歷史策略的同參數對照。

| 設定 | v2 與本次候選共同使用 |
|---|---|
| 來源 | Replogle K562gw profiles，minimum cells 25 |
| fallback | K562 essential + RPE1 的 generic response |
| 擾動效果倍率 | 0.3 |
| 整體計數倍率 | 0.15 |
| 細胞建構 | 原本 floor/ceil；未改成 NB 或 bootstrap |
| patterns / cells per target | 200 / 400，重複兩次 |
| seed | 0 |
| 正式 panel | A/B/C × 300 targets × 400 cells |
| 改動項 | gene strategy |

fallback 對象以 v2 原先選取的基因判定後固定，避免擴張基因軸時順便改變 fallback。
同一 seed 的 RNG 在新增基因後可能改變保留基因的細胞排列，但每基因 floor/ceil
計數直方圖與群體總量應保持不變；不宣稱新舊單一細胞逐值相同。
非常低的預測平均仍可能整數化為零；完整基因軸不代表強迫每基因至少一個非零值。

## 實作範圍

- 獨立設定：[experiments/transfer_v3_geneaxis.json](../experiments/transfer_v3_geneaxis.json)。
- 獨立 driver：[scripts/transfer_geneaxis_experiment.py](../scripts/transfer_geneaxis_experiment.py)。
- 重用既有 source/fallback 計算、submission writer、floor/ceil 與 PDS 函式。
- 沒有改寫 production package、CLI 預設或既有離線驗證。
- 不覆寫 `transfer_v2.h5ad` 或 `transfer_v2.vcc`，不加入 CD4、flow、diffusion。

## 同條件的雙向 PDS

沿用 `artifacts/offline_20260921/roundtrip.json` 中的**原本 32 個目標清單**，
不重新抽樣、不調參；完整共同輸出軸 7,226 基因。所有列固定 0.3 effect scale、
0.15 profile scale、200 patterns、400 cells、seed 0，寫出 H5AD 後再讀回評分。

| 輸出策略 | K562 → RPE1 PDS | RPE1 → K562 PDS |
|---|---:|---:|
| v2 gene strategy | 0.772177 | 0.774194 |
| 完整共同基因軸 | 0.774194 | 0.774194 |
| top-3000，同參數對照 | 0.517137 | 0.516129 |

相對真正的 v2，PDS 差異為 +0.002016 與 0。這是小型本地比較，沒有顯著性或
官方分數提升的保證。先前「約 0.77 vs 0.52」主要描述 top-3000 截斷，而非 v2
本身。此次 RPE1 作為 target 的 v2 filter 留下 7,225 基因；反向的 filter
已保留全部 7,226 基因，因此反向兩列一致是可預期的。

## 重現

```powershell
& '.venv/Scripts/python.exe' scripts/transfer_geneaxis_experiment.py build
& '.venv/Scripts/python.exe' scripts/transfer_geneaxis_experiment.py verify
& '.venv/Scripts/python.exe' scripts/transfer_geneaxis_experiment.py evaluate
```

build 拒絕覆寫既有候選輸出；原始 v2 submission 必須先通過已記錄的 SHA-256。
詳細設定、v2 重建抽查、900 群組完整 readback、雙向分數與官方 prep 結果都保存在
`artifacts/transfer_v3_geneaxis_audit/`。

## 完整建構與驗證結果

| 項目 | 結果 |
|---|---|
| H5AD | `artifacts/transfer_v3_geneaxis.h5ad` |
| 可提交封裝 | `artifacts/transfer_v3_geneaxis.vcc` |
| H5AD 大小 | 923,667,626 bytes（880.88 MiB） |
| VCC 大小 | 1,061,365,760 bytes（1,012.20 MiB） |
| 相對 v2 VCC | 增加 13.40 MiB |
| 邏輯形狀 | 360,000 × 18,533 |
| 群組 | 3 contexts × 300 targets，每組 400 細胞 |
| 非零項目 | 942,407,204；CLI 上限 4,750,000,000 的 19.84% |
| 最大每細胞總計數 | 3,336；低於 CLI 的 1,000,000 上限 |
| 全量 readback | 900/900 群組通過，有限非負整數、模式重複、群體總量均正確 |
| 原 v2 保留基因 | 全部 900 群組的群體總量與原檔完全一致 |
| 官方 `vcc prep` | 通過，`counts-preserved`、`verified_targets=true`、`reordered_genes=false` |
| 封裝後的矩陣 | 解壓後 CSR indptr、indices、data 全量逐值比較一致 |
| 封裝後 metadata | gene 順序與逐列 context/target label 一致 |
| 封裝獨立性 | 解包後 X 非 VDS，不依賴原始 H5AD 或其 uns 儲存區 |
| 原 v2 | H5AD 與 VCC 的 SHA-256 均與建構前一致 |
| 程式檢查 | 40 tests passed；新 driver Ruff 通過；git diff --check 通過 |

官方 prep 會移除僅供 VDS 使用的 `uns[_context_cells]`，將 row index 改為通用序號，
並可能對 target categorical dictionary 做自然排序。驗證比較的是每列實際標籤，
不是要求內部 category code 數字相同；矩陣與標籤的對應沒有改變。

本次只建立與驗證檔案，**沒有執行 submit，也沒有 commit/push**。
這個候選保留原本 floor/ceil 分布的限制，沒有混入先前 NB 或生成模型實驗。

官方打包命令：

```powershell
& "$env:USERPROFILE\.local\bin\vcc.exe" prep artifacts/transfer_v3_geneaxis.h5ad `
  -g artifacts/transfer_v3_geneaxis_audit/genes_headerless.csv `
  --perts dataset/pert_counts.csv -o artifacts/transfer_v3_geneaxis.vcc --json

& 'C:/Users/user/AppData/Roaming/uv/tools/vcc-cli/Scripts/python.exe' `
  scripts/transfer_geneaxis_experiment.py archive
```

`archive` 使用已安裝 vcc 的 Python（其中有 zstandard），解開實際 VCC 並做全量比對。
檔案雜湊與 CSR 內容雜湊見 `archive_validation.json`；本地官方 prep 完成不等於已取得
伺服器接受結果或排行榜分數。
