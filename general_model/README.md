# VCC 2026 通用擾動預測模型（期末版本 v5）

Arc Virtual Cell Challenge 2026：只給一種新細胞的對照組細胞，以及要用 CRISPRi 關掉的基因，
預測每個基因被關掉後 400 顆細胞、18,533 個基因的表現。期末考是三種模型沒看過的細胞（D、E、F），
每種 300 個基因。

這個資料夾是 2026/10/09 之後重新開始的版本，取代 repo 根目錄的舊流程（舊流程最佳 0.109）。
程式裡的路徑寫死在原本的工作機 `/home/zxc29051617/VCC`，換機器時要改。

## 成績

| 版本 | 期末上傳 | 內容 | H1 模擬考（官方評分） |
| --- | --- | --- | --- |
| v1 | 10/22 19:44 | 各來源平均（共識模型）＋細胞產生流程 | 0.272 |
| v2 | 10/23 01:30 | 改用訓練的 AdaptNet 預測總量 | 0.291 |
| v4 | 10/23 18:40 | ＋RPE1 資料、不移除對照組軸、目標權重、3 個模型平均 | 0.306 |
| **v5** | **10/24 00:24（目前算分）** | v4 改成移除 5 個對照組軸 | **0.306**；HCT116 / KOLF 區分度明顯較高 |

- 驗證期公開排行榜（A/B/C，舊流程 G3v）：0.239，第 165 名 / 1,635 隊。
- 期末分數在比賽結束、公布得獎者時才公開。
- 「藏起整種細胞」7 折交叉驗證：方向相似度 0.111（各來源平均）→ 0.152（AdaptNet＋RPE1），區分度 0.738 → 0.766。

## 模型

- **AdaptNet**（`scDFM/repro/model_v2_20261022/adaptnet.py`）：約 2 萬個參數的小型神經網路。
  預測「新細胞中關掉基因 X」時，參考其他細胞中關掉 X 的測量：(1) 依細胞相似度、資料品質、來源一致性
  替每個來源加權；(2) 拆成 200 組基因程式加細節；(3) 依新細胞對照組的狀態調整各程式強弱。
  訓練時每一步藏起一種細胞，只用其他細胞預測它。期末版本是 3 個亂數種子的平均。
- **GeneNet4**（`genenet.py`）：從沒被測過的基因改用 STRING＋GO 最相似的 32 個基因推測
  （期末考的基因都有測量，期末沒有用到這條路）。
- **產生細胞**（`tools/final_pipeline.sh`、`scDFM/vcc/render.py`）：總量依評分規則校準
  （budget = claim），從新細胞對照組抽 400 顆真實細胞，調整計數讓每個基因總量剛好等於預測。

## 資料夾

| 路徑 | 內容 |
| --- | --- |
| `scDFM/repro/model_v2_20261022/` | 期末模型、基準測試、模擬考、診斷工具；**`SUMMARY.md` 是最完整的實驗紀錄** |
| `scDFM/repro/next_20261019/` | A/B/C 時期的流程實驗（`SUMMARY.md`、`run.log`） |
| `scDFM/repro/signature_20261013/` | 來源加權實驗與 signature cache 產生程式 |
| `scDFM/vcc/` | 共識模型（`gm.py`）、資料讀取、細胞產生（`render.py`） |
| `tools/` | 期末流程 `final_pipeline.sh`、總量 `make_bulk_effects.py`、資料下載與轉換 |
| `vccsub/` | 寫出預測細胞與打包 `.vcc`（`make_effect_submission.py`、`pack_vcc.py`） |
| `docs/` | 給教授的進度簡報（`VCC_progress_report_20261024.pptx`，每頁有講稿備註） |

## 重建期末版本 v5

```bash
cd scDFM
RPE1=1 SEEDS=0,1,2 .venv/bin/python -I repro/model_v2_20261022/build_v2.py v4   # 訓練 3 個 AdaptNet，寫出 gm_final_v4pre*
cd ..
PRECOMP=/home/zxc29051617/VCC/data/stage3/effects/gm_final_v4pre RENDER=g1 SHAREDK=0 GAMMA=1 CTRLAXES=5 \
  TWPOW=0.5 CELLAGREE=1 CELLAMP=1.5 TOPFRAC=0.02 bash tools/final_pipeline.sh data/vcc2026_final gm_final_v5 D E F
# 上傳（只算最後一次，先和隊友確認）：~/.local/bin/vcc submit vccsub/out/gm_final_v5.vcc -m gm_final_v5 --wait
```

## 驗證

```bash
cd scDFM
# 官方評分模擬考：H1（5 組，深度最接近期末考）、HCT116 與 KOLF200（各 4 組，看原始分數）
RPE1=1 SEEDS=0,1,2 .venv/bin/python -I repro/model_v2_20261022/mock_exam.py 0,1,2,3,4 <名稱> "CELLAGREE=1;CELLS_FROM_A=1;CTRLAXES=5;TWPOW=0.5"
MOCK_SET=hct116 RPE1=1 SEEDS=0,1,2 .venv/bin/python -I repro/model_v2_20261022/mock_exam.py 0,1,2,3 <名稱> "<同上>"
.venv/bin/python -I repro/model_v2_20261022/mock_cmp.py 0,1,2,3,4 <名稱>        # H1 換算分數
.venv/bin/python -I repro/model_v2_20261022/mock_raw.py 0,1,2,3 hct116_<名稱>   # 原始分數
# 不產生細胞的快速試算：mse_proxy.py（mse 與 pds，pds 與官方值一致）、de_sim.py（四項 DE 分數）
# 藏起細胞 / 藏起基因的基準：bench.py、bench_genes.py、bench_sign.py、bench_w.py
```

## 不在 repo 裡的資料（約 40 GB）

- `data/stage3/model_v2/*.npz`：15 個公開 CRISPRi 實驗整理成的訓練資料（`build_data.py`、`rpe1_build.py`），
  來源：Replogle 2022（K562、RPE1）、X-Atlas/Orion（HCT116、HEK293T）、VCC 2025 H1、
  CD4 T 細胞全基因組 CRISPRi、Nadig 2025（HepG2、Jurkat）、Feng 2025 與 KOLF iPSC、VIPerturb K562。
- `data/stage3/stats`、`stats_ext`：共識模型用的各實驗彙總；`scDFM/repro/signature_20261013/cache`：來源一致性。
- `data/vcc2026`、`data/vcc2026_final`：主辦單位給的 A/B/C、D/E/F 對照組細胞。
- `data/mock_h1`、`data/mock_hct116`、`data/mock_kolf200`：模擬考資料與評分 bundle。
- `evalenv/`：官方評分程式 cell-eval2（vcc2026 preset）。

## 主要發現與限制

- 區分度（pds）已達實驗重複水準的約 87%；瓶頸在「哪些基因顯著改變、往哪變」：非必需基因
  在新細胞上，最有把握的預測方向只猜對 53–58%（必需基因 70–80%）。
- 試過但沒採用（數據在 `SUMMARY.md`）：加入細胞的共同反應、細胞週期比例、方向損失、對照組自然差異、
  異質性下降基因、表現量加權訓練、ELEM-1200RS / LINCS / 神經細胞資料等。
- 期末考細胞：D 像骨髓性白血病、E 像纖維母細胞、F 像上皮／間質癌細胞，訓練資料裡沒有相似細胞。
