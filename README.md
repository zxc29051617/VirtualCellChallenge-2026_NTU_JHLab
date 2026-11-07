# Virtual Cell Challenge 2026 — NTU JHLab

[Arc Virtual Cell Challenge 2026](https://virtualcellchallenge.org) 的參賽專案，在臺大資工系黃宣誠老師實驗室進行。

比賽給三種從來沒有被做過基因擾動實驗的細胞（期末代號 D、E、F），只提供未處理的對照組細胞，要預測每種細胞分別用
CRISPRi 關掉 300 個基因後，18,533 個基因的表現會怎麼改變，並以每個擾動 400 顆細胞的讀數交出。因為這三種細胞沒有任何
擾動資料，只能從其他細胞種類的公開實驗推測（零樣本）。

團隊：[zxc29051617](https://github.com/zxc29051617)、[scnpg](https://github.com/scnpg)

## 方法

使用設計的 **AdaptNet**（跨細胞種類的擾動反應遷移模型），詳細說明在
[`general_model/README.md`](general_model/README.md)。

1. 收集 15 個公開的 CRISPR 擾動資料集（9 種細胞、38,303 組「某種細胞中關掉某個基因」的測量）。
2. AdaptNet 預測新細胞中關掉某個基因時，每個基因的總量變化：它會判斷其他細胞的實驗結果哪些比較可信，再依新細胞的
   對照組狀態調整。
3. 從新細胞自己的對照組抽出 400 顆真實細胞，依預測調整讀數，產生要上傳的細胞。
4. 每個改動都先用模擬考（答案已知的細胞加上官方評分程式）驗證，有進步才採用。

## 結果

| 版本 | 內容 | 模擬考總分 |
|---|---|---|
| 原方法 | 各來源的擾動反應直接平均 | 0.272 |
| v2 | 總量改用 AdaptNet 預測 | 0.291 |
| **v5（最後上傳）** | 加入 RPE1 資料、3 個模型平均、移除 5 個對照組方向 | **0.306** |

- 分數的意義：0 ＝ 和官方的基準猜法一樣，1 ＝ 和同一個實驗重做一次一樣準。
- 模擬考的資料也拿來挑選設定，所以 0.306 是偏樂觀的開發集成績。
- 驗證期用其中的共識模型上傳，公開排行榜 0.239，第 165 名（共 1,635 隊）。

各版本的差異、六項評分的細項、試過但沒用的方法，見 `general_model/README.md` 第 9、10 節。

## 專案結構

```text
.
├── general_model/              # 通用模型
│   ├── README.md               # 背景知識、評分方式、模型、結果與重跑步驟
│   ├── pertrelay/              # PertRelay：模型（AdaptNet、GeneNet4）、細胞產生流程與實驗紀錄
│   ├── vccsub/                 # 產生與打包上傳檔
│   ├── tools/                  # 資料下載與整理
│   ├── docs/                   # 圖表與實驗紀錄
│   └── papers/                 # 參考論文清單
├── src/、scripts/、experiments/  # 比賽初期的基準流程
├── docs/                       # 比賽規則與初期的實驗紀錄
├── data/                       # 初期流程用的小型資料檔
└── tests/                      # 初期流程的測試
```

## 執行

不用下載資料也能重現模擬考的分數：

```bash
cd general_model/pertrelay
uv run --no-project --with numpy python model/mock_cmp.py 0,1,2,3,4 current v4_ax5
```

會印出原方法（`current`）與期末 v5（`v4_ax5`）在 5 組模擬考的總分 0.2723 與 0.3063。
完整的環境建立、資料下載、訓練與重建上傳檔的步驟，見 `general_model/README.md` 第 11 節。

## 比賽初期的基準流程

比賽初期隊友先做了一個簡單的基準：把公開細胞的擾動反應轉移到比賽細胞上，最好成績 0.109。相關程式在 `src/`、`scripts/`，
規則與實驗紀錄在 `docs/challenge_brief.md`、`docs/state_of_play.md`。
