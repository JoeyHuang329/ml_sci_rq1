# RQ1：人格特質對藥物使用的預測力 (Statistical Machine Learning Pipeline)

本專案將原 Colab notebook `ml_sci_process_rq1.ipynb` 拆解為符合
*separation of concerns* 原則的多檔模組化架構。所有設計皆以統計推論
(Statistical Inference) 為導向，而非單純預測準確率最大化。

---

## 目錄結構

```
rq1_project/
├── config/
│   ├── __init__.py
│   └── settings.py               # 隨機種子、特徵集合、超參數網格、路徑
├── data_io/
│   ├── __init__.py
│   ├── load_clean.py             # UCI 下載 + Semer 過濾
│   ├── deterministic_encoders.py # EthnicityBinarizer / CountryGrouper
│   └── targets.py                # 二值化 / 連續化
├── preprocessing/
│   ├── __init__.py
│   └── pipeline.py               # ColumnTransformer 工廠 (折內擬合)
├── splits.py                     # 逐藥物 Stratified Split
├── models/
│   ├── __init__.py
│   ├── ridge.py                  # 方法 A — Baseline
│   ├── lasso.py                  # 方法 B — Main (含 Bootstrap CI)
│   └── knn.py                    # 方法 C — Comparison
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # AUC / AUPRC / Brier / MSE + AUC CI
│   ├── delong.py                 # DeLong 配對檢定
│   └── inference.py              # AUC>0.5 單側檢定 + BH FDR 校正
├── reports/
│   ├── __init__.py
│   ├── tables.py                 # 統一評估表 (★ 修正原 bug 的核心)
│   └── figures.py                # 散佈圖、熱度圖、排名圖
├── scripts/
│   ├── __init__.py
│   └── run_rq1.py                # 主要執行入口
├── requirements.txt
└── README.md
```

---

## 執行方式

```bash
pip install -r requirements.txt

# 完整實驗 (B = 1000 Bootstrap, 約 30-60 分鐘)
python -m scripts.run_rq1

# 煙霧測試 (B = 50, 約 5 分鐘)
python -m scripts.run_rq1 --bootstrap 50
```

執行後在 `artifacts/tables/` 與 `artifacts/figures/` 下生成所有報表。

---

## 對原 notebook 的關鍵修正

### Bug A — §3.5.2 之 Lasso 機率被誤設為亂數佔位符

原 notebook 第 799 行：
```python
y_prob_lasso = np.random.uniform(0, 1, size=len(y_test_d))  # Placeholder!
```
**後果**：DeLong 檢定實際上是在比較「KNN 機率 vs 純隨機機率」，**所有
顯著性結論皆無效**。

**修正**：在 `models/lasso.py` 將 `predict_proba(X_test)[:, 1]` 直接存入
`LassoResult.y_pred_prob_test`；`reports/tables.py` 從該物件取出，
**全程無亂數**。

### Bug B — Ridge 評估時將 best_alpha 棄用，硬塞 alpha = 1.0

原 notebook 第 884 行：
```python
('ridge', Ridge(alpha=1.0, random_state=GLOBAL_SEED))
# Using alpha=1.0 as a proxy since best_alpha wasn't stored per drug
```
**後果**：Ridge 測試集 MSE 不對應於 §3.2 CV 找到的最佳 λ̂，
基準的統計可比性破壞。

**修正**：在 `models/ridge.py` 中以 `RidgeResult` dataclass 保存
`best_alpha` 與已 refit 的 `pipeline`，下游直接取用，不再重訓。

### Bug C — 為了取得測試集機率而重新訓練全部模型

原 notebook §3.5 末段 (864-915)：對 15 種藥物全部重新訓練 Ridge 與
Lasso，浪費約 2 倍計算時間。

**修正**：所有模型訓練後將 `pipeline` 與 `y_pred_prob_test` 一併存入
結果物件，評估模組直接消費。

### Bug D — ColumnTransformer 輸出順序的脆弱假設

原 notebook 第 419 行寫死 `coef_[:7]` 取心理係數，仰賴 transformers
列表的第一項剛好是 `psych_scale`。

**修正**：在 `preprocessing/pipeline.py` 提供
`psych_indices_in_default_pipeline()` 函數，集中管理位置切片，
未來新增 transformer 時只需修改一處。

### Bug E — Country/Ethnicity 編碼在 split 之前的「半洩漏」設計

原 notebook 在 split 之前就將 `country_grouped` 與 `ethnicity_white`
寫進 DataFrame。雖然兩者皆為決定性映射 (不依賴 y 或訓練集統計量)，
故**並非真實的資料洩漏**，但設計上不一致。

**修正**：將兩者封裝為 sklearn-compatible Transformer
(`EthnicityBinarizer`, `CountryGrouper`)，可選擇於 Pipeline 內或
全資料上一次性套用 (本專案採後者以節省每折運算)。

### Bug F — Module-level 副作用

原 notebook 在 import 時即執行 `warnings.filterwarnings('ignore')`、
`np.random.seed(GLOBAL_SEED)`、`!pip install`、`display()` 等
Colab 專屬指令。

**修正**：
- 全域 seed 不再透過 `np.random.seed`，改由各模組函數參數明確接收；
- Bootstrap 內層使用 `np.random.default_rng(seed + b)` 而非全域狀態；
- 移除所有 `!pip install`、`display()`、`%config` 等 Colab 殘留。

### Bug G — DeLong 邊界情況

原 notebook 之 `fast_delong_pvalue` 在 m=0 或 n=0 時 `np.cov` 退化，
也未檢查 NaN。

**修正**：`evaluation/delong.py` 明確處理 m=0/n=1/m=1 等退化情形，
回傳 `degenerate=True` 旗標供下游判斷。

### Bug H — 記憶體：`rq1_splits` 重複儲存 X 副本 15 份

原 notebook 將 (X_train, X_test, y_train, y_test) 完整切片儲存 15 份，
記憶體足跡 = O(N · |D|)。

**修正**：`splits.DrugSplit` 為 frozen dataclass，只儲存 `train_idx`,
`test_idx` 兩個 `pd.Index`，記憶體 O(N)。

---

## 統計學設計要點

| 項目 | 設計 | 統計學動機 |
|---|---|---|
| 標準化 / Target Encoding | 折內擬合於 sklearn Pipeline | 杜絕 optimistic bias |
| Stratified Split (per drug) | 各藥物獨立分層 | 避免極端不均衡藥物之 Test Set 退化 |
| Ridge ℓ₂ baseline | 不稀疏化 | 揭示「未剔除任何訊號」之邊際效應地圖 |
| Lasso ℓ₁ main | 自動特徵選擇 | 提供 log-odds-ratio 之可解釋係數 |
| Bootstrap CI (固定 λ̂) | 非參數 percentile | Lasso 估計量無封閉式漸近分布 |
| KNN (psych-only) | 無母數對照 | 結構誤設診斷 (Linear vs Non-linear) |
| DeLong test | 配對 AUC 檢定 | 同一測試集上之相依性處理 |
| BH FDR 校正 | α = 0.05 | 15 種藥物之多重比較控制 |

---

## 已知限制 (Future Work)

1. **Post-Selection Inference**：固定 λ̂ 之 Bootstrap CI 未納入「λ 選擇」之
   不確定性。若需嚴格事後選擇推論，建議引入 Lee et al. (2016) 之
   selection-conditional 推論框架。

2. **CL 連續化假設**：Ridge baseline 假設 CL0-CL6 等距，但兩相鄰等級
   間之心理距離未必相等。未來可採序列邏輯迴歸 (Cumulative Logit Model)
   作為 Ridge 之替代基準。

3. **KNN 距離度量**：歐式距離隱含「所有 psych 維度等權重」的假設。
   可考慮 Mahalanobis 距離以反映特徵間相關性。
