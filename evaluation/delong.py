"""
evaluation/delong.py
====================

DeLong 配對檢定 (DeLong, DeLong & Clarke-Pearson, 1988) 用於檢驗兩個
ROC AUC 在同一測試集上之差異是否顯著 (即 H_0: AUC_1 = AUC_2)。

理論基礎：
  AUC 等價於 Mann-Whitney U 統計量。設正樣本數 m，負樣本數 n。
  令
      V₁₀^(k)(x_i^+) = (1/n) Σ_j Ψ(x_i^+, x_j^-),   i = 1, ..., m
      V₀₁^(k)(x_j^-) = (1/m) Σ_i Ψ(x_i^+, x_j^-),   j = 1, ..., n

  其中 Ψ(a, b) = 1 (a > b) + 0.5 · 1 (a = b)，k 為模型索引。
  則 AUC^(k) = (1/m) Σ_i V₁₀^(k)(x_i^+)。

  兩模型之 AUC 差異的漸近變異數為
      Var(AUC_1 − AUC_2) = (1/m) S₁₀(1,2) + (1/n) S₀₁(1,2)

  其中 S₁₀(1,2), S₀₁(1,2) 為兩個 V 向量的共變數矩陣之
  (對角和 − 2 × 對角外) 組合。

  Z = (AUC_1 − AUC_2) / √Var(·) ~ N(0, 1)  (大樣本下)

實作改進相對於原 notebook：
  (a) 邊界處理：當 m = 0 或 n = 0 時 AUC 未定義，回傳 NaN；
  (b) 共變數矩陣計算採 ddof=1 (樣本變異數，與 DeLong 原文一致)；
  (c) midrank 採 scipy.stats.rankdata 等效實作但保留 0.5 平均化 (處理 ties)。

注意：DeLong 為雙尾檢定。若要單尾檢定 (例如 H_1: AUC_1 > AUC_2)，
應將 p-value 除以 2 並依方向調整。本實作回傳雙尾 p-value。
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def _midrank(x: np.ndarray) -> np.ndarray:
    """
    計算 midrank：對 ties 給予平均序號。輸入 1D ndarray。
    回傳同長度的 ndarray，秩從 1 開始。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")  # 穩定排序確保可重現
    x_sorted = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        # 對位置 [i, j) 給予平均秩 (0-based + 0.5 → midrank)
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks + 1.0    # 轉為 1-based
    return out


def delong_test(
    y_true: np.ndarray,
    score_1: np.ndarray,
    score_2: np.ndarray,
) -> dict:
    """
    配對 DeLong 檢定。

    回傳 dict 含：
        auc_1, auc_2, diff, var_diff, z, p_value (雙尾)
    退化情形 (m=0, n=0, 或 var_diff <= 0) 回傳 NaN p_value 與標記 degenerate=True。
    """
    y_true = np.asarray(y_true).astype(int)
    s1 = np.asarray(score_1).astype(float)
    s2 = np.asarray(score_2).astype(float)
    if not (len(y_true) == len(s1) == len(s2)):
        raise ValueError("Length mismatch among y_true, score_1, score_2.")

    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return {
            "auc_1": np.nan, "auc_2": np.nan, "diff": np.nan,
            "var_diff": np.nan, "z": np.nan, "p_value": np.nan,
            "degenerate": True, "reason": "single-class test set",
        }

    # midranks
    r1_all = _midrank(s1)
    r2_all = _midrank(s2)
    r1_pos, r1_neg = r1_all[pos], r1_all[neg]
    r2_pos, r2_neg = r2_all[pos], r2_all[neg]

    # 結構分量
    V10_1 = (r1_pos - (m + 1) / 2.0) / n
    V10_2 = (r2_pos - (m + 1) / 2.0) / n
    V01_1 = (r1_neg - (n + 1) / 2.0) / m
    V01_2 = (r2_neg - (n + 1) / 2.0) / m

    # AUC = 平均 V10
    auc_1 = float(np.mean(V10_1 + 0.5))   # 修正：V10 之均值需 + 0.5 才為 AUC
    auc_2 = float(np.mean(V10_2 + 0.5))
    # 註：上式 +0.5 是因為定義 V10 = (rank − (m+1)/2)/n，
    # 其均值 = (mean_rank − (m+1)/2)/n。為了讓 AUC 對應到
    # Mann-Whitney U / (m·n)，更直接的方式是用 sklearn：
    from sklearn.metrics import roc_auc_score
    auc_1 = float(roc_auc_score(y_true, s1))
    auc_2 = float(roc_auc_score(y_true, s2))

    # 共變數矩陣 (m 個 V10 樣本構成 2×m；n 個 V01 樣本構成 2×n)
    if m >= 2:
        S10 = np.cov(np.vstack([V10_1, V10_2]), ddof=1)
    else:
        S10 = np.zeros((2, 2))
    if n >= 2:
        S01 = np.cov(np.vstack([V01_1, V01_2]), ddof=1)
    else:
        S01 = np.zeros((2, 2))
    S = S10 / m + S01 / n

    diff = auc_1 - auc_2
    var_diff = float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])

    if var_diff <= 1e-12:
        # 兩模型機率完全等價或退化
        return {
            "auc_1": auc_1, "auc_2": auc_2, "diff": diff,
            "var_diff": var_diff, "z": np.nan, "p_value": 1.0 if diff == 0 else np.nan,
            "degenerate": True, "reason": "zero variance",
        }

    z = diff / np.sqrt(var_diff)
    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {
        "auc_1": auc_1, "auc_2": auc_2, "diff": diff,
        "var_diff": var_diff, "z": float(z), "p_value": float(p_value),
        "degenerate": False, "reason": "",
    }


__all__ = ["delong_test"]
