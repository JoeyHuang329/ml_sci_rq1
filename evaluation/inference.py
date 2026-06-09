"""
evaluation/inference.py
=======================

統計推論工具：

(1) auc_one_sided_test
    對單一藥物之 AUC，檢定 H_0: AUC = 0.5 vs H_1: AUC > 0.5。
    本研究 §2.1 中 RQ1 之原始假設即為此單側形式。
    採 Mann-Whitney U 統計量之漸近常態近似 (Hanley & McNeil, 1982)：

        Var(AUC) ≈ AUC(1−AUC) + (m−1)·(Q1 − AUC²) + (n−1)·(Q2 − AUC²)
                   ─────────────────────────────────────────────────
                                        m · n
        Q1 = AUC / (2 − AUC),   Q2 = 2·AUC² / (1 + AUC)

    Z = (AUC − 0.5) / SE(AUC),  p = 1 − Φ(Z)

(2) benjamini_hochberg
    對一組 p-value 進行 Benjamini-Hochberg (1995) 多重比較校正，
    控制 False Discovery Rate (FDR) 在 α 水準下。
    為 Bonferroni 之較不保守替代方案，特別適合對多個藥物之同時推論。
"""

from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
from scipy.stats import norm

from config.settings import FDR_ALPHA


def auc_one_sided_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """檢定 AUC > 0.5 (單側)。回傳 auc, se, z, p_value。"""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return {"auc": np.nan, "se": np.nan, "z": np.nan, "p_value": np.nan,
                "degenerate": True}

    auc = float(roc_auc_score(y_true, y_prob))
    m, n = int(pos), int(neg)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var_auc = (auc * (1 - auc)
               + (m - 1) * (q1 - auc * auc)
               + (n - 1) * (q2 - auc * auc)) / (m * n)
    se = float(np.sqrt(max(var_auc, 0.0)))
    if se <= 0.0:
        return {"auc": auc, "se": 0.0, "z": np.inf,
                "p_value": 0.0 if auc > 0.5 else 1.0,
                "degenerate": False}
    z = (auc - 0.5) / se
    p_one_sided = float(1.0 - norm.cdf(z))
    return {"auc": auc, "se": se, "z": float(z),
            "p_value": p_one_sided, "degenerate": False}


def benjamini_hochberg(
    pvals: Sequence[float],
    alpha: float = FDR_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BH (1995) FDR 校正。

    參數
    ----
    pvals : 一組 p-value (可含 NaN，將以 1.0 取代)
    alpha : 控制水準

    回傳
    ----
    (reject, pvals_corrected)
      reject              : 布林陣列，是否在 α 水準下拒絕 H_0
      pvals_corrected     : adjusted p-values (亦稱 q-values)
    """
    p = np.asarray(pvals, dtype=float).copy()
    p[np.isnan(p)] = 1.0
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    # adjusted p_(i) = min_{k >= i} ( n / k · p_(k) )，由大到小回掃保證單調
    factor = n / np.arange(1, n + 1)
    raw_adj = ranked * factor
    # 由後向前取累積最小值，再回填位置
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    pvals_corrected = np.empty(n, dtype=float)
    pvals_corrected[order] = adj_sorted
    reject = pvals_corrected <= alpha
    return reject, pvals_corrected


__all__ = ["auc_one_sided_test", "benjamini_hochberg"]
