"""
evaluation/metrics.py
=====================

統一的評估指標計算：

  AUC-ROC   ── 排序型 (rank-based) 預測效能，對閾值與類別比例皆不敏感；
               為主要指標 (與 §2.6 對齊)。AUC = P(p̂(x|y=1) > p̂(x|y=0))，
               即無母數 Mann-Whitney U 統計量的標準化形式。

  AUPRC     ── Average Precision，於不均衡資料下優於 AUC；
               依 Saito & Rehmsmeier (2015)，當 positive class 稀少時，
               AUC-ROC 會因 TN 主導而高估，AUPRC 則直接反映精準度。

  Brier     ── (1/n) Σ_i (p̂_i − y_i)²，校準度指標 (calibration)；
               衡量「預測機率」與「真實事件率」之吻合程度，是分布層面
               (而非排序層面) 之品質指標。

  MSE       ── Ridge baseline 用，連續化目標 z 上的均方誤差。

附 Bootstrap CI：
  非參數法估計 AUC 之 95% CI。對測試集 (X_test, y_test, p̂_test) 進行
  B 次放回抽樣，每次重抽樣計算 AUC，取 2.5/97.5 分位數。
  此 CI 反映「測試集有限性」之不確定性，非「模型擬合」之不確定性。
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, mean_squared_error,
)

from config.settings import N_BOOTSTRAP, CI_LOWER_Q, CI_UPPER_Q, GLOBAL_SEED


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """同時計算 AUC / AUPRC / Brier。輸入皆為 1D ndarray。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def compute_regression_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> dict:
    z_true = np.asarray(z_true).astype(float)
    z_pred = np.asarray(z_pred).astype(float)
    return {"mse": float(mean_squared_error(z_true, z_pred))}


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = GLOBAL_SEED,
) -> Tuple[float, float, float]:
    """
    對 AUC 進行非參數 percentile Bootstrap CI。

    回傳 (auc_point, ci_lower, ci_upper)。

    退化情況：若某次 bootstrap 樣本中只有單一類別，AUC 未定義，
    跳過該次重抽樣以 nanpercentile 計算分位數。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auc_point = float(roc_auc_score(y_true, y_prob))

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    aucs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(y_b, p_b)
    ci_lo = float(np.nanpercentile(aucs, CI_LOWER_Q))
    ci_hi = float(np.nanpercentile(aucs, CI_UPPER_Q))
    return auc_point, ci_lo, ci_hi


__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "bootstrap_auc_ci",
]
