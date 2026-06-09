"""
models/ridge.py
===============

方法 A (Baseline)：Ridge 線性迴歸於連續化 CL 等級 z ∈ {0,...,6}。

數學形式 (§2.3)：
    β̂^Ridge_d = argmin_β   (1 / 2n) Σ_i (z^{(d)}_i − β_0 − x_i^T β)² + λ ||β||₂²

設計理由：
  - ℓ_2 懲罰下，所有 β̂_j 同時保留 (不像 ℓ_1 會自動稀疏)。在共線性
    特徵下 (例如 SS 與 Impulsive，相關係數 ≈ 0.62)，OLS 之解
    β̂ = (X^T X)^(-1) X^T z 在 X^T X 近奇異時數值不穩；λ I 之收縮
    將條件數 (Condition Number) 控制在 O(λ_max / λ) 內，保證解之
    穩定性 (Hoerl & Kennard, 1970)。

  - 因心理計量特徵已 StandardScaler 標準化，β̂_j 可解讀為
    「該特質提升 1 個標準差 → 預期 CL 等級提升 β̂_j」。

  - z 為序列等距假設 (Linearization of Ordinal Variables)，嚴格而言
    違反 OLS 之同質變異與正態殘差假設。然作為 baseline，其價值在於
    提供 "未稀疏化" 之線性邊際效應地圖。

修正之前 notebook 的問題：
  原 §3.5.3 末段為了重算測試集 Ridge MSE，硬塞 alpha=1.0 作為「proxy」，
  完全棄置 CV 找到的 λ̂。此處改為將 best_alpha_ 與已擬合的 pipeline
  存入 RidgeResult，下游評估直接取用，杜絕「proxy alpha」之統計謬誤。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config.settings import (
    PSYCH_FEATURES, RIDGE_ALPHAS, CV_FOLDS, GLOBAL_SEED, FULL_FEATURE_ORDER,
)
from preprocessing import get_default_pipeline, psych_indices_in_default_pipeline
from splits import DrugSplit


@dataclass
class RidgeResult:
    """
    單一藥物的 Ridge 結果。
    保存最佳 pipeline (含 preprocessor) 以避免下游評估重新訓練。
    """
    drug: str
    best_alpha: float
    cv_best_mse: float            # 對應 best_alpha 之 CV 平均負 MSE 之相反數
    coef_psych: np.ndarray        # shape (7,) 標準化後之 psych 係數
    coef_full: np.ndarray         # shape (12,) 完整輸出空間之係數
    pipeline: Pipeline = field(repr=False)


def fit_ridge_single(
    X_train: pd.DataFrame,
    z_train: pd.Series,
    *,
    alphas: np.ndarray = RIDGE_ALPHAS,
    cv: int = CV_FOLDS,
    random_state: int = GLOBAL_SEED,
) -> RidgeResult:
    """
    對單一藥物擬合 Ridge：以 10-Fold CV 在對數網格搜尋 λ̂，
    準則為負均方誤差 (neg_MSE) 最大化。
    """
    pipe = Pipeline([
        ("preprocessor", get_default_pipeline()),
        ("ridge", Ridge(random_state=random_state)),
    ])

    grid = GridSearchCV(
        pipe,
        param_grid={"ridge__alpha": alphas},
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,            # 在最佳參數下，於全訓練集重新擬合 (供測試集使用)
    )
    grid.fit(X_train, z_train)

    best_pipe: Pipeline = grid.best_estimator_
    best_alpha = float(best_pipe.named_steps["ridge"].alpha)

    coef_full = best_pipe.named_steps["ridge"].coef_.ravel()
    sl = psych_indices_in_default_pipeline()
    coef_psych = coef_full[sl]

    return RidgeResult(
        drug=z_train.name if z_train.name else "drug",
        best_alpha=best_alpha,
        cv_best_mse=float(-grid.best_score_),
        coef_psych=coef_psych,
        coef_full=coef_full,
        pipeline=best_pipe,
    )


def fit_ridge_all(
    X: pd.DataFrame,
    z: pd.DataFrame,
    splits: Dict[str, DrugSplit],
) -> Dict[str, RidgeResult]:
    """
    對 15 種藥物分別擬合 Ridge baseline，回傳 dict[drug -> RidgeResult]。

    輸入
    ----
    X       : 已 deterministic-encoded 之特徵矩陣 (含 country_grouped, ethnicity_white)
    z       : N × |D| 連續化目標
    splits  : 來自 splits.build_per_drug_splits
    """
    feature_cols = [c for c in FULL_FEATURE_ORDER if c in X.columns]
    results: Dict[str, RidgeResult] = {}
    for drug, sp in splits.items():
        X_tr = X.loc[sp.train_idx, feature_cols]
        z_tr = z.loc[sp.train_idx, drug].rename(drug)
        results[drug] = fit_ridge_single(X_tr, z_tr)
    return results


def coefficients_matrix(results: Dict[str, RidgeResult]) -> pd.DataFrame:
    """
    將每個藥物之 coef_psych 整理為 7 (psych) × |D| (drugs) 矩陣 B̂。
    """
    return pd.DataFrame(
        {drug: r.coef_psych for drug, r in results.items()},
        index=PSYCH_FEATURES,
    )


__all__ = [
    "RidgeResult", "fit_ridge_single", "fit_ridge_all", "coefficients_matrix",
]
