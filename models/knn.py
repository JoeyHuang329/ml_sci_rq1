"""
models/knn.py
=============

方法 C (Comparison)：k-Nearest Neighbors 分類器。

數學形式 (§2.5)：
    p̂^KNN_d(x_0) = (1/k) Σ_{i ∈ N_k(x_0)} y^{(d)}_i

其中 N_k(x_0) 為標準化心理特徵空間 R^7 中歐式距離下 x_0 之 k 個近鄰。

統計學定位 (§2.5)：
  KNN 屬無母數方法 (Stone 1977 證明其 universal consistency)。
  與 Lasso 邏輯迴歸對比可回答結構誤設 (Structural Misspecification) 之問題：

      AUC^KNN  >>  AUC^Lasso   ⇒   存在顯著的非線性或交互作用結構
      AUC^KNN  ≈   AUC^Lasso   ⇒   線性附加模型已捕捉主要訊號

  此種「線性 vs 無母數」的對比，在 Hastie, Tibshirani & Friedman (ESL,
  §2.5) 中被稱為「靈活性 vs 偏誤-變異權衡」的標準診斷實驗。

距離度量純化：
  歐式距離 d(x_i, x_j) = √Σ_l (x_il − x_jl)² 對混合型變數
  (連續心理計量 vs 離散人口統計) 之解釋力不對等。本研究嚴格將距離計算
  限制於標準化後的 7 維 psych space，其餘變數於 KNN 階段被丟棄。
  維度 p=7 遠小於樣本量 n ≈ 1500，故維度詛咒影響輕微。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config.settings import (
    KNN_K_GRID, CV_FOLDS, GLOBAL_SEED, PSYCH_FEATURES,
)
from preprocessing import get_psych_only_pipeline
from splits import DrugSplit


@dataclass
class KNNResult:
    drug: str
    best_k: int
    cv_best_auc: float
    pipeline: Pipeline = field(repr=False)
    y_pred_prob_test: np.ndarray = field(repr=False)


def fit_knn_single(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    k_grid: List[int] = None,
    cv: int = CV_FOLDS,
) -> KNNResult:
    if k_grid is None:
        k_grid = list(KNN_K_GRID)

    pipe = Pipeline([
        ("preprocessor", get_psych_only_pipeline()),
        ("knn", KNeighborsClassifier(metric="euclidean", n_jobs=-1)),
    ])

    grid = GridSearchCV(
        pipe,
        param_grid={"knn__n_neighbors": k_grid},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_pipe: Pipeline = grid.best_estimator_
    best_k = int(best_pipe.named_steps["knn"].n_neighbors)
    y_pred_prob = best_pipe.predict_proba(X_test)[:, 1]

    return KNNResult(
        drug=y_train.name if y_train.name else "drug",
        best_k=best_k,
        cv_best_auc=float(grid.best_score_),
        pipeline=best_pipe,
        y_pred_prob_test=y_pred_prob,
    )


def fit_knn_all(
    X: pd.DataFrame,
    y_binary: pd.DataFrame,
    splits: Dict[str, DrugSplit],
    *,
    verbose: bool = True,
) -> Dict[str, KNNResult]:
    # KNN 僅需 psych features，但 X 可包含全欄；ColumnTransformer 會挑出
    psych_cols = [c for c in PSYCH_FEATURES if c in X.columns]
    if len(psych_cols) != len(PSYCH_FEATURES):
        missing = set(PSYCH_FEATURES) - set(psych_cols)
        raise KeyError(f"Missing psych feature columns: {missing}")

    results: Dict[str, KNNResult] = {}
    for drug, sp in splits.items():
        if verbose:
            print(f"  [KNN] fitting drug='{drug}'")
        X_tr = X.loc[sp.train_idx, psych_cols]
        X_te = X.loc[sp.test_idx, psych_cols]
        y_tr = y_binary.loc[sp.train_idx, drug].rename(drug)
        y_te = y_binary.loc[sp.test_idx, drug].rename(drug)
        results[drug] = fit_knn_single(X_tr, y_tr, X_te, y_te)
    return results


__all__ = ["KNNResult", "fit_knn_single", "fit_knn_all"]
