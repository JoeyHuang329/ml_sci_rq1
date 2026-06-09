"""
models/lasso.py
===============

方法 B (Main)：ℓ_1 正規化邏輯迴歸 (Lasso Logistic Regression)。

數學形式 (§2.4)：
    β̂^Lasso_d = argmin_β  −(1/n) Σ_i [ y_i log p_i + (1 − y_i) log(1 − p_i) ]
                              + λ ||β||₁
    p_i = σ( β_0 + x_i^T β ),   σ(z) = 1 / (1 + e^{-z})

統計學優勢：
  (1) 邏輯迴歸係數 β̂_j 具直接的 log-odds-ratio 詮釋；
      在 StandardScaler 後，exp(β̂_j) = 該特質提升 1 σ 之 odds-ratio。
  (2) ℓ_1 懲罰自動執行特徵選擇 (Tibshirani 1996)；
      共線性特徵中 (例如 SS 與 Impulsive) 之「擇一保留」行為本身即提供
      診斷訊息 (與 RQ2 之穩定性分析呼應)。
  (3) 對極不均衡藥物，啟用 class_weight='balanced' 以使用反頻率加權，
      等價於對少數類別樣本給予 1/p̂ 的權重。

Bootstrap CI (§2.4 末段)：
  Lasso 估計量不具封閉形式之漸近分布 (Knight & Fu 2000 指出 ℓ_1 解
  在 β_j = 0 處之分布有點質量，非高斯)。故採非參數 Bootstrap：
      For b = 1, ..., B:
        (X*_b, y*_b) ← 對 (X_train, y_train) 之放回抽樣 (n samples)
        β̂*_b ← 對 (X*_b, y*_b) 重新擬合 (固定 λ = λ̂^CV)
      CI_{j} = [ Q_{2.5}(β̂*_{b,j}), Q_{97.5}(β̂*_{b,j}) ]

  注意：固定 λ 之 Bootstrap 並未納入「λ 選擇之不確定性」，是常見的
  簡化方法 (post-selection inference 議題)。若需嚴格之事後選擇推論，
  應在每個 bootstrap 樣本內重做 CV，但計算成本將提升一個量級。
  本實作於 docstring 與 README 標註此限制。

修正之前 notebook 的問題：
  原 notebook §3.5.2 之 y_prob_lasso 被誤設為 np.random.uniform 佔位符，
  導致 DeLong 檢定全部失效。此模組將 best_pipe 與測試集機率
  y_pred_prob 一併存入 LassoResult，下游評估直接取用，徹底杜絕此類錯誤。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from config.settings import (
    PSYCH_FEATURES, LASSO_CS, CV_FOLDS, GLOBAL_SEED,
    IMBALANCE_THRESHOLD, N_BOOTSTRAP, CI_LOWER_Q, CI_UPPER_Q, FULL_FEATURE_ORDER,
)
from preprocessing import get_default_pipeline, psych_indices_in_default_pipeline
from splits import DrugSplit


@dataclass
class LassoResult:
    drug: str
    best_C: float
    class_weight: Optional[str]
    positive_rate: float
    cv_best_auc: float
    coef_psych: np.ndarray            # (7,) 點估計
    ci_lower_psych: np.ndarray        # (7,) Bootstrap 2.5 分位
    ci_upper_psych: np.ndarray        # (7,) Bootstrap 97.5 分位
    boot_coefs: np.ndarray = field(repr=False)   # (B, 7) raw bootstrap matrix
    pipeline: Pipeline = field(repr=False)
    y_pred_prob_test: np.ndarray = field(repr=False)    # 測試集正樣本機率


def _build_lasso_pipeline(C: float, class_weight: Optional[str],
                          random_state: int = GLOBAL_SEED) -> Pipeline:
    """組裝 (preprocessor) → (LogisticRegression with ℓ_1) Pipeline。"""
    return Pipeline([
        ("preprocessor", get_default_pipeline()),
        ("lasso", LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=random_state,
        )),
    ])


def fit_lasso_single(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    Cs: np.ndarray = LASSO_CS,
    cv: int = CV_FOLDS,
    n_bootstrap: int = N_BOOTSTRAP,
    imbalance_threshold: float = IMBALANCE_THRESHOLD,
    random_state: int = GLOBAL_SEED,
) -> LassoResult:
    """對單一藥物擬合 Lasso，並執行 Bootstrap CI。"""
    pos_rate = float(y_train.mean())
    cw: Optional[str] = "balanced" if pos_rate < imbalance_threshold else None

    # ---- (1) 10-Fold CV 搜尋最佳 C (= 1/λ) ----
    pipe = Pipeline([
        ("preprocessor", get_default_pipeline()),
        ("lasso", LogisticRegression(
            penalty="l1", solver="saga",
            class_weight=cw, max_iter=2000, random_state=random_state,
        )),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={"lasso__C": Cs},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_pipe: Pipeline = grid.best_estimator_
    best_C = float(best_pipe.named_steps["lasso"].C)
    cv_best_auc = float(grid.best_score_)

    # ---- (2) 點估計係數 ----
    coef_full = best_pipe.named_steps["lasso"].coef_.ravel()
    sl = psych_indices_in_default_pipeline()
    coef_psych = coef_full[sl]

    # ---- (3) 測試集機率 (供 DeLong 檢定使用) ----
    y_pred_prob = best_pipe.predict_proba(X_test)[:, 1]

    # ---- (4) Bootstrap CI ----
    # 固定 C = best_C；每次重抽樣完整重擬合 pipeline (含 TargetEncoder)
    # 以保持其於不同 bootstrap 樣本上的擬合一致性。
    boot_coefs = np.empty((n_bootstrap, len(coef_psych)), dtype=float)
    for b in range(n_bootstrap):
        X_b, y_b = resample(
            X_train, y_train,
            replace=True,
            n_samples=len(X_train),
            random_state=random_state + b + 1,   # 內外層獨立播種
        )
        # 跳過退化樣本 (極端不均衡藥物 + 小樣本時可能 bootstrap 出全 0 / 全 1)
        if y_b.nunique() < 2:
            boot_coefs[b, :] = np.nan
            continue
        pipe_b = _build_lasso_pipeline(C=best_C, class_weight=cw,
                                       random_state=random_state)
        pipe_b.fit(X_b, y_b)
        boot_coefs[b, :] = pipe_b.named_steps["lasso"].coef_.ravel()[sl]

    # 以 nanpercentile 容忍退化樣本
    ci_lo = np.nanpercentile(boot_coefs, CI_LOWER_Q, axis=0)
    ci_hi = np.nanpercentile(boot_coefs, CI_UPPER_Q, axis=0)

    return LassoResult(
        drug=y_train.name if y_train.name else "drug",
        best_C=best_C,
        class_weight=cw,
        positive_rate=pos_rate,
        cv_best_auc=cv_best_auc,
        coef_psych=coef_psych,
        ci_lower_psych=ci_lo,
        ci_upper_psych=ci_hi,
        boot_coefs=boot_coefs,
        pipeline=best_pipe,
        y_pred_prob_test=y_pred_prob,
    )


def fit_lasso_all(
    X: pd.DataFrame,
    y_binary: pd.DataFrame,
    splits: Dict[str, DrugSplit],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    verbose: bool = True,
) -> Dict[str, LassoResult]:
    """對 15 種藥物分別擬合 Lasso，回傳 dict[drug -> LassoResult]。"""
    feature_cols = [c for c in FULL_FEATURE_ORDER if c in X.columns]
    results: Dict[str, LassoResult] = {}
    for drug, sp in splits.items():
        if verbose:
            print(f"  [Lasso] fitting drug='{drug}' "
                  f"(n_train={sp.n_train}, n_test={sp.n_test})")
        X_tr = X.loc[sp.train_idx, feature_cols]
        X_te = X.loc[sp.test_idx, feature_cols]
        y_tr = y_binary.loc[sp.train_idx, drug].rename(drug)
        y_te = y_binary.loc[sp.test_idx, drug].rename(drug)
        results[drug] = fit_lasso_single(
            X_tr, y_tr, X_te, y_te,
            n_bootstrap=n_bootstrap,
        )
    return results


def coefficients_with_ci(results: Dict[str, LassoResult]) -> pd.DataFrame:
    """
    長格式 (long-format) 推論表：
      drug, feature, coef, ci_lower, ci_upper, significant_at_95
    """
    rows = []
    for drug, r in results.items():
        for j, feat in enumerate(PSYCH_FEATURES):
            lo, hi = float(r.ci_lower_psych[j]), float(r.ci_upper_psych[j])
            b = float(r.coef_psych[j])
            rows.append({
                "drug": drug,
                "feature": feat,
                "coef": b,
                "ci_lower": lo,
                "ci_upper": hi,
                # CI 不含 0 即視為 95% 信賴水準下顯著
                "significant_at_95": bool((lo > 0) or (hi < 0)),
            })
    return pd.DataFrame(rows)


__all__ = [
    "LassoResult", "fit_lasso_single", "fit_lasso_all", "coefficients_with_ci",
]
