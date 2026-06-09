"""
reports/tables.py
=================

組裝最終的統一評估表格。這是整個 RQ1 流程的**事實表**，輸出 CSV
供論文撰寫與圖表生成使用。

★ 此模組是修正原 Colab notebook §3.5 三個關鍵 bug 的核心位置 ★

修正一覽：
  Bug A (§3.5.2 第 799 行)：原 notebook 對 y_prob_lasso 使用
                            np.random.uniform 佔位符。
        修正：直接由 LassoResult.y_pred_prob_test 取出真實機率。

  Bug B (§3.5.3 第 884 行)：原 notebook 為了重算 Ridge MSE，
                            硬塞 alpha=1.0 作為 proxy，捨棄 CV 找到之 λ̂。
        修正：直接由已 refit 之 RidgeResult.pipeline 在測試集上預測。

  Bug C (§3.5 末段)：原 notebook 為了取得測試集機率，將 Ridge 與
                     Lasso 整個重新訓練了一次 (浪費約 15 × 兩個 grid
                     search 的時間)。
        修正：所有測試集機率與預測值，皆從一次擬合的 *_Result 物件中取出。

統計學完整性：
  本模組不重新訓練任何模型，僅消費已擬合的 pipeline，從而保證：
    (i) 結果完全可重現；
    (ii) Ridge / Lasso / KNN 評估皆使用同一份「保留至最後」之測試集；
    (iii) 任何 CV 過程之超參數選擇被完整保留。
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from config.settings import PSYCH_FEATURES, FULL_FEATURE_ORDER
from models import RidgeResult, LassoResult, KNNResult
from splits import DrugSplit
from evaluation import (
    compute_classification_metrics, compute_regression_metrics,
    bootstrap_auc_ci, delong_test,
    auc_one_sided_test, benjamini_hochberg,
)


def build_unified_evaluation_table(
    splits: Dict[str, DrugSplit],
    y_binary: pd.DataFrame,
    z_continuous: pd.DataFrame,
    X: pd.DataFrame,
    ridge_results: Dict[str, RidgeResult],
    lasso_results: Dict[str, LassoResult],
    knn_results: Dict[str, KNNResult],
    *,
    n_bootstrap_auc: int = 1000,
) -> pd.DataFrame:
    """
    回傳含以下欄位的 DataFrame，index = drug：

      [Lasso]   AUC, AUC_CI_lo, AUC_CI_hi, AUPRC, Brier
      [KNN]     AUC, AUC_CI_lo, AUC_CI_hi, AUPRC, Brier
      [Ridge]   MSE
      [Lasso H0: AUC=0.5]  p_value, q_value (FDR), significant
      [Lasso vs KNN DeLong] z, p_value, q_value (FDR), significant
      [DESC]    pos_rate (測試集), best_C (Lasso), best_k (KNN), best_alpha (Ridge)
    """
    rows = []
    auc_h0_pvals = []
    delong_pvals = []
    drug_order = list(splits.keys())

    for drug in drug_order:
        sp = splits[drug]
        y_te = y_binary.loc[sp.test_idx, drug].to_numpy().astype(int)
        z_te = z_continuous.loc[sp.test_idx, drug].to_numpy().astype(float)

        # ----- 取得各模型測試集預測 (★ 無重新訓練 ★) -----
        p_lasso = lasso_results[drug].y_pred_prob_test
        p_knn = knn_results[drug].y_pred_prob_test
        ridge_pipe = ridge_results[drug].pipeline
        ridge_feature_cols = [c for c in FULL_FEATURE_ORDER if c in X.columns]
        X_te = X.loc[sp.test_idx, ridge_feature_cols]
        z_pred_ridge = ridge_pipe.predict(X_te)

        # ----- 計算指標 -----
        m_lasso = compute_classification_metrics(y_te, p_lasso)
        m_knn = compute_classification_metrics(y_te, p_knn)
        m_ridge = compute_regression_metrics(z_te, z_pred_ridge)

        # ----- Bootstrap 95% CI for AUC (兩個分類模型分別計算) -----
        _, auc_lo_lasso, auc_hi_lasso = bootstrap_auc_ci(
            y_te, p_lasso, n_bootstrap=n_bootstrap_auc,
        )
        _, auc_lo_knn, auc_hi_knn = bootstrap_auc_ci(
            y_te, p_knn, n_bootstrap=n_bootstrap_auc,
        )

        # ----- H_0: AUC = 0.5 (單側) for Lasso -----
        h0 = auc_one_sided_test(y_te, p_lasso)
        auc_h0_pvals.append(h0["p_value"] if not h0["degenerate"] else np.nan)

        # ----- DeLong: Lasso vs KNN (雙側) -----
        dl = delong_test(y_te, p_lasso, p_knn)
        delong_pvals.append(dl["p_value"] if not dl["degenerate"] else np.nan)

        rows.append({
            "drug": drug,
            "n_test": sp.n_test,
            "pos_rate_test": float(np.mean(y_te)),
            # ---- Lasso ----
            "Lasso_AUC": m_lasso["auc"],
            "Lasso_AUC_CI_lo": auc_lo_lasso,
            "Lasso_AUC_CI_hi": auc_hi_lasso,
            "Lasso_AUPRC": m_lasso["auprc"],
            "Lasso_Brier": m_lasso["brier"],
            "Lasso_best_C": lasso_results[drug].best_C,
            "Lasso_class_weight": str(lasso_results[drug].class_weight),
            # ---- KNN ----
            "KNN_AUC": m_knn["auc"],
            "KNN_AUC_CI_lo": auc_lo_knn,
            "KNN_AUC_CI_hi": auc_hi_knn,
            "KNN_AUPRC": m_knn["auprc"],
            "KNN_Brier": m_knn["brier"],
            "KNN_best_k": knn_results[drug].best_k,
            # ---- Ridge ----
            "Ridge_MSE": m_ridge["mse"],
            "Ridge_best_alpha": ridge_results[drug].best_alpha,
            # ---- 推論結果 (FDR 校正前) ----
            "Lasso_H0_AUC_p_raw": h0["p_value"] if not h0["degenerate"] else np.nan,
            "DeLong_p_raw": dl["p_value"] if not dl["degenerate"] else np.nan,
            "DeLong_z": dl["z"] if not dl["degenerate"] else np.nan,
        })

    df = pd.DataFrame(rows).set_index("drug")

    # ----- FDR 校正 (兩組 p-value 各自校正) -----
    rej_h0, q_h0 = benjamini_hochberg(auc_h0_pvals)
    rej_dl, q_dl = benjamini_hochberg(delong_pvals)

    df["Lasso_H0_AUC_q_FDR"] = q_h0
    df["Lasso_AUC_signif_vs_0.5"] = rej_h0
    df["DeLong_q_FDR"] = q_dl
    df["Lasso_vs_KNN_signif"] = rej_dl

    return df


def build_ridge_coefficient_table(
    ridge_results: Dict[str, RidgeResult],
) -> pd.DataFrame:
    """
    Ridge 標準化係數矩陣 B̂：rows = 心理特徵 (7), columns = 藥物 (15)。
    """
    return pd.DataFrame(
        {drug: r.coef_psych for drug, r in ridge_results.items()},
        index=PSYCH_FEATURES,
    )


def build_lasso_inference_table(
    lasso_results: Dict[str, LassoResult],
) -> pd.DataFrame:
    """
    Lasso 之長格式 (long-format) 推論表：
      drug × feature × (coef, ci_lo, ci_hi, exp(coef)=OR, significant)
    """
    rows = []
    for drug, r in lasso_results.items():
        for j, feat in enumerate(PSYCH_FEATURES):
            b = float(r.coef_psych[j])
            lo = float(r.ci_lower_psych[j])
            hi = float(r.ci_upper_psych[j])
            rows.append({
                "drug": drug,
                "feature": feat,
                "beta": b,
                "exp_beta_odds_ratio": float(np.exp(b)) if not np.isnan(b) else np.nan,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_excludes_zero": bool((lo > 0) or (hi < 0)),
            })
    return pd.DataFrame(rows)


__all__ = [
    "build_unified_evaluation_table",
    "build_ridge_coefficient_table",
    "build_lasso_inference_table",
]
