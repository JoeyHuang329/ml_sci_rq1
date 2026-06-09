"""
data_io/targets.py
==================

職責：依 §2.2 將 CL0-CL6 之原始序列分類目標變數操作化為
  (a) 二值高風險指標 y ∈ {0, 1}    -- 給 Lasso / KNN 用
  (b) 連續化等距等級 z ∈ {0,...,6}  -- 給 Ridge baseline 用

統計學討論：
  - (a) 二值化以 CL3 為閾值。CL3 = "Used in last year"，操作化為
        "近期活躍使用者"。此閾值兼顧 (i) 行為心理學上「活躍個案」的
        最小單位，(ii) 在 15 種藥物上避免極端 imbalance (使 AUC 不退化)。

  - (b) 連續化將序列等級視為等距，違反 OLS 同質變異與正態殘差假設
        (CL0 與 CL1 之心理距離未必等同 CL5 與 CL6)。然作為 baseline
        其價值在於提供「不剔除任何訊號條件下之線性邊際效應估計」，
        並可與 Ridge ℓ_2 正規化結合以穩定共線性下之係數。
"""

from __future__ import annotations
from typing import Iterable
import pandas as pd

from config.settings import HIGH_RISK_CL, CL_TO_ORDINAL


def binarize_targets(y_clean: pd.DataFrame, target_drugs: Iterable[str]) -> pd.DataFrame:
    """
    依 §2.2 將目標變數二值化：
        y^{(d)}_i = 1[ CL^{(d)}_i ∈ {CL3, CL4, CL5, CL6} ]

    回傳 N × |D| 的二元 DataFrame。
    """
    y_binary = pd.DataFrame(index=y_clean.index)
    for drug in target_drugs:
        if drug not in y_clean.columns:
            raise KeyError(f"Drug '{drug}' not present in y_clean columns")
        y_binary[drug] = y_clean[drug].isin(HIGH_RISK_CL).astype(int)
    return y_binary


def ordinalize_targets(y_clean: pd.DataFrame, target_drugs: Iterable[str]) -> pd.DataFrame:
    """
    依 §2.3 將 CL0-CL6 映射為連續整數 z ∈ {0, 1, ..., 6}。
    """
    z = pd.DataFrame(index=y_clean.index)
    for drug in target_drugs:
        if drug not in y_clean.columns:
            raise KeyError(f"Drug '{drug}' not present in y_clean columns")
        z[drug] = y_clean[drug].map(CL_TO_ORDINAL)
        if z[drug].isna().any():
            raise ValueError(
                f"Unknown CL label encountered in column '{drug}'. "
                f"Expected only {sorted(CL_TO_ORDINAL.keys())}."
            )
    return z


__all__ = ["binarize_targets", "ordinalize_targets"]
