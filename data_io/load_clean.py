"""
data_io/load_clean.py
=====================

職責：
  (1) 由 UCI Repository 取得 Drug Consumption (Quantified) 資料集 (ID=373)；
  (2) 清洗 Semer 虛構藥物的過度宣稱者 (Over-claimer)。

統計學定位：
  Semer (Semeron) 為設計之心理測量陷阱題 (Lie Scale / Trap question)。在 §1.1，
  凡 Semer 自報等級 >= CL1 之觀測值代表受試者於虛構藥物上自承使用，違反基本
  反應誠實性 (Response Validity)。保留會對

      argmin_β  L(β) + λ ||β||_p,   p ∈ {1, 2}

  之懲罰式 M-估計量造成估計偏誤 (Estimation Bias)，因為這些觀測值之 design
  matrix x_i 並未對應到真實的母體分布。

  完整清洗準則：(a) Semer == CL0 之子集，(b) Semer 欄位於建模階段完全捨棄。

預期結果：N_clean = 1,877 (與 EDA 報告 §3.3 一致)。
"""

from __future__ import annotations
from typing import Tuple
import pandas as pd

# 延遲匯入 ucimlrepo，避免測試環境無此依賴時 import 失敗
def _fetch_uci(dataset_id: int = 373):
    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo(id=dataset_id)


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """從 UCI repo 取得原始特徵矩陣 X 與目標矩陣 y。"""
    ds = _fetch_uci(373)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    return X, y


def remove_semer_overclaimers(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    semer_col: str = "semer",
    valid_value: str = "CL0",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    依規劃書 §1.1 移除 Semer >= CL1 的觀測值，並從 y 移除 semer 欄位。

    參數
    ----
    X : (N, p) 原始特徵矩陣
    y : (N, m) 原始目標矩陣 (含 semer 欄位)
    semer_col : Semer 欄位名稱
    valid_value : 視為「誠實作答」之分類等級

    回傳
    ----
    (X_clean, y_clean) 兩者皆為長度 N_clean 之 DataFrame；y_clean 已刪除 semer。
    """
    if semer_col not in y.columns:
        raise KeyError(
            f"Expected target column '{semer_col}' for trap-question filtering, "
            f"got columns: {list(y.columns)}"
        )

    valid_mask = (y[semer_col] == valid_value)
    X_clean = X.loc[valid_mask].copy()
    y_clean = y.loc[valid_mask].drop(columns=[semer_col]).copy()

    # Sanity check：依 EDA 應為 1877
    expected = 1877
    if len(X_clean) != expected:
        # 不 raise，僅警告；新版本資料若有微調仍可繼續
        import warnings
        warnings.warn(
            f"Cleaned sample size {len(X_clean)} differs from EDA-reported {expected}.",
            RuntimeWarning,
        )
    return X_clean, y_clean


__all__ = ["load_raw", "remove_semer_overclaimers"]
