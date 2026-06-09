"""
data_io/deterministic_encoders.py
==================================

職責：將原始連續量化的 Country、Ethnicity 欄位轉為符合 §1.3 規範的形式。
這兩個轉換是「決定性映射」(deterministic / data-independent)，亦即
轉換規則完全由領域常識決定，不依賴訓練集統計量，因此理論上不存在資料洩漏
風險。然而為了維持「所有特徵工程皆封裝於 Pipeline 內」的設計哲學
(以便 cross_validate / GridSearchCV 可在折內套用)，我們仍將其包裝為
sklearn-compatible Transformer。

兩個轉換器：
  EthnicityBinarizer  -- 將 ethnicity 之原始實數轉為 1{White}
  CountryGrouper      -- 將 country 之原始實數轉為 {UK, USA, Other} 字串
                         (後續再交由 TargetEncoder 進行 E[y|country] 編碼)

設計理由：
  ethnicity 原始 7 類，其中 White 佔 91.25%。若直接 One-Hot 並施加 ℓ_1
  懲罰，極稀疏類別 (n<5) 之係數會極不穩定。二值化將群體區隔縮到一個
  自由度，最大化資訊保留同時穩定懲罰路徑。

  country 原始 7 類，UK 與 USA 合計 84.9%。三類分組減少自由度後採
  Target Encoding 將 7-1=6 維 dummy 壓縮為 1 維連續變數。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# 原始量化值 (依 UCI 資料說明 ml_sci_raw_data.txt)
_WHITE_RAW = -0.31685
_UK_RAW = 0.96082
_USA_RAW = -0.57009
_NUMERIC_TOL = 1e-4  # 浮點誤差容忍度


class EthnicityBinarizer(BaseEstimator, TransformerMixin):
    """
    f(ethnicity) = 1 if ethnicity ≈ -0.31685 (White) else 0

    Sklearn-compat：fit() 為 no-op，transform() 純函數。
    """

    def __init__(self, ethnicity_col: str = "ethnicity",
                 output_col: str = "ethnicity_white"):
        self.ethnicity_col = ethnicity_col
        self.output_col = output_col

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.ethnicity_col in X_out.columns:
            X_out[self.output_col] = (
                np.isclose(X_out[self.ethnicity_col], _WHITE_RAW, atol=_NUMERIC_TOL)
                .astype(int)
            )
        return X_out


class CountryGrouper(BaseEstimator, TransformerMixin):
    """
    將 country (連續實數) 映射為類別字串 {UK, USA, Other}。

    為何不在這裡直接做 Target Encoding？
    因為 Target Encoding 需要 y，必須留在 fit-on-train-only 的位置。
    這裡僅做決定性的字串映射；隨後的 sklearn.preprocessing.TargetEncoder
    會在 ColumnTransformer 中以折內擬合方式接手。
    """

    def __init__(self, country_col: str = "country",
                 output_col: str = "country_grouped"):
        self.country_col = country_col
        self.output_col = output_col

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.country_col not in X_out.columns:
            return X_out

        def _map(v: float) -> str:
            if np.isclose(v, _UK_RAW, atol=_NUMERIC_TOL):
                return "UK"
            if np.isclose(v, _USA_RAW, atol=_NUMERIC_TOL):
                return "USA"
            return "Other"

        X_out[self.output_col] = X_out[self.country_col].map(_map)
        return X_out


def apply_deterministic_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    一次套用兩個決定性編碼。注意：此函數不依賴 y，因此可以在 train/test split
    之前安全套用 (與其在 Pipeline 內每折重算造成不必要的計算)。

    若想要在 Pipeline 內統一處理 (例如要做 Permutation Importance 而希望
    transformer 完整鏈接)，可改用 EthnicityBinarizer + CountryGrouper。
    """
    X_out = EthnicityBinarizer().fit_transform(X)
    X_out = CountryGrouper().fit_transform(X_out)
    return X_out


__all__ = [
    "EthnicityBinarizer",
    "CountryGrouper",
    "apply_deterministic_encoding",
]
