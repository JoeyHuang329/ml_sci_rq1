"""
preprocessing/pipeline.py
==========================

工廠函數 (Factory) 提供「未擬合」的 ColumnTransformer，用於：
  1. 心理計量特徵的 StandardScaler  (μ̂, σ̂² 折內擬合)
  2. country_grouped 的 TargetEncoder ( Ê[y|country] 折內擬合)
  3. 其他 passthrough 特徵 (age, gender, education, ethnicity_white)

統計學要點 (§1.5)：
  StandardScaler 與 TargetEncoder 皆需「資料相依」的擬合，必須嚴格遵守
  「fit on train fold only, transform on validation fold」鐵律。將兩者
  封裝為 sklearn Pipeline / ColumnTransformer 後，當外層使用 cross_validate
  或 GridSearchCV 時，scikit-learn 會自動於每個折內呼叫 .fit_transform
  與 .transform，從根本上隔絕資料洩漏 (Data Leakage)。

  特別注意 TargetEncoder 內部仍有 cv 參數 (用於 leave-one-fold-out 平滑)，
  這是 sklearn ≥1.3 之預設行為，可減少對小類別過擬合。

為什麼提供兩個變體？
  - get_default_pipeline()    : 用於 Ridge / Lasso，完整 12 維特徵
  - get_psych_only_pipeline() : 用於 KNN，僅標準化後的 7 維心理特徵
                                 (歐式距離只在 psych space 上具備幾何意義；
                                  人口統計學的離散變數若直接放進距離計算
                                  將扭曲鄰近結構)
"""

from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder

from config.settings import PSYCH_FEATURES, GLOBAL_SEED


# 用於 Ridge / Lasso：完整特徵
def get_default_pipeline() -> ColumnTransformer:
    """
    回傳「未擬合」之 ColumnTransformer：
        ┌────────────────────────────────────────────┐
        │ psych_scale     : StandardScaler  → 7 維   │
        │ country_target  : TargetEncoder   → 1 維   │
        │ remainder       : passthrough     → 4 維   │
        │   (age, gender, education, ethnicity_white)│
        └────────────────────────────────────────────┘
    輸出總維度 = 12。欄位名稱可由 .get_feature_names_out() 取得。
    """
    psych_features = list(PSYCH_FEATURES)  # 防止外部修改

    preprocessor = ColumnTransformer(
        transformers=[
            ("psych_scale", StandardScaler(), psych_features),
            ("country_target",
             TargetEncoder(smooth="auto", cv=5, random_state=GLOBAL_SEED),
             ["country_grouped"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,   # 保留簡潔欄位名供下游解析
    )
    return preprocessor


# 用於 KNN：僅標準化心理特徵，其餘丟棄 (drop)
def get_psych_only_pipeline() -> ColumnTransformer:
    """
    KNN 專用前處理：因為歐式距離須在同質連續空間內計算，且本研究 §2.5
    強調「人格特質與藥物使用的關係」之非線性結構，故距離度量限制於
    標準化後的心理特徵子空間。其餘變數於此階段被丟棄 (remainder='drop')。

    輸出維度 = 7。
    """
    psych_features = list(PSYCH_FEATURES)
    preprocessor = ColumnTransformer(
        transformers=[
            ("psych_scale", StandardScaler(), psych_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def psych_indices_in_default_pipeline() -> slice:
    """
    輔助函數：回傳 default pipeline 輸出中對應 psych_features 之欄位切片。
    由於我們將 psych_scale 作為第一個 transformer 且輸出維度 = 7，
    其位置固定為 [0:7]。若未來新增 transformer，請同步更新此函數
    或改用 .get_feature_names_out() 動態查詢。
    """
    return slice(0, len(PSYCH_FEATURES))


__all__ = [
    "get_default_pipeline",
    "get_psych_only_pipeline",
    "psych_indices_in_default_pipeline",
]
