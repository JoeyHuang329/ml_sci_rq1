"""
config/settings.py
==================

集中管理 RQ1 全部實驗的常數，包括：

(1) 隨機種子協定 (Random Seed Protocol)
    依規劃書 §1.6，所有外層流程 (切分、Bootstrap、CV) 共用 GLOBAL_SEED，
    內層蒙地卡羅迴圈以 GLOBAL_SEED + b 獨立播種，
    確保在分散式或多檔架構下仍可完全重現。

(2) 特徵集合定義 X_psych / X_demo / X_full
    將心理計量與人口統計層級分離，方便後續 RQ2-RQ4 共用，並避免
    Pipeline 內部因為 ColumnTransformer 輸出順序而引起的脆弱性。

(3) 目標變數與藥物清單
    依 §2.2，排除 Semer (虛構)、Alcohol/Caff/Choc (極端不均衡)，
    最終 |D| = 15。

(4) 二值化閾值 CL3
    依 §2.2，將 "近一年內使用 (recently active)" 操作化為高風險指標，
    本資料集屬序列等級 CL0-CL6，CL3 為「過去一年」。

(5) 超參數搜尋網格
    所有網格皆為對數均勻 (Log-uniform)，符合凸優化中正規化項的尺度不變性。
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# 1. 隨機種子協定
# ---------------------------------------------------------------------------
GLOBAL_SEED: int = 42

# ---------------------------------------------------------------------------
# 2. 特徵集合定義
# ---------------------------------------------------------------------------
# 心理計量 (NEO-FFI-R + BIS-11 + ImpSS) ——皆已由 UCI 量化為實數
# 注意：原始資料集將 "Impulsive" 拼寫為 "impuslive"，保留以匹配欄位名
PSYCH_FEATURES: list[str] = [
    "nscore", "escore", "oscore", "ascore", "cscore",
    "impuslive", "ss",
]

# 人口統計學原始欄位 (尚未經編碼)
DEMO_FEATURES_RAW: list[str] = [
    "age", "gender", "education", "country", "ethnicity",
]

# 經 EthnicityBinarizer / CountryGrouper 編碼後的衍生欄位
DEMO_FEATURES_ENCODED: list[str] = [
    "age", "gender", "education", "ethnicity_white", "country_grouped",
]

# 進入 Pipeline 之欄位順序 (重要：固定順序避免 ColumnTransformer 輸出脆弱性)
FULL_FEATURE_ORDER: list[str] = PSYCH_FEATURES + [
    "country_grouped", "age", "gender", "education", "ethnicity_white",
]

# ---------------------------------------------------------------------------
# 3. 目標變數定義
# ---------------------------------------------------------------------------
# 完全排除：合法日常物質 (極端不均衡會導致 AUC 退化) + 虛構藥物
EXCLUDED_DRUGS: list[str] = ["alcohol", "caff", "choc", "semer"]

# 15 種建模目標藥物 (依字母順序，與 UCI 欄位匹配)
TARGET_DRUGS: list[str] = [
    "amphet", "amyl", "benzos", "cannabis", "coke",
    "crack", "ecstasy", "heroin", "ketamine", "legalh",
    "lsd", "meth", "mushrooms", "nicotine", "vsa",
]

# 二值化閾值：CL3 = "近一年內使用" 視為高風險樣本 (y=1)
HIGH_RISK_CL: set[str] = {"CL3", "CL4", "CL5", "CL6"}

# 連續化映射 (Ridge baseline 用)：CL0-CL6 -> 0-6 等距
CL_TO_ORDINAL: dict[str, int] = {f"CL{k}": k for k in range(7)}

# ---------------------------------------------------------------------------
# 4. Train/Test 切分常數
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20
CV_FOLDS: int = 10            # K-Fold Stratified CV，K=10

# ---------------------------------------------------------------------------
# 5. 模型超參數搜尋網格 (對數網格)
# ---------------------------------------------------------------------------
# Ridge：lambda ∈ [10^-4, 10^2]，13 點
RIDGE_ALPHAS: np.ndarray = np.logspace(-4, 2, num=13)

# Lasso Logistic：sklearn 的 C = 1/lambda，11 點
LASSO_CS: np.ndarray = np.logspace(-3, 2, num=11)

# KNN：k 採奇數+幾何序列以兼顧局部與全域
KNN_K_GRID: list[int] = [5, 7, 10, 15, 25, 50, 100]

# 類別不均衡閾值：低於此比例採 class_weight="balanced"
IMBALANCE_THRESHOLD: float = 0.15

# ---------------------------------------------------------------------------
# 6. Bootstrap 推論
# ---------------------------------------------------------------------------
N_BOOTSTRAP: int = 1000       # 95% CI 之 Bootstrap 重抽樣次數
CI_LOWER_Q: float = 2.5
CI_UPPER_Q: float = 97.5

# ---------------------------------------------------------------------------
# 7. 多重比較校正
# ---------------------------------------------------------------------------
FDR_ALPHA: float = 0.05       # Benjamini-Hochberg 控制 FDR 之顯著水準

# ---------------------------------------------------------------------------
# 8. 路徑常數
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
TABLES_DIR: Path = ARTIFACTS_DIR / "tables"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"


def ensure_dirs() -> None:
    """建立所有必要的輸出目錄（若不存在）"""
    for d in (ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "GLOBAL_SEED",
    "PSYCH_FEATURES", "DEMO_FEATURES_RAW", "DEMO_FEATURES_ENCODED", "FULL_FEATURE_ORDER",
    "EXCLUDED_DRUGS", "TARGET_DRUGS", "HIGH_RISK_CL", "CL_TO_ORDINAL",
    "TEST_SIZE", "CV_FOLDS",
    "RIDGE_ALPHAS", "LASSO_CS", "KNN_K_GRID", "IMBALANCE_THRESHOLD",
    "N_BOOTSTRAP", "CI_LOWER_Q", "CI_UPPER_Q",
    "FDR_ALPHA",
    "PROJECT_ROOT", "ARTIFACTS_DIR", "TABLES_DIR", "FIGURES_DIR", "MODELS_DIR",
    "ensure_dirs",
]
