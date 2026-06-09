"""
splits.py
=========

依規劃書 §1.4 對每種藥物 d 個別執行 80:20 Stratified Train/Test Split，
分層變數為該藥物的二值化標籤 y^{(d)}。

為什麼必須「逐藥物獨立切分」？
  不同藥物的正樣本比例差異極大 (Coke ~25%, Heroin ~5%)。
  若以單一切分通用，極端不均衡之藥物在 Test Set 可能出現「正樣本數 = 0」
  的退化情形，使 ROC AUC 無法計算。

實作優化：相對於原 notebook 直接儲存 15 份 (X_train, X_test, y_train, y_test)
副本 (對於 1877 × 12 之 DataFrame，× 15 將耗約 6×記憶體)，此處只儲存
**索引** (Index)，由呼叫端依需要 .loc[] 切片，記憶體足跡 O(N) 而非 O(N·|D|)。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import TEST_SIZE, GLOBAL_SEED


@dataclass(frozen=True)
class DrugSplit:
    """
    單一藥物 d 的訓練/測試索引切分結果。
    使用 frozen dataclass 防止意外突變，符合「split 一經產生不可變」之契約。
    """
    drug: str
    train_idx: pd.Index
    test_idx: pd.Index

    def __post_init__(self):
        # Sanity check：兩集合互斥
        if len(self.train_idx.intersection(self.test_idx)) != 0:
            raise ValueError(
                f"Train/Test indices overlap for drug '{self.drug}'."
            )

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


def build_per_drug_splits(
    X: pd.DataFrame,
    y_binary: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = GLOBAL_SEED,
) -> Dict[str, DrugSplit]:
    """
    對 y_binary 之每個欄位 (藥物) 進行分層切分，回傳 dict[drug -> DrugSplit]。

    參數
    ----
    X         : 特徵矩陣 (尚未編碼或已 deterministic-encoded 皆可，因切分僅依 y)
    y_binary  : N × |D| 的 0/1 標籤矩陣 (來自 data_io.binarize_targets)
    test_size : 預設 0.20
    random_state : 統一種子

    回傳
    ----
    dict 鍵為藥物名稱，值為 DrugSplit。
    """
    splits: Dict[str, DrugSplit] = {}
    idx_full = X.index.to_numpy()
    for drug in y_binary.columns:
        y_d = y_binary[drug]
        # 對「索引陣列」做分層切分，避免複製整個 X
        tr_idx_arr, te_idx_arr = train_test_split(
            idx_full,
            test_size=test_size,
            stratify=y_d.loc[idx_full].values,
            random_state=random_state,
        )
        splits[drug] = DrugSplit(
            drug=drug,
            train_idx=pd.Index(tr_idx_arr),
            test_idx=pd.Index(te_idx_arr),
        )
    return splits


def verify_stratification(
    splits: Dict[str, DrugSplit],
    y_binary: pd.DataFrame,
    *,
    tol: float = 0.01,
) -> pd.DataFrame:
    """
    驗證各藥物 Train/Test 之正樣本比例對齊。
    回傳一張稽核表：藥物 × (n_train, n_test, p_train, p_test, |diff|)。
    """
    rows = []
    for drug, sp in splits.items():
        y_tr = y_binary.loc[sp.train_idx, drug]
        y_te = y_binary.loc[sp.test_idx, drug]
        rows.append({
            "drug": drug,
            "n_train": sp.n_train,
            "n_test": sp.n_test,
            "p_train": float(y_tr.mean()),
            "p_test": float(y_te.mean()),
            "abs_diff": float(abs(y_tr.mean() - y_te.mean())),
            "passes_tol": bool(abs(y_tr.mean() - y_te.mean()) <= tol),
        })
    return pd.DataFrame(rows).set_index("drug")


__all__ = ["DrugSplit", "build_per_drug_splits", "verify_stratification"]
