"""
scripts/run_rq1.py
==================

RQ1 主要執行腳本。序貫流程：

  1. 載入 → 清洗 → 決定性編碼 (Semer 過濾 + Ethnicity 二值化 + Country 三類分組)
  2. 目標變數操作化 (二值化 + 連續化)
  3. 逐藥物分層切分
  4. 三模型擬合：Ridge baseline / Lasso main / KNN comparison
  5. 統一評估表 (含 DeLong 與 FDR 校正)
  6. 圖表輸出 (CSV + PNG)

使用方式 (從專案根目錄)：
  python -m scripts.run_rq1
"""

from __future__ import annotations
import sys
from pathlib import Path
import time
import pandas as pd

# 確保可從專案根目錄匯入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    TARGET_DRUGS, EXCLUDED_DRUGS, ensure_dirs,
    TABLES_DIR, FIGURES_DIR, ARTIFACTS_DIR,
)
from data_io import (
    load_raw, remove_semer_overclaimers, apply_deterministic_encoding,
    binarize_targets, ordinalize_targets,
)
from splits import build_per_drug_splits, verify_stratification
from models import fit_ridge_all, fit_lasso_all, fit_knn_all
from reports import (
    build_unified_evaluation_table,
    build_ridge_coefficient_table,
    build_lasso_inference_table,
    plot_lasso_vs_knn_auc,
    plot_coefficient_heatmap,
    plot_auc_ranking,
)


def main(n_bootstrap: int = 1000) -> None:
    ensure_dirs()
    t0 = time.time()

    # ─── 1. 載入與清洗 ─────────────────────────────────────────────
    print("[1/6] Loading & cleaning ...")
    X_raw, y_raw = load_raw()
    X_clean, y_clean = remove_semer_overclaimers(X_raw, y_raw)
    print(f"      raw N = {len(X_raw)}, cleaned N = {len(X_clean)}")

    # 套用決定性編碼 (產生 country_grouped, ethnicity_white)
    X = apply_deterministic_encoding(X_clean)

    # ─── 2. 目標變數 ────────────────────────────────────────────────
    print("[2/6] Operationalizing targets ...")
    target_drugs = [d for d in TARGET_DRUGS if d in y_clean.columns]
    if len(target_drugs) != 15:
        missing = set(TARGET_DRUGS) - set(target_drugs)
        raise RuntimeError(f"Target drugs missing in dataset: {missing}")
    y_binary = binarize_targets(y_clean, target_drugs)
    z_continuous = ordinalize_targets(y_clean, target_drugs)

    # ─── 3. 切分 ────────────────────────────────────────────────────
    print("[3/6] Per-drug stratified split ...")
    splits = build_per_drug_splits(X, y_binary)
    audit = verify_stratification(splits, y_binary)
    audit.to_csv(TABLES_DIR / "00_split_audit.csv")
    print("      split audit saved.")

    # ─── 4. 三模型擬合 ──────────────────────────────────────────────
    print("[4/6] Fitting Ridge baseline (15 drugs) ...")
    ridge_results = fit_ridge_all(X, z_continuous, splits)

    print("[4/6] Fitting Lasso main + bootstrap CI (B={}) ...".format(n_bootstrap))
    lasso_results = fit_lasso_all(X, y_binary, splits, n_bootstrap=n_bootstrap)

    print("[4/6] Fitting KNN comparison (15 drugs) ...")
    knn_results = fit_knn_all(X, y_binary, splits)

    # ─── 5. 統一評估表 ──────────────────────────────────────────────
    print("[5/6] Building unified evaluation table ...")
    eval_df = build_unified_evaluation_table(
        splits=splits,
        y_binary=y_binary,
        z_continuous=z_continuous,
        X=X,
        ridge_results=ridge_results,
        lasso_results=lasso_results,
        knn_results=knn_results,
    )
    eval_df.to_csv(TABLES_DIR / "01_unified_evaluation.csv")

    ridge_coef_df = build_ridge_coefficient_table(ridge_results)
    ridge_coef_df.to_csv(TABLES_DIR / "02_ridge_coefficients.csv")

    lasso_inf_df = build_lasso_inference_table(lasso_results)
    lasso_inf_df.to_csv(TABLES_DIR / "03_lasso_inference_long.csv", index=False)

    # 也儲存 Lasso 係數的寬格式 (7 × 15) 供繪圖
    lasso_coef_wide = lasso_inf_df.pivot(
        index="feature", columns="drug", values="beta",
    ).loc[ridge_coef_df.index, ridge_coef_df.columns]
    lasso_coef_wide.to_csv(TABLES_DIR / "04_lasso_coefficients_wide.csv")

    # ─── 6. 圖表輸出 ────────────────────────────────────────────────
    print("[6/6] Producing figures ...")
    plot_lasso_vs_knn_auc(
        eval_df, save_path=FIGURES_DIR / "fig1_lasso_vs_knn_auc.png",
    )
    plot_coefficient_heatmap(
        ridge_coef_df,
        title="Ridge Standardized Coefficients (linear on z ∈ {0,...,6})",
        save_path=FIGURES_DIR / "fig2_ridge_coef_heatmap.png",
    )
    plot_coefficient_heatmap(
        lasso_coef_wide,
        title="Lasso Logistic Coefficients (log-odds per 1σ change)",
        save_path=FIGURES_DIR / "fig3_lasso_coef_heatmap.png",
    )
    plot_auc_ranking(
        eval_df, save_path=FIGURES_DIR / "fig4_lasso_auc_ranking.png",
    )

    elapsed = time.time() - t0
    print(f"\n[DONE] All artifacts under {ARTIFACTS_DIR}/")
    print(f"       elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples for Lasso CI (default 1000; "
             "use small values like 50 for quick smoke tests).",
    )
    args = ap.parse_args()
    main(n_bootstrap=args.bootstrap)
