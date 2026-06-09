"""
reports/figures.py
==================

繪圖工具。本研究 §2.6 規劃的三項主要視覺化輸出：

  Figure 1: Lasso vs KNN AUC 散佈圖 (Structural Misspecification Diagnostic)
            ── 落在 y=x 下方的藥物表示「線性附加模型已足夠」；
               落在 y=x 上方且 DeLong FDR 顯著者，提示存在非線性結構。

  Figure 2: Lasso 標準化係數熱度圖 (Standardized Effect Heatmap)
            ── 顯示 7 個人格特質對 15 種藥物之 log-odds 影響強度與方向。

  Figure 3: Ridge 標準化係數熱度圖 (作為對照，未稀疏化版本)。

為什麼將圖與表分離？
  「事實表」屬資料表單；「圖」屬視覺化敘事。在多檔結構下，使用者可以
  選擇只重新生成圖 (例如為了論文修改顏色) 而不重跑模型，符合 separation
  of concerns 原則。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import PSYCH_FEATURES


def plot_lasso_vs_knn_auc(
    eval_table: pd.DataFrame,
    *,
    save_path: Path | None = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    AUC 散佈圖：x = Lasso AUC, y = KNN AUC，附 y=x 對角線。
    DeLong FDR 顯著者以紅色標示。
    """
    fig, ax = plt.subplots(figsize=figsize)

    sig = eval_table["Lasso_vs_KNN_signif"].astype(bool).to_numpy()
    colors = np.where(sig, "tab:red", "tab:blue")

    ax.scatter(
        eval_table["Lasso_AUC"], eval_table["KNN_AUC"],
        c=colors, s=70, alpha=0.75, edgecolor="black", linewidth=0.7,
    )

    lo = min(eval_table["Lasso_AUC"].min(), eval_table["KNN_AUC"].min()) - 0.03
    hi = max(eval_table["Lasso_AUC"].max(), eval_table["KNN_AUC"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")

    for drug, row in eval_table.iterrows():
        ax.annotate(
            drug,
            xy=(row["Lasso_AUC"], row["KNN_AUC"]),
            xytext=(4, 3), textcoords="offset points", fontsize=8,
        )

    ax.set_xlabel("Lasso Logistic Regression  AUC", fontsize=11)
    ax.set_ylabel("k-Nearest Neighbors  AUC", fontsize=11)
    ax.set_title("Structural Misspecification Diagnostic\n"
                 "(Red = DeLong significant at FDR < 0.05)", fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_coefficient_heatmap(
    coef_matrix: pd.DataFrame,
    *,
    title: str = "Standardized Coefficients",
    save_path: Path | None = None,
    figsize: tuple = (9, 5),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    coef_matrix : (7 traits) × (|D| drugs) 之 DataFrame；數值為標準化係數。
    使用 diverging colormap，0 為白色。
    """
    fig, ax = plt.subplots(figsize=figsize)

    vmax = float(np.nanmax(np.abs(coef_matrix.to_numpy())))
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(
        coef_matrix.to_numpy(),
        cmap=cmap,
        vmin=-vmax, vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(coef_matrix.shape[1]))
    ax.set_xticklabels(coef_matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(coef_matrix.shape[0]))
    ax.set_yticklabels(coef_matrix.index, fontsize=10)

    # 在每格中央寫上數值
    for i in range(coef_matrix.shape[0]):
        for j in range(coef_matrix.shape[1]):
            val = coef_matrix.iat[i, j]
            if np.isnan(val):
                txt = ""
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7,
                    color="black" if abs(val) < vmax * 0.55 else "white")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_auc_ranking(
    eval_table: pd.DataFrame,
    *,
    save_path: Path | None = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """
    水平 bar chart：依 Lasso AUC 由高至低排序，附 95% CI error bar。
    回應規劃書 §2.6「預測性排名表」之視覺化版本。
    """
    df = eval_table.sort_values("Lasso_AUC", ascending=True)
    aucs = df["Lasso_AUC"].to_numpy()
    err_lo = aucs - df["Lasso_AUC_CI_lo"].to_numpy()
    err_hi = df["Lasso_AUC_CI_hi"].to_numpy() - aucs

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df.index, aucs, xerr=[err_lo, err_hi],
            color="tab:blue", alpha=0.75, edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="gray", lw=1.0, capsize=3))
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.0, label="AUC = 0.5")
    ax.set_xlabel("Lasso AUC  (95% Bootstrap CI)", fontsize=11)
    ax.set_title("Predictability of Drug-Use by Personality (Ranked)", fontsize=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


__all__ = [
    "plot_lasso_vs_knn_auc",
    "plot_coefficient_heatmap",
    "plot_auc_ranking",
]
