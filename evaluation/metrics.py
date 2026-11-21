"""
evaluation/metrics.py
=====================
Comprehensive model evaluation:
    - Accuracy, ROC-AUC, Log-Loss, Brier Score
    - Calibration curve (reliability diagram)
    - Confusion matrix
    - Feature importance plot
    - Head-to-head model comparison table

Author: Ahad Hashmani
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from models.train import ModelResult

log = logging.getLogger(__name__)


def print_comparison_table(results: List[ModelResult]) -> None:
    """Print a formatted side-by-side model comparison."""
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON")
    print("=" * 65)
    print(f"  {'Metric':<20}", end="")
    for r in results:
        print(f"  {r.name:<22}", end="")
    print()
    print("  " + "-" * 61)

    metrics = [
        ("Accuracy",    "accuracy"),
        ("ROC-AUC",     "roc_auc"),
        ("Log-Loss",    "log_loss_score"),
        ("Brier Score", "brier_score"),
        ("Train Games", "n_train"),
        ("Test Games",  "n_test"),
    ]
    for label, attr in metrics:
        print(f"  {label:<20}", end="")
        for r in results:
            val = getattr(r, attr)
            fmt = f"{val:.4f}" if isinstance(val, float) else f"{val:,}"
            print(f"  {fmt:<22}", end="")
        print()

    print("=" * 65)
    print()


def print_feature_importance(result: ModelResult, top_n: int = 15) -> None:
    """Print ranked feature importances."""
    if not result.feature_importance:
        return
    sorted_feats = sorted(result.feature_importance.items(), key=lambda x: -x[1])
    print(f"\n  Top {top_n} Features — {result.name}")
    print("  " + "-" * 50)
    for feat, imp in sorted_feats[:top_n]:
        bar = "█" * int(imp * 200)
        clean = feat.replace("DIFF_", "").replace("_", " ").lower()
        print(f"  {clean:<35} {imp:.4f}  {bar}")
    print()


def plot_evaluation(
    results: List[ModelResult],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "evaluation/scorecast_eval.png",
) -> None:
    """
    Generate a 2x2 evaluation dashboard:
        [0,0] ROC curves
        [0,1] Calibration curves
        [1,0] Confusion matrix (best model)
        [1,1] Feature importance (best model)
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("ScorecastML — Model Evaluation Dashboard", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_roc   = fig.add_subplot(gs[0, 0])
    ax_cal   = fig.add_subplot(gs[0, 1])
    ax_cm    = fig.add_subplot(gs[1, 0])
    ax_feat  = fig.add_subplot(gs[1, 1])

    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # ── ROC curves ──────────────────────────────────────────────────
    for i, result in enumerate(results):
        y_prob = result.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=colors[i],
                    label=f"{result.name} (AUC={result.roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(fontsize=9)
    ax_roc.grid(True, alpha=0.3)

    # ── Calibration curves ──────────────────────────────────────────
    for i, result in enumerate(results):
        y_prob = result.model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, "s-", color=colors[i],
                    label=f"{result.name} (Brier={result.brier_score:.3f})")

    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Curves (Reliability Diagram)")
    ax_cal.legend(fontsize=9)
    ax_cal.grid(True, alpha=0.3)

    # ── Confusion matrix (best model by ROC-AUC) ────────────────────
    best = max(results, key=lambda r: r.roc_auc)
    y_pred = best.model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Away Win", "Home Win"],
        ax=ax_cm,
        colorbar=False,
        cmap="Blues",
    )
    ax_cm.set_title(f"Confusion Matrix\n{best.name}")

    # ── Feature importance ───────────────────────────────────────────
    if best.feature_importance:
        top_feats = sorted(best.feature_importance.items(), key=lambda x: -x[1])[:12]
        feat_labels = [f.replace("DIFF_", "").replace("_", " ").lower() for f, _ in top_feats]
        feat_vals   = [v for _, v in top_feats]
        y_pos = range(len(feat_labels))
        ax_feat.barh(list(y_pos), feat_vals, color="#2196F3", alpha=0.8)
        ax_feat.set_yticks(list(y_pos))
        ax_feat.set_yticklabels(feat_labels, fontsize=8)
        ax_feat.invert_yaxis()
        ax_feat.set_xlabel("Importance")
        ax_feat.set_title(f"Feature Importance\n{best.name}")
        ax_feat.grid(True, axis="x", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Evaluation plot saved to %s", save_path)
    print(f"  Evaluation plot saved → {save_path}")


def run_full_evaluation(
    results: List[ModelResult],
    matchup_df: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    """Run all evaluation outputs."""
    from models.train import time_split

    _, test_df = time_split(matchup_df, test_seasons=1)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["HOME_WIN"].values

    print_comparison_table(results)

    best = max(results, key=lambda r: r.roc_auc)
    print_feature_importance(best)

    for r in results:
        print(r.summary())
        print()

    plot_evaluation(results, X_test, y_test)
