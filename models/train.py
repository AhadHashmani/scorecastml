"""
models/train.py
===============
Trains and compares two models for NBA game outcome prediction:

    1. Logistic Regression   — interpretable baseline
    2. Gradient Boosted Trees (sklearn GradientBoostingClassifier)
                             — higher capacity, captures non-linearities

Both are trained with time-based cross-validation (no data leakage:
training always uses games that happened before the test games).

Outputs:
    - Trained model objects
    - Feature importance rankings
    - Serialised models to disk (models/saved/)

Author: Ahad Hashmani
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent / "saved"
SAVE_DIR.mkdir(exist_ok=True)


@dataclass
class ModelResult:
    name: str
    model: object
    feature_names: List[str]
    accuracy: float
    roc_auc: float
    log_loss_score: float
    brier_score: float
    n_train: int
    n_test: int
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  Model       : {self.name}",
            f"  Train / Test: {self.n_train:,} / {self.n_test:,} games",
            f"  Accuracy    : {self.accuracy:.4f}",
            f"  ROC-AUC     : {self.roc_auc:.4f}",
            f"  Log-Loss    : {self.log_loss_score:.4f}",
            f"  Brier Score : {self.brier_score:.4f}",
        ]
        if self.feature_importance:
            top5 = sorted(
                self.feature_importance.items(), key=lambda x: -x[1]
            )[:5]
            lines.append("  Top features:")
            for feat, imp in top5:
                lines.append(f"    {feat:<35} {imp:.4f}")
        return "\n".join(lines)


def time_split(
    df: pd.DataFrame,
    test_seasons: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split.
    Test set = the most recent `test_seasons` seasons.
    Train set = everything before.
    """
    seasons = sorted(df["SEASON"].unique())
    test_season_ids = seasons[-test_seasons:]
    train_df = df[~df["SEASON"].isin(test_season_ids)].copy()
    test_df  = df[ df["SEASON"].isin(test_season_ids)].copy()
    log.info(
        "Train: %d games (%s)  |  Test: %d games (%s)",
        len(train_df), ", ".join(seasons[:-test_seasons]),
        len(test_df),  ", ".join(test_season_ids),
    )
    return train_df, test_df


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Logistic regression with L2 regularisation and calibration.
    Scaling is essential for LR to converge well.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    coefs = pipe.named_steps["clf"].coef_[0]
    importance = {f: abs(float(c)) for f, c in zip(feature_names, coefs)}
    return pipe, importance


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
) -> Tuple[GradientBoostingClassifier, Dict[str, float]]:
    """
    Gradient boosted trees.
    Hyperparameters chosen for a good bias-variance tradeoff on
    ~5000-game training sets.
    """
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-4,
    )
    clf.fit(X_train, y_train)

    importance = {
        f: float(imp)
        for f, imp in zip(feature_names, clf.feature_importances_)
    }
    return clf, importance


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    feature_names: List[str],
    n_train: int,
    feature_importance: Dict[str, float],
) -> ModelResult:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return ModelResult(
        name=name,
        model=model,
        feature_names=feature_names,
        accuracy=round(accuracy_score(y_test, y_pred), 4),
        roc_auc=round(roc_auc_score(y_test, y_proba), 4),
        log_loss_score=round(log_loss(y_test, y_proba), 4),
        brier_score=round(brier_score_loss(y_test, y_proba), 4),
        n_train=n_train,
        n_test=len(y_test),
        feature_importance=feature_importance,
    )


def train_and_evaluate(
    matchup_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[ModelResult, ModelResult]:
    """
    Full training pipeline. Returns (lr_result, gbt_result).
    """
    from features.engineer import get_feature_columns

    if feature_cols is None:
        feature_cols = get_feature_columns(matchup_df)

    train_df, test_df = time_split(matchup_df, test_seasons=1)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["HOME_WIN"].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["HOME_WIN"].values

    log.info("Training Logistic Regression on %d games...", len(train_df))
    lr_model, lr_imp = train_logistic_regression(X_train, y_train, feature_cols)
    lr_result = evaluate_model(
        lr_model, X_test, y_test,
        "Logistic Regression", feature_cols, len(train_df), lr_imp
    )

    log.info("Training Gradient Boosting on %d games...", len(train_df))
    gbt_model, gbt_imp = train_gradient_boosting(X_train, y_train, feature_cols)
    gbt_result = evaluate_model(
        gbt_model, X_test, y_test,
        "Gradient Boosting", feature_cols, len(train_df), gbt_imp
    )

    # Save models
    for result, fname in [(lr_result, "lr_model.pkl"), (gbt_result, "gbt_model.pkl")]:
        path = SAVE_DIR / fname
        with open(path, "wb") as f:
            pickle.dump({"model": result.model, "feature_cols": feature_cols}, f)
        log.info("Saved %s to %s", result.name, path)

    return lr_result, gbt_result


def load_model(model_name: str = "gbt") -> Tuple[object, List[str]]:
    """Load a saved model from disk."""
    fname = f"{model_name}_model.pkl"
    path = SAVE_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run training first.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"]
