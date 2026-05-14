"""Evaluation helpers for binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:  # noqa: ANN001
    """Return positive-class probability or a calibrated-like score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def choose_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
) -> float:
    """Choose a classification threshold on validation data."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_value = -np.inf
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        if metric == "recall_at_precision_50":
            precision = precision_score(y_true, y_pred, zero_division=0)
            value = recall_score(y_true, y_pred, zero_division=0) if precision >= 0.5 else -1
        else:
            value = f1_score(y_true, y_pred, zero_division=0)
        if value > best_value:
            best_threshold = threshold
            best_value = value
    return float(best_threshold)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute all metrics used in the experiment table."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict[str, float | int]:  # noqa: ANN001
    """Evaluate a fitted model on a given split."""
    scores = predict_positive_probability(model, X)
    return evaluate_predictions(y, scores, threshold=threshold)
