"""Model construction, metrics and CP1 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import FeatureEngineer, add_engineered_features, build_preprocessor

MetricDict = dict[str, float]


def make_pipeline(
    estimator: Any,
    x_train: pd.DataFrame,
    *,
    use_feature_engineering: bool,
) -> Pipeline:
    """Build a full sklearn Pipeline fitted without data leakage."""
    steps: list[tuple[str, Any]] = []
    df_for_columns = x_train.copy()

    if use_feature_engineering:
        steps.append(("feature_engineering", FeatureEngineer()))
        df_for_columns = add_engineered_features(df_for_columns)

    steps.extend(
        [
            ("preprocessor", build_preprocessor(df_for_columns)),
            ("classifier", estimator),
        ]
    )
    return Pipeline(steps)


def get_candidate_models() -> dict[str, tuple[Any, bool]]:
    """Return CP1 baseline and first model candidates.

    bool flag means whether to use feature engineering.
    """
    return {
        "baseline_logreg_no_fe": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "logreg_fe": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "knn_fe": (
            KNeighborsClassifier(n_neighbors=25, weights="distance"),
            True,
        ),
        "random_forest_fe": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            True,
        ),
        "extra_trees_fe": (
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            True,
        ),
        "gradient_boosting_fe": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            True,
        ),
        "hist_gradient_boosting_fe": (
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    }


def get_param_distributions(model_name: str) -> dict[str, list[Any]]:
    """Small hyperparameter grids for optional CP1 tuning."""
    grids: dict[str, dict[str, list[Any]]] = {
        "logreg_fe": {
            "classifier__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "random_forest_fe": {
            "classifier__n_estimators": [250, 400, 600],
            "classifier__max_depth": [None, 6, 10, 16],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", 0.7],
        },
        "extra_trees_fe": {
            "classifier__n_estimators": [250, 400, 600],
            "classifier__max_depth": [None, 6, 10, 16],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", 0.7],
        },
        "hist_gradient_boosting_fe": {
            "classifier__max_iter": [150, 250, 350],
            "classifier__learning_rate": [0.03, 0.06, 0.1],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__l2_regularization": [0.0, 0.1, 1.0],
        },
    }
    return grids.get(model_name, {})


def predict_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Get positive-class scores for ROC-AUC/PR-AUC."""
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(classifier, "decision_function"):
        raw_scores = model.decision_function(x)
        return 1 / (1 + np.exp(-raw_scores))
    return model.predict(x)


def calculate_metrics(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> MetricDict:
    """Calculate classification metrics."""
    y_pred = model.predict(x)
    y_score = predict_scores(model, x)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(y, y_score)
        metrics["pr_auc"] = average_precision_score(y, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return {key: float(value) for key, value in metrics.items()}


def fit_single_model(
    model_name: str,
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    use_feature_engineering: bool,
    tune: bool,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one model and return validation metrics row."""
    model = make_pipeline(estimator, x_train, use_feature_engineering=use_feature_engineering)

    best_params: dict[str, Any] = {}
    if tune and get_param_distributions(model_name):
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=get_param_distributions(model_name),
            n_iter=8,
            scoring="f1",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        fitted_model = search.best_estimator_
        best_params = search.best_params_
    else:
        fitted_model = model.fit(x_train, y_train)

    metrics = calculate_metrics(fitted_model, x_valid, y_valid)
    row: dict[str, Any] = {
        "model": model_name,
        "feature_engineering": use_feature_engineering,
        "tuned": tune and bool(best_params),
        "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
        **metrics,
    }
    return fitted_model, row


def choose_best_model(results: list[dict[str, Any]]) -> str:
    """Choose best model by validation F1, then by PR-AUC."""
    df = pd.DataFrame(results)
    ordered = df.sort_values(["f1", "pr_auc"], ascending=False)
    return str(ordered.iloc[0]["model"])


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist fitted sklearn pipeline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: MetricDict, path: str | Path) -> None:
    """Save JSON metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
