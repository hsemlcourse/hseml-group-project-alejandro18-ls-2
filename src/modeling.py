"""Model definitions, hyperparameter search, and interpretability helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_SELECTION_METRIC, RANDOM_STATE
from src.metrics import choose_threshold, evaluate_model, predict_positive_probability
from src.preprocessing import build_preprocessor

try:  # pragma: no cover - optional dependency may be unavailable locally
    from xgboost import XGBClassifier
except Exception:  # noqa: BLE001
    XGBClassifier = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency may be unavailable locally
    from lightgbm import LGBMClassifier
except Exception:  # noqa: BLE001
    LGBMClassifier = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelSpec:
    """A trainable model specification for CP2 experiments."""

    name: str
    estimator: Pipeline
    param_distributions: dict[str, list[Any]]
    search: bool = True
    note: str = ""


def make_pipeline(
    model: object,
    *,
    use_feature_engineering: bool = True,
    scale_numeric: bool = True,
    clip_outliers: bool = True,
) -> Pipeline:
    """Combine preprocessing and estimator into one leakage-safe pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    use_feature_engineering=use_feature_engineering,
                    scale_numeric=scale_numeric,
                    clip_outliers=clip_outliers,
                ),
            ),
            ("model", model),
        ]
    )


def build_model_specs(random_state: int = RANDOM_STATE) -> list[ModelSpec]:
    """Return baseline, classic models, boosting models, and ensembles."""
    specs: list[ModelSpec] = [
        ModelSpec(
            name="baseline_logreg_raw_no_fe",
            estimator=make_pipeline(
                LogisticRegression(max_iter=2000, random_state=random_state),
                use_feature_engineering=False,
                scale_numeric=True,
                clip_outliers=False,
            ),
            param_distributions={},
            search=False,
            note="Baseline: Logistic Regression без feature engineering и clipping.",
        ),
        ModelSpec(
            name="logreg_balanced_fe",
            estimator=make_pipeline(
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
                scale_numeric=True,
            ),
            param_distributions={
                "model__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"],
            },
            note="Линейная модель с class_weight для дисбаланса классов.",
        ),
        ModelSpec(
            name="knn_fe",
            estimator=make_pipeline(KNeighborsClassifier(), scale_numeric=True),
            param_distributions={
                "model__n_neighbors": [5, 9, 15, 25, 35],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            note="Нелинейная distance-based модель как альтернативный класс алгоритмов.",
        ),
        ModelSpec(
            name="decision_tree_fe",
            estimator=make_pipeline(
                DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
                scale_numeric=False,
            ),
            param_distributions={
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_leaf": [5, 10, 20, 50],
                "model__criterion": ["gini", "entropy", "log_loss"],
            },
            note="Интерпретируемое дерево решений.",
        ),
        ModelSpec(
            name="random_forest_fe",
            estimator=make_pipeline(
                RandomForestClassifier(
                    n_estimators=180,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=random_state,
                ),
                scale_numeric=False,
            ),
            param_distributions={
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_leaf": [2, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.7],
            },
            note="Bagging-ансамбль, устойчивый к выбросам и нелинейностям.",
        ),
        ModelSpec(
            name="extra_trees_fe",
            estimator=make_pipeline(
                ExtraTreesClassifier(
                    n_estimators=220,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=random_state,
                ),
                scale_numeric=False,
            ),
            param_distributions={
                "model__n_estimators": [250, 400, 600],
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_leaf": [2, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.8],
            },
            note="Случайный лес с более сильной рандомизацией.",
        ),
        ModelSpec(
            name="gradient_boosting_fe",
            estimator=make_pipeline(
                GradientBoostingClassifier(
                    learning_rate=0.07,
                    random_state=random_state,
                ),
                scale_numeric=False,
            ),
            param_distributions={
                "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
                "model__n_estimators": [80, 150, 220],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.75, 0.9, 1.0],
            },
            note="Boosting из sklearn для табличных данных.",
        ),
    ]

    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                name="xgboost_fe",
                estimator=make_pipeline(
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_estimators=180,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                    scale_numeric=False,
                ),
                param_distributions={
                    "model__n_estimators": [200, 300, 500],
                    "model__max_depth": [3, 4, 5, 7],
                    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                    "model__scale_pos_weight": [1.0, 3.0, 5.5, 7.0],
                },
                note="Gradient boosting с учетом class imbalance через scale_pos_weight.",
            )
        )

    if LGBMClassifier is not None:
        specs.append(
            ModelSpec(
                name="lightgbm_fe",
                estimator=make_pipeline(
                    LGBMClassifier(
                        objective="binary",
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                    scale_numeric=False,
                ),
                param_distributions={
                    "model__n_estimators": [200, 400, 600],
                    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                    "model__num_leaves": [15, 31, 63],
                    "model__min_child_samples": [10, 20, 40, 80],
                    "model__subsample": [0.7, 0.85, 1.0],
                    "model__colsample_bytree": [0.7, 0.85, 1.0],
                },
                note="LightGBM: быстрый boosting для табличных данных.",
            )
        )

    voting_model = VotingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3, random_state=random_state),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
            (
                "hgb",
                GradientBoostingClassifier(
                    learning_rate=0.05,
                    n_estimators=150,
                    max_depth=3,
                    random_state=random_state,
                ),
            ),
        ],
        voting="soft",
        weights=[1, 2, 2],
        n_jobs=1,
    )
    specs.append(
        ModelSpec(
            name="soft_voting_ensemble",
            estimator=make_pipeline(voting_model, scale_numeric=True),
            param_distributions={
                "model__weights": [[1, 1, 1], [1, 2, 2], [2, 2, 3], [1, 3, 3]],
            },
            note="Явное ансамблирование: soft voting по линейной, bagging и boosting моделям.",
        )
    )

    stacking_model = StackingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3, random_state=random_state),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
            (
                "hgb",
                GradientBoostingClassifier(learning_rate=0.05, n_estimators=150, max_depth=3, random_state=random_state),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
        cv=3,
    )
    specs.append(
        ModelSpec(
            name="stacking_ensemble",
            estimator=make_pipeline(stacking_model, scale_numeric=True),
            param_distributions={
                "model__final_estimator__C": [0.1, 0.3, 1.0, 3.0],
            },
            note="Явное ансамблирование: stacking с meta-моделью Logistic Regression.",
        )
    )

    specs.append(
        ModelSpec(
            name="logreg_svd_dim_reduction",
            estimator=Pipeline(
                steps=[
                    (
                        "preprocess",
                        build_preprocessor(
                            use_feature_engineering=True,
                            scale_numeric=True,
                            clip_outliers=True,
                        ),
                    ),
                    ("svd", TruncatedSVD(n_components=12, random_state=random_state)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_distributions={
                "svd__n_components": [6, 8, 10, 12, 16],
                "model__C": [0.1, 0.3, 1.0, 3.0],
            },
            note="Эксперимент с уменьшением размерности после one-hot encoding.",
        )
    )
    return specs


def fit_model_spec(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 12,
    cv_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one model specification with optional randomized hyperparameter search."""
    start = perf_counter()
    if spec.search and spec.param_distributions:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=spec.estimator,
            param_distributions=spec.param_distributions,
            n_iter=min(n_iter, _grid_size(spec.param_distributions)),
            scoring=MODEL_SELECTION_METRIC,
            cv=cv,
            random_state=random_state,
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        search_info = {
            "cv_best_score": float(search.best_score_),
            "best_params": search.best_params_,
            "searched": True,
        }
    else:
        best_estimator = clone(spec.estimator)
        best_estimator.fit(X_train, y_train)
        search_info = {"cv_best_score": None, "best_params": {}, "searched": False}
    search_info["fit_seconds"] = round(perf_counter() - start, 3)
    return best_estimator, search_info


def run_experiments(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    *,
    n_iter: int = 12,
    cv_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Train all model specs and evaluate them on the validation split."""
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for spec in build_model_specs(random_state=random_state):
        model, search_info = fit_model_spec(
            spec,
            X_train,
            y_train,
            n_iter=n_iter,
            cv_splits=cv_splits,
            random_state=random_state,
        )
        val_scores = predict_positive_probability(model, X_val)
        threshold = choose_threshold(y_val, val_scores, metric="f1")
        val_metrics = evaluate_model(model, X_val, y_val, threshold=threshold)
        row = {
            "model": spec.name,
            "note": spec.note,
            **search_info,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        rows.append(row)
        fitted_models[spec.name] = model

    results = pd.DataFrame(rows).sort_values(
        by=f"val_{MODEL_SELECTION_METRIC}", ascending=False
    )
    return results, fitted_models


def select_best_model(results: pd.DataFrame) -> str:
    """Select the model with the best validation average precision."""
    return str(results.iloc[0]["model"])


def refit_final_model(best_model: Pipeline, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> Pipeline:
    """Refit the already selected model family and hyperparameters on train+validation."""
    final_model = clone(best_model)
    final_model.fit(X_train_val, y_train_val)
    return final_model


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Persist a sklearn pipeline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Save JSON with readable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def get_transformed_feature_names(model: Pipeline) -> list[str]:
    """Return names after preprocessing for models without dimensionality reduction."""
    preprocess = model.named_steps.get("preprocess")
    if preprocess is None:
        return []
    columns = preprocess.named_steps.get("columns")
    if columns is None:
        return []
    return list(columns.get_feature_names_out())


def model_feature_importance(model: Pipeline, top_n: int = 25) -> pd.DataFrame:
    """Extract coefficient/importances for models that expose them."""
    feature_names = get_transformed_feature_names(model)
    estimator = model.named_steps.get("model")
    if estimator is None or not feature_names:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(values) != len(feature_names):
        # Dimensionality-reduction pipelines expose coefficients in latent space,
        # not in the original transformed feature space. Use permutation importance instead.
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def permutation_importance_table(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    top_n: int = 20,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute model-agnostic permutation importance on raw validation/test rows."""
    sample_size = min(len(X), 1500)
    X_sample = X.sample(sample_size, random_state=random_state)
    y_sample = y.loc[X_sample.index]
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring=MODEL_SELECTION_METRIC,
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _grid_size(param_distributions: dict[str, list[Any]]) -> int:
    size = 1
    for values in param_distributions.values():
        size *= len(values)
    return size
