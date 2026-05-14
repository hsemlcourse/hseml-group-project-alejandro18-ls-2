"""Smoke tests for the CP2 pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET
from src.metrics import evaluate_model
from src.modeling import make_pipeline
from src.preprocessing import FeatureEngineer, QuantileClipper, train_val_test_split


def make_sample_data(n_rows: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "Administrative": rng.integers(0, 8, n_rows),
            "Administrative_Duration": rng.gamma(2, 30, n_rows),
            "Informational": rng.integers(0, 4, n_rows),
            "Informational_Duration": rng.gamma(2, 20, n_rows),
            "ProductRelated": rng.integers(1, 80, n_rows),
            "ProductRelated_Duration": rng.gamma(4, 100, n_rows),
            "BounceRates": rng.uniform(0, 0.2, n_rows),
            "ExitRates": rng.uniform(0, 0.25, n_rows),
            "PageValues": rng.gamma(1.2, 10, n_rows),
            "SpecialDay": rng.choice([0, 0.2, 0.6, 1.0], n_rows),
            "Month": rng.choice(["Feb", "Mar", "May", "Nov", "Dec"], n_rows),
            "OperatingSystems": rng.integers(1, 4, n_rows).astype(str),
            "Browser": rng.integers(1, 5, n_rows).astype(str),
            "Region": rng.integers(1, 5, n_rows).astype(str),
            "TrafficType": rng.integers(1, 8, n_rows).astype(str),
            "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor"], n_rows),
            "Weekend": rng.choice([True, False], n_rows),
        }
    )
    y = ((X["PageValues"] > X["PageValues"].median()) & (X["ExitRates"] < 0.16)).astype(int)
    return X, y.rename(TARGET)


def test_expected_columns_in_sample() -> None:
    X, y = make_sample_data()
    assert set(NUMERIC_COLUMNS).issubset(X.columns)
    assert set(CATEGORICAL_COLUMNS).issubset(X.columns)
    assert y.name == TARGET
    assert set(y.unique()) == {0, 1}


def test_feature_engineering_adds_columns() -> None:
    X, _ = make_sample_data()
    transformed = FeatureEngineer().fit_transform(X)
    assert "total_pages" in transformed.columns
    assert "page_value_log1p" in transformed.columns
    assert "season" in transformed.columns


def test_quantile_clipper_caps_outliers() -> None:
    X, _ = make_sample_data()
    X.loc[0, "PageValues"] = 10_000
    clipper = QuantileClipper(columns=["PageValues"], lower=0.05, upper=0.95).fit(X)
    clipped = clipper.transform(X)
    assert clipped["PageValues"].max() < 10_000


def test_train_val_test_split_is_stratified() -> None:
    X, y = make_sample_data(120)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert y_train.mean() > 0
    assert y_val.mean() > 0
    assert y_test.mean() > 0


def test_model_pipeline_fits_and_evaluates() -> None:
    X, y = make_sample_data(100)
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)
    model = make_pipeline(LogisticRegression(max_iter=1000), scale_numeric=True)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_val, y_val)
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
