"""Preprocessing, cleaning, feature engineering, and leakage-safe splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_CATEGORICAL_COLUMNS_WITH_FEATURES,
    ALL_NUMERIC_COLUMNS_WITH_FEATURES,
    CATEGORICAL_COLUMNS,
    MONTH_TO_INDEX,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create behavior-based features from raw session columns."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        eps = 1e-9

        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        total_pages = df["Administrative"] + df["Informational"] + df["ProductRelated"]
        total_duration = (
            df["Administrative_Duration"]
            + df["Informational_Duration"]
            + df["ProductRelated_Duration"]
        )

        df["total_pages"] = total_pages
        df["total_duration"] = total_duration
        df["product_page_share"] = df["ProductRelated"] / (total_pages + eps)
        df["info_page_share"] = df["Informational"] / (total_pages + eps)
        df["admin_page_share"] = df["Administrative"] / (total_pages + eps)
        df["avg_time_per_page"] = total_duration / (total_pages + eps)
        df["bounce_exit_ratio"] = df["BounceRates"] / (df["ExitRates"] + eps)
        df["page_value_log1p"] = np.log1p(df["PageValues"].clip(lower=0))
        df["engagement_score"] = np.log1p(total_pages) * np.log1p(total_duration)
        df["month_index"] = df["Month"].map(MONTH_TO_INDEX).fillna(0).astype(float)
        df["is_returning_visitor"] = (
            df["VisitorType"].astype(str).str.lower().eq("returning_visitor")
        ).astype(str)
        df["is_new_visitor"] = (
            df["VisitorType"].astype(str).str.lower().eq("new_visitor")
        ).astype(str)
        df["is_special_day"] = (df["SpecialDay"] > 0).astype(str)
        df["season"] = df["month_index"].map(_month_to_season).astype(str)
        return df


def _month_to_season(month_index: float) -> str:
    month = int(month_index) if not pd.isna(month_index) else 0
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    if month in {9, 10, 11}:
        return "autumn"
    return "unknown"


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers using quantiles fitted only on the training fold."""

    def __init__(self, columns: list[str] | None = None, lower: float = 0.01, upper: float = 0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "QuantileClipper":
        columns = self.columns or list(X.select_dtypes(include=np.number).columns)
        self.columns_ = [column for column in columns if column in X.columns]
        numeric = X[self.columns_].apply(pd.to_numeric, errors="coerce")
        self.lower_bounds_ = numeric.quantile(self.lower)
        self.upper_bounds_ = numeric.quantile(self.upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for column in self.columns_:
            df[column] = pd.to_numeric(df[column], errors="coerce").clip(
                lower=self.lower_bounds_[column],
                upper=self.upper_bounds_[column],
            )
        return df


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train/validation/test split.

    All preprocessing transformers are fitted after this split, preventing statistics from the
    validation/test data from leaking into training.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(
    *,
    use_feature_engineering: bool = True,
    scale_numeric: bool = True,
    clip_outliers: bool = True,
) -> Pipeline:
    """Build a preprocessing pipeline for sklearn-compatible estimators."""
    numeric_columns = ALL_NUMERIC_COLUMNS_WITH_FEATURES if use_feature_engineering else NUMERIC_COLUMNS
    categorical_columns = (
        ALL_CATEGORICAL_COLUMNS_WITH_FEATURES if use_feature_engineering else CATEGORICAL_COLUMNS
    )

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_columns),
            ("cat", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    steps: list[tuple[str, object]] = []
    if use_feature_engineering:
        steps.append(("features", FeatureEngineer()))
    if clip_outliers:
        steps.append(("clip_outliers", QuantileClipper(columns=numeric_columns)))
    steps.append(("columns", column_transformer))
    return Pipeline(steps)
