"""Data loading, cleaning, splitting and feature engineering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TARGET_COLUMN

COLUMN_ALIASES = {
    "Administrative Duration": "Administrative_Duration",
    "Informational Duration": "Informational_Duration",
    "Product Related": "ProductRelated",
    "Product Related Duration": "ProductRelated_Duration",
    "Product-Related Duration": "ProductRelated_Duration",
    "Bounce Rate": "BounceRates",
    "Exit Rate": "ExitRates",
    "Page Value": "PageValues",
    "Special Day": "SpecialDay",
    "Operating System": "OperatingSystems",
    "Operating Systems": "OperatingSystems",
    "Traffic Type": "TrafficType",
    "Visitor Type": "VisitorType",
}

BASE_NUMERIC_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

BASE_CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "TotalPages",
    "TotalDuration",
    "ProductTimePerPage",
    "AdminTimePerPage",
    "InfoTimePerPage",
    "EngagementRate",
]

ENGINEERED_CATEGORICAL_COLUMNS = [
    "IsReturningVisitor",
    "HasSpecialDay",
]

COLUMNS_ALLOWED_TO_DROP_FOR_ABLATION = {"PageValues"}


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train/validation/test split."""

    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers using quantiles estimated only on the training fold."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x: np.ndarray, y: pd.Series | None = None) -> QuantileClipper:
        x_array = np.asarray(x, dtype=float)
        self.lower_bounds_ = np.nanquantile(x_array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(x_array, self.upper_quantile, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        return np.clip(x_array, self.lower_bounds_, self.upper_bounds_)

def to_object_dataframe(x):
    """Convert categorical block to object dtype; module-level for joblib pickling."""
    return x.astype("object") if hasattr(x, "astype") else x


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add deterministic row-wise features without using target statistics."""

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(pd.DataFrame(x).copy())


def _to_snake_like(name: str) -> str:
    """Normalize whitespace while preserving official dataset names."""
    return str(name).strip().replace("-", "_")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize alternative Kaggle/UCI column names to one convention."""
    renamed = {}
    for column in df.columns:
        stripped = _to_snake_like(column)
        renamed[column] = COLUMN_ALIASES.get(column, COLUMN_ALIASES.get(stripped, stripped))
    return df.rename(columns=renamed)


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Convert bool-like strings/numbers to booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series

    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
    }
    lowered = series.map(lambda value: str(value).strip().lower() if pd.notna(value) else value)
    parsed = lowered.map(mapping)

    if parsed.isna().any():
        bad_values = sorted(series[parsed.isna()].dropna().astype(str).unique())
        raise ValueError(f"Cannot parse boolean values in column {series.name}: {bad_values}")

    return parsed.astype(bool)


def read_raw_data(path: str | Path) -> pd.DataFrame:
    """Read raw CSV and apply lightweight column normalization."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file was not found: {path}. "
            "Run `python -m src.download_data` or place CSV into data/raw/."
        )

    df = pd.read_csv(path)
    return normalize_column_names(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data without learning any train-dependent statistics."""
    df = normalize_column_names(df.copy())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column `{TARGET_COLUMN}` is missing.")

    for column in [TARGET_COLUMN, "Weekend"]:
        if column in df.columns:
            df[column] = parse_bool_series(df[column])

    for column in BASE_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        if column in df.columns:
            df[column] = df[column].astype("Int64").astype("string")

    for column in ["Month", "VisitorType"]:
        if column in df.columns:
            df[column] = df[column].astype("string")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped duplicated rows: {dropped}")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic features computed inside each row."""
    df = df.copy()

    def get(column: str, default: float = 0.0) -> pd.Series:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(default, index=df.index)

    administrative = get("Administrative")
    informational = get("Informational")
    product_related = get("ProductRelated")

    admin_duration = get("Administrative_Duration")
    info_duration = get("Informational_Duration")
    product_duration = get("ProductRelated_Duration")

    bounce_rates = get("BounceRates")
    exit_rates = get("ExitRates")
    special_day = get("SpecialDay")

    df["TotalPages"] = administrative + informational + product_related
    df["TotalDuration"] = admin_duration + info_duration + product_duration
    df["ProductTimePerPage"] = product_duration / (product_related + 1.0)
    df["AdminTimePerPage"] = admin_duration / (administrative + 1.0)
    df["InfoTimePerPage"] = info_duration / (informational + 1.0)
    df["EngagementRate"] = (1.0 - bounce_rates.clip(0, 1)) * (1.0 - exit_rates.clip(0, 1))

    if "VisitorType" in df.columns:
        df["IsReturningVisitor"] = df["VisitorType"].astype(str).eq("Returning_Visitor")
    else:
        df["IsReturningVisitor"] = False

    df["HasSpecialDay"] = special_day.fillna(0).gt(0)
    return df


def make_xy(df: pd.DataFrame, drop_columns: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and binary target."""
    drop_columns = set(drop_columns or [])
    unknown_drops = drop_columns.difference(COLUMNS_ALLOWED_TO_DROP_FOR_ABLATION)
    if unknown_drops:
        raise ValueError(f"Unexpected columns requested for ablation: {unknown_drops}")

    df = clean_data(df)
    y = df[TARGET_COLUMN].astype(int)
    x = df.drop(columns=[TARGET_COLUMN, *drop_columns], errors="ignore")
    return x, y


def make_train_valid_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Make stratified 70/15/15 split."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    return DatasetSplit(
        x_train=x_train.reset_index(drop=True),
        x_valid=x_valid.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def infer_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer feature lists while keeping discrete numeric IDs categorical."""
    existing_numeric = [
        column
        for column in [*BASE_NUMERIC_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS]
        if column in df.columns
    ]
    existing_categorical = [
        column
        for column in [*BASE_CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL_COLUMNS]
        if column in df.columns
    ]

    remaining = [
        column
        for column in df.columns
        if column not in set(existing_numeric).union(existing_categorical)
    ]
    for column in remaining:
        if pd.api.types.is_numeric_dtype(df[column]):
            existing_numeric.append(column)
        else:
            existing_categorical.append(column)

    return existing_numeric, existing_categorical


def build_preprocessor(df_for_columns: pd.DataFrame) -> ColumnTransformer:
    """Build sklearn preprocessing for numeric and categorical columns."""
    numeric_columns, categorical_columns = infer_feature_types(df_for_columns)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(lower_quantile=0.01, upper_quantile=0.99)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Mixed pandas dtypes (string, boolean and nullable integer-as-string)
            # are converted to object so SimpleImputer handles them consistently.
            ("to_object", FunctionTransformer(to_object_dataframe, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def save_splits(split: DatasetSplit, output_dir: str | Path) -> None:
    """Save split CSV files with target column for transparency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, x_part, y_part in [
        ("train", split.x_train, split.y_train),
        ("valid", split.x_valid, split.y_valid),
        ("test", split.x_test, split.y_test),
    ]:
        part = x_part.copy()
        part[TARGET_COLUMN] = y_part.values
        part.to_csv(output_dir / f"{name}.csv", index=False)
