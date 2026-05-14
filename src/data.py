"""Data loading and validation utilities."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from src.config import REQUIRED_COLUMNS, TARGET

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
DEFAULT_RAW_PATH = Path("data/raw/online_shoppers_intention.csv")


def download_from_uci(output_path: str | Path = DEFAULT_RAW_PATH) -> Path:
    """Download the raw CSV from UCI and return the output path.

    Kaggle requires authentication for most CLI downloads, so the public UCI mirror is used
    as the reproducible source. The dataset is the same Online Shoppers Purchasing Intention
    dataset and contains the `Revenue` class label.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = output_path.with_suffix(".zip")
    urlretrieve(UCI_ZIP_URL, zip_path)  # noqa: S310 - fixed public academic dataset URL

    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            msg = f"No CSV file found in downloaded archive: {zip_path}"
            raise FileNotFoundError(msg)
        with archive.open(csv_names[0]) as source, output_path.open("wb") as destination:
            destination.write(source.read())
    zip_path.unlink(missing_ok=True)
    return output_path


def read_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Read raw data and validate the expected schema."""
    data_path = Path(data_path)
    if not data_path.exists():
        msg = (
            f"Data file not found: {data_path}. Run `make download` or place "
            "online_shoppers_intention.csv into data/raw/."
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(data_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        msg = f"Dataset has unexpected schema. Missing columns: {missing}"
        raise ValueError(msg)
    return df


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Revenue to integer labels while preserving all feature columns."""
    result = df.copy()
    if result[TARGET].dtype == bool:
        result[TARGET] = result[TARGET].astype(int)
    else:
        result[TARGET] = (
            result[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
        )
    if result[TARGET].isna().any():
        msg = "Target column contains values that cannot be mapped to 0/1."
        raise ValueError(msg)
    result[TARGET] = result[TARGET].astype(int)
    return result


def load_dataset(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate, minimally clean, and split into X/y."""
    df = read_raw_data(data_path)
    df = normalize_target(df)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
