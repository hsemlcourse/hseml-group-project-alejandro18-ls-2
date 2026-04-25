"""Simple EDA artifacts for CP1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    EDA_SUMMARY_PATH,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    REPORT_IMAGES_DIR,
    TARGET_COLUMN,
)
from src.preprocessing import BASE_NUMERIC_COLUMNS, clean_data, read_raw_data


def find_csv(path: Path) -> Path:
    """Find raw CSV path."""
    if path.exists():
        return path
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/. Run `python -m src.download_data` first."
        )
    return csv_files[0]


def save_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot target class counts."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No purchase", "Purchase"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts.values)
    plt.title("Target distribution")
    plt.ylabel("Sessions")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close()


def save_conversion_by_month(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot conversion rate by month."""
    if "Month" not in df.columns:
        return

    conversion = df.groupby("Month", observed=True)[TARGET_COLUMN].mean().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(conversion.index.astype(str), conversion.values)
    plt.title("Conversion rate by month")
    plt.ylabel("Share of sessions with purchase")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "conversion_by_month.png", dpi=160)
    plt.close()


def save_conversion_by_visitor_type(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot conversion rate by visitor type."""
    if "VisitorType" not in df.columns:
        return

    conversion = (
        df.groupby("VisitorType", observed=True)[TARGET_COLUMN].mean().sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(conversion.index.astype(str), conversion.values)
    plt.title("Conversion rate by visitor type")
    plt.ylabel("Share of sessions with purchase")
    plt.xlabel("Visitor type")
    plt.tight_layout()
    plt.savefig(output_dir / "conversion_by_visitor_type.png", dpi=160)
    plt.close()


def save_numeric_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot numeric correlations including target."""
    available = [column for column in BASE_NUMERIC_COLUMNS if column in df.columns]
    if not available:
        return

    corr_df = df[[*available, TARGET_COLUMN]].copy()
    corr_df[TARGET_COLUMN] = corr_df[TARGET_COLUMN].astype(int)
    corr = corr_df.corr(numeric_only=True)

    plt.figure(figsize=(11, 9))
    image = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Numeric feature correlations")
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_correlation.png", dpi=160)
    plt.close()


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    """Create a compact JSON summary for README/report."""
    target_rate = float(df[TARGET_COLUMN].mean())
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    summary: dict[str, object] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": TARGET_COLUMN,
        "positive_class_share": target_rate,
        "duplicates_after_cleaning": int(df.duplicated().sum()),
        "missing_values": missing.astype(int).to_dict(),
    }

    if "VisitorType" in df.columns:
        summary["conversion_by_visitor_type"] = (
            df.groupby("VisitorType", observed=True)[TARGET_COLUMN].mean().sort_values(ascending=False).to_dict()
        )
    if "Month" in df.columns:
        summary["conversion_by_month"] = (
            df.groupby("Month", observed=True)[TARGET_COLUMN].mean().sort_values(ascending=False).to_dict()
        )

    return summary


def run_eda(input_path: Path = RAW_DATA_PATH) -> dict[str, object]:
    """Run EDA and save summary/figures."""
    path = find_csv(input_path)
    df = clean_data(read_raw_data(path))

    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    EDA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_target_distribution(df, REPORT_IMAGES_DIR)
    save_conversion_by_month(df, REPORT_IMAGES_DIR)
    save_conversion_by_visitor_type(df, REPORT_IMAGES_DIR)
    save_numeric_correlation(df, REPORT_IMAGES_DIR)

    summary = build_summary(df)
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CP1 EDA artifacts.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Path to raw CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eda(args.input)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
