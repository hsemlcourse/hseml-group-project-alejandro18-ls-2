"""Generate CP2 EDA visualizations and data-quality summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import NUMERIC_COLUMNS, RANDOM_STATE, TARGET
from src.data import load_dataset
from src.preprocessing import build_preprocessor


def outlier_summary(X: pd.DataFrame) -> pd.DataFrame:
    """Estimate outlier counts using the IQR rule for numeric features."""
    rows: list[dict[str, float | int | str]] = []
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(X[column], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = values.lt(lower) | values.gt(upper)
        rows.append(
            {
                "feature": column,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(mask.sum()),
                "outlier_share": float(mask.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_share", ascending=False)


def plot_target_distribution(y: pd.Series, output_dir: Path) -> None:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No purchase", "Purchase"], counts.values)
    ax.set_title("Target distribution: Revenue")
    ax.set_ylabel("Sessions")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value}\n{value / len(y):.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)


def plot_conversion_by_category(df: pd.DataFrame, category: str, output_dir: Path) -> None:
    rates = df.groupby(category)[TARGET].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rates.index.astype(str), rates.values)
    ax.set_title(f"Conversion rate by {category}")
    ax.set_ylabel("Purchase share")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / f"conversion_by_{category}.png", dpi=160)
    plt.close(fig)


def plot_numeric_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    selected = ["PageValues", "ProductRelated", "ProductRelated_Duration", "ExitRates", "BounceRates"]
    for column in selected:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df.loc[df[TARGET].eq(label), column].dropna().values for label in [0, 1]]
        ax.boxplot(data, tick_labels=["No purchase", "Purchase"], showfliers=False)
        ax.set_title(f"{column} by target")
        ax.set_ylabel(column)
        fig.tight_layout()
        fig.savefig(output_dir / f"boxplot_{column}_by_target.png", dpi=160)
        plt.close(fig)


def plot_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    corr = df[[*NUMERIC_COLUMNS, TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr.values, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation matrix for numeric features")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=160)
    plt.close(fig)


def plot_svd_projection(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> None:
    preprocessor = build_preprocessor(use_feature_engineering=True, scale_numeric=True, clip_outliers=True)
    transformed = preprocessor.fit_transform(X, y)
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    coords = svd.fit_transform(transformed)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, alpha=0.45, s=12)
    ax.set_title("2D projection after preprocessing and SVD")
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    fig.colorbar(scatter, ax=ax, label="Revenue")
    fig.tight_layout()
    fig.savefig(output_dir / "svd_projection.png", dpi=160)
    plt.close(fig)


def generate_eda(data_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(data_path)
    df = X.copy()
    df[TARGET] = y

    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns_without_target",
                "positive_rate",
                "duplicates_after_loading_removed",
                "missing_values_total",
            ],
            "value": [
                len(df),
                X.shape[1],
                y.mean(),
                0,
                int(df.isna().sum().sum()),
            ],
        }
    )
    summary.to_csv(output_dir / "data_quality_summary.csv", index=False)
    outlier_summary(X).to_csv(output_dir / "outlier_summary.csv", index=False)

    plot_target_distribution(y, output_dir)
    for category in ["Month", "VisitorType", "Weekend", "TrafficType"]:
        plot_conversion_by_category(df, category, output_dir)
    plot_numeric_boxplots(df, output_dir)
    plot_correlation(df, output_dir)
    plot_svd_projection(X, y, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA artifacts for CP2.")
    parser.add_argument("--data-path", default="data/raw/online_shoppers_intention.csv")
    parser.add_argument("--output-dir", default="report/images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_eda(args.data_path, args.output_dir)
