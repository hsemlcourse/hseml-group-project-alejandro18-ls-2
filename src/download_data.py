"""Download the dataset into data/raw/.

Primary source is the Kaggle dataset from the project statement:
https://www.kaggle.com/datasets/adilshamim8/online

Fallback is UCI repository id=468, which hosts the same Online Shoppers Purchasing
Intention dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, RAW_DATA_PATH


def _copy_first_csv(source_dir: Path, destination: Path) -> bool:
    csv_files = sorted(source_dir.rglob("*.csv"))
    if not csv_files:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_files[0], destination)
    return True


def download_from_kaggle() -> bool:
    """Try downloading public Kaggle dataset through kagglehub."""
    try:
        import kagglehub
    except ImportError:
        return False

    try:
        downloaded_dir = Path(kagglehub.dataset_download("adilshamim8/online"))
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")
        return False

    return _copy_first_csv(downloaded_dir, RAW_DATA_PATH)


def download_from_uci() -> bool:
    """Fallback download from UCI repository."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return False

    try:
        dataset = fetch_ucirepo(id=468)
        features = dataset.data.features
        targets = dataset.data.targets
        df = pd.concat([features, targets], axis=1)
    except Exception as exc:
        print(f"UCI download failed: {exc}")
        return False

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    return True


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists():
        print(f"Raw dataset already exists: {RAW_DATA_PATH}")
        return

    if download_from_kaggle():
        print(f"Downloaded dataset from Kaggle to {RAW_DATA_PATH}")
        return

    if download_from_uci():
        print(f"Downloaded dataset from UCI fallback to {RAW_DATA_PATH}")
        return

    raise RuntimeError(
        "Could not download data automatically. "
        "Please download CSV from Kaggle and place it into data/raw/."
    )


if __name__ == "__main__":
    main()
