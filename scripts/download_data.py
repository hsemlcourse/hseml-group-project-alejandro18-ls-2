"""Download Online Shoppers Purchasing Intention data.

The course project dataset link points to Kaggle. For reproducible automated runs this script
uses the public UCI mirror of the same dataset, because it does not require a Kaggle token.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import download_from_uci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw CSV dataset from UCI.")
    parser.add_argument("--output", default="data/raw/online_shoppers_intention.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = download_from_uci(Path(args.output))
    print(f"Downloaded dataset to {output}")
