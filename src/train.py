"""Main CP1 training script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    EXPERIMENTS_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    TEST_METRICS_PATH,
)
from src.modeling import (
    calculate_metrics,
    choose_best_model,
    fit_single_model,
    get_candidate_models,
    save_metrics,
    save_model,
)
from src.preprocessing import make_train_valid_test_split, make_xy, read_raw_data, save_splits


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


def run_experiments(input_path: Path = RAW_DATA_PATH, *, tune: bool = False) -> pd.DataFrame:
    """Run baseline, model candidates and one ablation experiment."""
    raw_path = find_csv(input_path)
    raw_df = read_raw_data(raw_path)

    x, y = make_xy(raw_df)
    split = make_train_valid_test_split(x, y)
    save_splits(split, PROCESSED_DATA_DIR)

    fitted_models = {}
    results = []

    for model_name, (estimator, use_feature_engineering) in get_candidate_models().items():
        print(f"Training {model_name}...")
        model, row = fit_single_model(
            model_name,
            estimator,
            split.x_train,
            split.y_train,
            split.x_valid,
            split.y_valid,
            use_feature_engineering=use_feature_engineering,
            tune=tune,
        )
        fitted_models[model_name] = model
        results.append(row)

    # Ablation: model without PageValues to check how much the strongest GA metric drives quality.
    x_no_page_values, y_no_page_values = make_xy(raw_df, drop_columns={"PageValues"})
    ablation_split = make_train_valid_test_split(x_no_page_values, y_no_page_values)
    rf_estimator, rf_use_fe = get_candidate_models()["random_forest_fe"]
    ablation_model, ablation_row = fit_single_model(
        "random_forest_no_page_values",
        rf_estimator,
        ablation_split.x_train,
        ablation_split.y_train,
        ablation_split.x_valid,
        ablation_split.y_valid,
        use_feature_engineering=rf_use_fe,
        tune=False,
    )
    fitted_models["random_forest_no_page_values"] = ablation_model
    results.append(ablation_row)

    results_df = pd.DataFrame(results).sort_values(["f1", "pr_auc"], ascending=False)
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(EXPERIMENTS_PATH, index=False)

    best_name = choose_best_model(results)
    best_model = fitted_models[best_name]

    if best_name == "random_forest_no_page_values":
        test_metrics = calculate_metrics(best_model, ablation_split.x_test, ablation_split.y_test)
    else:
        test_metrics = calculate_metrics(best_model, split.x_test, split.y_test)

    save_model(best_model, BEST_MODEL_PATH)
    save_metrics(test_metrics, TEST_METRICS_PATH)

    print("\nValidation experiments:")
    print(results_df.to_string(index=False))
    print("\nBest model:", best_name)
    print("Test metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP1 training experiments.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Path to raw CSV.")
    parser.add_argument("--tune", action="store_true", help="Run small RandomizedSearchCV grids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(args.input, tune=args.tune)


if __name__ == "__main__":
    main()
