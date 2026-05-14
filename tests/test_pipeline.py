"""Smoke tests for CP1 preprocessing/modeling pipeline."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.modeling import calculate_metrics, make_pipeline
from src.preprocessing import (
    add_engineered_features,
    clean_data,
    make_train_valid_test_split,
    make_xy,
)


def make_sample_df() -> pd.DataFrame:
    rows = []
    months = ["Feb", "Mar", "May", "Nov"]
    visitor_types = ["Returning_Visitor", "New_Visitor"]
    for idx in range(40):
        is_purchase = idx % 5 == 0
        rows.append(
            {
                "Administrative": idx % 4,
                "Administrative_Duration": float(idx * 2),
                "Informational": idx % 3,
                "Informational_Duration": float(idx),
                "ProductRelated": 3 + idx % 7,
                "ProductRelated_Duration": float(10 + idx * 3),
                "BounceRates": 0.01 * (idx % 10),
                "ExitRates": 0.02 * (idx % 8),
                "PageValues": 20.0 if is_purchase else 0.5 * (idx % 3),
                "SpecialDay": 0.2 if idx % 11 == 0 else 0.0,
                "Month": months[idx % len(months)],
                "OperatingSystems": idx % 3 + 1,
                "Browser": idx % 4 + 1,
                "Region": idx % 5 + 1,
                "TrafficType": idx % 6 + 1,
                "VisitorType": visitor_types[idx % len(visitor_types)],
                "Weekend": idx % 2 == 0,
                "Revenue": is_purchase,
            }
        )
    return pd.DataFrame(rows)


def test_clean_data_and_feature_engineering() -> None:
    df = make_sample_df()
    cleaned = clean_data(df)
    engineered = add_engineered_features(cleaned)

    assert "Revenue" in cleaned.columns
    assert cleaned["Revenue"].dtype == bool
    assert "TotalPages" in engineered.columns
    assert "EngagementRate" in engineered.columns


def test_model_pipeline_smoke() -> None:
    df = make_sample_df()
    x, y = make_xy(df)
    split = make_train_valid_test_split(x, y)

    model = make_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=500),
        split.x_train,
        use_feature_engineering=True,
    )
    model.fit(split.x_train, split.y_train)
    metrics = calculate_metrics(model, split.x_valid, split.y_valid)

    assert "f1" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0
