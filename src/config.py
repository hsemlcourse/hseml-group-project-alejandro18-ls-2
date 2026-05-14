"""Project-level constants for the Online Shoppers CP2 pipeline."""

from __future__ import annotations

RANDOM_STATE = 42
TARGET = "Revenue"

NUMERIC_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

REQUIRED_COLUMNS = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET]

ENGINEERED_NUMERIC_COLUMNS = [
    "total_pages",
    "total_duration",
    "product_page_share",
    "info_page_share",
    "admin_page_share",
    "avg_time_per_page",
    "bounce_exit_ratio",
    "page_value_log1p",
    "engagement_score",
    "month_index",
]

ENGINEERED_CATEGORICAL_COLUMNS = [
    "is_returning_visitor",
    "is_new_visitor",
    "is_special_day",
    "season",
]

ALL_NUMERIC_COLUMNS_WITH_FEATURES = [*NUMERIC_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS]
ALL_CATEGORICAL_COLUMNS_WITH_FEATURES = [*CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL_COLUMNS]

MONTH_TO_INDEX = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

MODEL_SELECTION_METRIC = "average_precision"
SECONDARY_METRICS = [
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "accuracy",
]
