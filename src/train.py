"""Train CP2 models, run hyperparameter search, and save reproducible artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import MODEL_SELECTION_METRIC, RANDOM_STATE
from src.data import load_dataset
from src.metrics import choose_threshold, evaluate_model, predict_positive_probability
from src.modeling import (
    model_feature_importance,
    permutation_importance_table,
    refit_final_model,
    run_experiments,
    save_json,
    save_model,
    select_best_model,
)
from src.preprocessing import train_val_test_split


def plot_feature_importance(table: pd.DataFrame, output_path: Path) -> None:
    if table.empty:
        return
    top = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    value_column = "importance" if "importance" in top.columns else "importance_mean"
    ax.barh(top["feature"], top[value_column])
    ax.set_title("Top feature importances")
    ax.set_xlabel(value_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_cp2_markdown(
    experiment_table: pd.DataFrame,
    final_metrics: dict[str, Any],
    best_model_name: str,
    model_path: Path,
) -> str:
    top_table = experiment_table.head(10).copy()
    cols = [
        "model",
        "searched",
        "cv_best_score",
        f"val_{MODEL_SELECTION_METRIC}",
        "val_roc_auc",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_threshold",
        "fit_seconds",
        "note",
    ]
    top_table = top_table[[column for column in cols if column in top_table.columns]]
    return f"""# CP2 results — Online Shoppers Purchase Intention

## Что закрыто в CP2

- Стратифицированный train/validation/test split 70/15/15; test не участвует в подборе гиперпараметров и threshold.
- Очистка: удаление дублей, приведение target к 0/1, imputer внутри pipeline, quantile clipping выбросов по train-fold статистикам.
- Feature engineering: поведенческие агрегаты по страницам, длительности, долям типов страниц, PageValues log1p, признаки visitor/special day/season.
- Модели: baseline + Logistic Regression + KNN + Decision Tree + Random Forest + ExtraTrees + HistGradientBoosting + XGBoost/LightGBM при наличии + soft voting + stacking.
- Перебор гиперпараметров: RandomizedSearchCV с StratifiedKFold, scoring=`average_precision`.
- Уменьшение размерности: отдельный эксперимент `logreg_svd_dim_reduction`.
- Интерпретация: feature importance, permutation importance, confusion-matrix counts в метриках.

## Главная метрика

Основная метрика — **Average Precision / PR-AUC**, потому что положительный класс покупки редкий, а бизнесу важнее ранжировать и находить потенциальных покупателей, чем максимизировать обычную accuracy. ROC-AUC, F1, precision, recall и balanced accuracy используются как дополнительные метрики. Для бинарного решения threshold выбирается на validation по F1.

## Лучшая модель

**{best_model_name}**

Модель сохранена в `{model_path}`.

## Test metrics

| Metric | Value |
|---|---:|
""" + "\n".join(
        f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |"
        for key, value in final_metrics.items()
    ) + f"""

## Experiment table, top 10 by validation {MODEL_SELECTION_METRIC}

{top_table.to_markdown(index=False)}

## Как интерпретировать

1. Сначала сравниваем модели по validation PR-AUC: она лучше отражает качество ранжирования покупателей при дисбалансе классов.
2. Затем смотрим F1/precision/recall на выбранном threshold, чтобы понять компромисс между количеством найденных покупателей и ложными срабатываниями.
3. Feature importance и permutation importance показывают, какие факторы реально влияют на модель. Для этой задачи обычно ожидаются PageValues, ExitRates/BounceRates, ProductRelated и признаки VisitorType/Month, но окончательный вывод нужно делать по сохранённым таблицам.
"""


def train_and_evaluate(
    data_path: str | Path,
    output_dir: str | Path,
    models_dir: str | Path,
    *,
    n_iter: int = 8,
    cv_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, random_state=random_state
    )

    experiment_table, fitted_models = run_experiments(
        X_train,
        X_val,
        y_train,
        y_val,
        n_iter=n_iter,
        cv_splits=cv_splits,
        random_state=random_state,
    )
    experiments_path = output_dir / "experiment_results.csv"
    experiment_table.to_csv(experiments_path, index=False)

    best_model_name = select_best_model(experiment_table)
    best_model = fitted_models[best_model_name]
    validation_scores = predict_positive_probability(best_model, X_val)
    threshold = choose_threshold(y_val, validation_scores, metric="f1")

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    final_model = refit_final_model(best_model, X_train_val, y_train_val)
    test_metrics = evaluate_model(final_model, X_test, y_test, threshold=threshold)
    test_metrics["selected_model"] = best_model_name
    test_metrics["selection_metric"] = MODEL_SELECTION_METRIC
    test_metrics["train_rows"] = len(X_train)
    test_metrics["validation_rows"] = len(X_val)
    test_metrics["test_rows"] = len(X_test)

    model_path = save_model(final_model, models_dir / "best_model.joblib")
    metrics_path = save_json(test_metrics, output_dir / "test_metrics.json")

    test_predictions = X_test.copy()
    test_predictions["y_true"] = y_test
    test_predictions["score"] = predict_positive_probability(final_model, X_test)
    test_predictions["prediction"] = (test_predictions["score"] >= threshold).astype(int)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    importance = model_feature_importance(final_model)
    if not importance.empty:
        importance.to_csv(output_dir / "feature_importance.csv", index=False)
        plot_feature_importance(importance, output_dir / "feature_importance.png")

    permutation = permutation_importance_table(final_model, X_val, y_val)
    permutation.to_csv(output_dir / "permutation_importance.csv", index=False)
    plot_feature_importance(permutation, output_dir / "permutation_importance.png")

    cp2_markdown = build_cp2_markdown(experiment_table, test_metrics, best_model_name, model_path)
    (output_dir / "cp2_results.md").write_text(cp2_markdown, encoding="utf-8")

    return {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "experiments_path": str(experiments_path),
        "metrics_path": str(metrics_path),
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP2 training and experiments.")
    parser.add_argument("--data-path", default="data/raw/online_shoppers_intention.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_and_evaluate(
        data_path=args.data_path,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
    )
    print(result)
