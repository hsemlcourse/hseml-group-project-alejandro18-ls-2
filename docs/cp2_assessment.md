# CP2 assessment and max-score checklist

## Важное ограничение оценки

В загруженных файлах есть требования курса, шаблон README, описание задачи, тип задачи и ссылка на датасет, но нет текущего содержимого GitHub-репозитория: нет notebook/script outputs, моделей, таблиц экспериментов, Dockerfile, requirements с версиями, линтеров и тестов. Поэтому текущий уровень можно оценить только по видимым артефактам, а не по фактической ветке `cp2`.

## Предварительная оценка по видимым артефактам

| Категория CP2 | Максимум CP2 | Видимый текущий уровень | Почему |
|---|---:|---:|---|
| Обработка данных | 5 | 0–1 | Есть постановка задачи и ссылка на датасет, но нет подтвержденных EDA, очистки, outlier analysis, leakage-safe split и визуализаций с выводами. |
| Моделирование и эксперименты | 18 | 0 | Нет подтвержденных baseline, 4–5 моделей, ensembles, hyperparameter search, таблицы экспериментов и обоснования финальной модели. |
| Качество кода и воспроизводимость | 7 | 0 | Нет подтвержденных Docker/docker-compose, ruff/flake8, fixed seed в коде, requirements с версиями и воспроизводимой структуры. |
| **Итого** | **30** | **0–1 / 30** | Это не оценка реальной работы в GitHub, а оценка только по загруженным файлам. |

## Что закрывает этот CP2-пакет

### Обработка данных

- [x] Проверка схемы датасета.
- [x] Удаление дублей.
- [x] Приведение target `Revenue` к 0/1.
- [x] Imputation внутри sklearn pipeline.
- [x] Outlier analysis: IQR summary.
- [x] Outlier handling: quantile clipping, fit только на train/CV fold.
- [x] Feature engineering с поведенческими агрегатами.
- [x] Визуализации зависимостей признаков от target.
- [x] Stratified train/val/test split и пояснение, как избегается data leakage.
- [x] Обоснование основной метрики PR-AUC и дополнительных метрик.

### Моделирование и эксперименты

- [x] Baseline без feature engineering.
- [x] 4–5+ моделей: Logistic Regression, KNN, Decision Tree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM.
- [x] Явное ансамблирование: soft voting и stacking.
- [x] RandomizedSearchCV для перебора гиперпараметров.
- [x] Таблица экспериментов в `artifacts/experiment_results.csv`.
- [x] Эксперимент с уменьшением размерности: `logreg_svd_dim_reduction`.
- [x] Выбор финальной модели по validation PR-AUC.
- [x] Интерпретируемость: feature importance и permutation importance.

### Качество кода и воспроизводимость

- [x] Чистая структура проекта.
- [x] `requirements.txt` с зафиксированными версиями.
- [x] Fixed seed в `src/config.py`.
- [x] Ruff в `pyproject.toml`.
- [x] Smoke-тесты в `tests/test_pipeline.py`.
- [x] Dockerfile.
- [x] docker-compose.yml.
- [x] README с запуском и структурой.

## Перед сдачей CP2

1. Запустить `make all`.
2. Проверить, что появились:
   - `report/images/*.png`;
   - `artifacts/experiment_results.csv`;
   - `artifacts/test_metrics.json`;
   - `artifacts/cp2_results.md`;
   - `models/best_model.joblib`.
3. Открыть `artifacts/cp2_results.md` и перенести топовую таблицу результатов в README, если нужно.
4. Запустить `make lint` и `make test`.
5. Закоммитить в ветку `cp2`.
