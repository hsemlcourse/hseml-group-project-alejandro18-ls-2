# CP2 methodology notes

## Постановка задачи

Нужно предсказать, завершится ли пользовательская сессия покупкой. Это бинарная классификация: `Revenue = 1` означает покупку, `Revenue = 0` — отсутствие покупки.

## Метрики

Основная метрика — `average_precision` / PR-AUC. Причина: классы несбалансированы, покупка встречается заметно реже, чем отсутствие покупки. В такой задаче accuracy может быть завышенной: модель может почти всегда отвечать «покупки нет» и всё равно получать хороший accuracy. PR-AUC оценивает, насколько хорошо модель ранжирует редкие положительные объекты.

Дополнительные метрики:

-   ROC-AUC — общая разделимость классов.
-   F1 — баланс precision и recall на выбранном threshold.
-   Precision — насколько чистые положительные предсказания.
-   Recall — какую долю покупателей нашли.
-   Balanced accuracy — accuracy с учетом дисбаланса.

Если метрики расходятся, приоритет такой: сначала validation PR-AUC, затем F1/precision/recall на threshold, выбранном на validation.

## Очистка и data leakage

Данные сначала делятся на train/validation/test в пропорции 70/15/15 со stratify по target. Все операции, у которых есть fit-статистика, находятся внутри sklearn pipeline:

-   imputers;
-   quantile clipping;
-   one-hot encoding;
-   scaling;
-   dimensionality reduction.

Это важно: медианы, квантили, категории one-hot и параметры scaler не видят validation/test до оценки.

## Выбросы

Для финансово-поведенческих данных выбросы ожидаемы: длинные сессии, много просмотренных товаров, высокие `PageValues`. Поэтому стратегия не удаляет строки агрессивно. Вместо этого:

1.  `src.eda.outlier_summary` считает IQR summary и долю выбросов.
2.  В pipeline используется `QuantileClipper`, который обрезает числовые признаки по 1% и 99% квантилям, fit только на train/CV fold.

Так выбросы не ломают линейные/KNN-модели, но информация о сильной вовлеченности пользователя сохраняется.

## Feature engineering

Добавлены признаки, которые помогают выразить поведение сессии:

-   `total_pages` — суммарное количество просмотренных страниц.
-   `total_duration` — суммарная длительность сессии по типам страниц.
-   `avg_time_per_page` — среднее время на страницу.
-   `product_page_share`, `info_page_share`, `admin_page_share` — структура сессии.
-   `bounce_exit_ratio` — соотношение bounce и exit rates.
-   `page_value_log1p` — лог-преобразование сильно скошенного PageValues.
-   `engagement_score` — взаимодействие глубины и длительности сессии.
-   `is_returning_visitor`, `is_new_visitor`, `is_special_day`, `season`, `month_index`.

## Визуализации и ожидаемые выводы

После `make eda` появляются графики в `report/images`:

-   `target_distribution.png` — показывает дисбаланс классов.
-   `conversion_by_Month.png` — помогает увидеть сезонность конверсии.
-   `conversion_by_VisitorType.png` — сравнение новых и возвращающихся посетителей.
-   `conversion_by_Weekend.png` — проверка эффекта выходного дня.
-   `conversion_by_TrafficType.png` — проверка разных источников трафика.
-   `boxplot_PageValues_by_target.png` — обычно PageValues заметно выше у покупок.
-   `boxplot_ExitRates_by_target.png` и `boxplot_BounceRates_by_target.png` — ожидается, что высокие bounce/exit связаны с меньшей вероятностью покупки.
-   `correlation_matrix.png` — числовые зависимости и мультиколлинеарность.
-   `svd_projection.png` — визуальная проверка разделимости после preprocessing.

## Эксперименты

`src/modeling.py` запускает следующие группы экспериментов:

1.  Baseline Logistic Regression без feature engineering.
2.  Линейная модель с feature engineering и class weights.
3.  KNN как альтернативная нелинейная модель.
4.  Decision Tree как интерпретируемая модель.
5.  Random Forest и ExtraTrees как bagging ensembles.
6.  GradientBoosting, XGBoost и LightGBM как boosting-модели.
7.  Soft Voting ensemble.
8.  Stacking ensemble.
9.  Logistic Regression + TruncatedSVD как эксперимент с уменьшением размерности.

Гиперпараметры перебираются через `RandomizedSearchCV` с `StratifiedKFold` и scoring `average_precision`.

## Выбор финальной модели

Финальная модель выбирается по validation PR-AUC. После выбора:

1.  threshold подбирается на validation по F1;
2.  модель с выбранными гиперпараметрами refit на train+validation;
3.  test split используется один раз для финальной оценки;
4.  сохраняются `test_metrics.json`, `test_predictions.csv`, `best_model.joblib`.

## Интерпретируемость

Сохраняются два типа importance:

-   `feature_importance.csv` — если выбранная модель имеет coefficients или feature_importances;
-   `permutation_importance.csv` — model-agnostic importance по validation data.

В финальных выводах нужно ссылаться на эти таблицы: почему выбранная модель лучше baseline/альтернатив, какие признаки важнее и насколько результат устойчив по нескольким метрикам.
