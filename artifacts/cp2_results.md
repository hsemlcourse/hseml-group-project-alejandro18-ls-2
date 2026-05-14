# CP2 results — Online Shoppers Purchase Intention

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

**gradient_boosting_fe**

Модель сохранена в `models\best_model.joblib`.

## Test metrics

| Metric | Value |
|---|---:|
| average_precision | 0.761206 |
| roc_auc | 0.932829 |
| f1 | 0.695798 |
| precision | 0.669903 |
| recall | 0.723776 |
| balanced_accuracy | 0.828878 |
| accuracy | 0.901147 |
| threshold | 0.345000 |
| tp | 207 |
| fp | 102 |
| tn | 1443 |
| fn | 79 |
| selected_model | gradient_boosting_fe |
| selection_metric | average_precision |
| train_rows | 8543 |
| validation_rows | 1831 |
| test_rows | 1831 |

## Experiment table, top 10 by validation average_precision

| model                | searched   |   cv_best_score |   val_average_precision |   val_roc_auc |   val_f1 |   val_precision |   val_recall |   val_threshold |   fit_seconds | note                                                                        |
|:---------------------|:-----------|----------------:|------------------------:|--------------:|---------:|----------------:|-------------:|----------------:|--------------:|:----------------------------------------------------------------------------|
| gradient_boosting_fe | True       |        0.747463 |                0.770912 |      0.940524 | 0.711256 |        0.666667 |     0.762238 |           0.345 |       102.914 | Boosting из sklearn для табличных данных.                                   |
| xgboost_fe           | True       |        0.728648 |                0.764822 |      0.940263 | 0.698835 |        0.666667 |     0.734266 |           0.395 |        61.902 | Gradient boosting с учетом class imbalance через scale_pos_weight.          |
| soft_voting_ensemble | True       |        0.747895 |                0.76193  |      0.941682 | 0.703947 |        0.664596 |     0.748252 |           0.545 |       125.72  | Явное ансамблирование: soft voting по линейной, bagging и boosting моделям. |
| extra_trees_fe       | True       |        0.752474 |                0.759994 |      0.938912 | 0.688356 |        0.674497 |     0.702797 |           0.755 |       137.346 | Случайный лес с более сильной рандомизацией.                                |
| lightgbm_fe          | True       |        0.734815 |                0.758314 |      0.934501 | 0.695652 |        0.692042 |     0.699301 |           0.7   |        53.118 | LightGBM: быстрый boosting для табличных данных.                            |
| stacking_ensemble    | True       |        0.742027 |                0.753986 |      0.941281 | 0.695205 |        0.681208 |     0.70979  |           0.83  |       323.154 | Явное ансамблирование: stacking с meta-моделью Logistic Regression.         |
| random_forest_fe     | True       |        0.743445 |                0.752412 |      0.936459 | 0.681458 |        0.623188 |     0.751748 |           0.615 |       136.572 | Bagging-ансамбль, устойчивый к выбросам и нелинейностям.                    |
| decision_tree_fe     | True       |        0.689041 |                0.718872 |      0.928959 | 0.667797 |        0.648026 |     0.688811 |           0.7   |         4.174 | Интерпретируемое дерево решений.                                            |
| knn_fe               | True       |        0.69216  |                0.712144 |      0.917833 | 0.662632 |        0.584    |     0.765734 |           0.295 |         6.043 | Нелинейная distance-based модель как альтернативный класс алгоритмов.       |
| logreg_balanced_fe   | True       |        0.662937 |                0.679197 |      0.931254 | 0.700461 |        0.624658 |     0.797203 |           0.675 |         4.22  | Линейная модель с class_weight для дисбаланса классов.                      |

## Как интерпретировать

1. Сначала сравниваем модели по validation PR-AUC: она лучше отражает качество ранжирования покупателей при дисбалансе классов.
2. Затем смотрим F1/precision/recall на выбранном threshold, чтобы понять компромисс между количеством найденных покупателей и ложными срабатываниями.
3. Feature importance и permutation importance показывают, какие факторы реально влияют на модель. Для этой задачи обычно ожидаются PageValues, ExitRates/BounceRates, ProductRelated и признаки VisitorType/Month, но окончательный вывод нужно делать по сохранённым таблицам.
