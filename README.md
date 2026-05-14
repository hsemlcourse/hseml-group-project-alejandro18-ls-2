# ML Project — Прогнозирование покупки пользователем интернет-магазина

**Студент:** `Яворский Александр Алексеевич`

**Группа:** `БИВ238`

## Оглавление

1.  [Описание задачи](#описание-задачи)
2.  [Структура репозитория](#структура-репозитория)
3.  [Запуск](#запуск)
4.  [Данные](#данные)
5.  [Что реализовано к CP1](#что-реализовано-к-cp1)
6.  [Результаты](#результаты)
7.  [Контроль качества](#контроль-качества)

## Описание задачи {#описание-задачи}

**Задача:** бинарная классификация пользовательских сессий интернет-магазина.

**Что предсказываем:** завершится ли сессия покупкой.

**Таргет:** `Revenue`.

**Положительный класс:** `Revenue = True`, то есть покупка была совершена.

**Датасет:** Kaggle — `adilshamim8/online`, Online Shoppers Purchasing Intention / Predictive Modeling of E-Commerce Purchase Intent.

**Почему датасет подходит:**

-   строк больше 10 000;
-   признаков больше 10;
-   есть бизнес-интерпретация: заранее находить сессии с высокой вероятностью покупки и точнее настраивать маркетинговые действия.

**Главная метрика:** `F1` по положительному классу.

Я выбираю `F1`, потому что классы несбалансированы: покупок существенно меньше, чем сессий без покупки. Для бизнеса важно не только находить как можно больше потенциальных покупателей (`recall`), но и не помечать слишком много обычных сессий как покупателей (`precision`). Дополнительно считаются `ROC-AUC`, `PR-AUC`, `precision`, `recall`, `balanced_accuracy`.

## Структура репозитория {#структура-репозитория}

Структура сохранена близко к шаблону, но без файлов для CP2/CP3, которые пока не нужны.

``` text
.
├── data
│   ├── processed               # train/validation/test и краткое EDA-описание
│   └── raw                     # исходный CSV online_shoppers_intention.csv
├── models                      # best_model.joblib и experiments.csv
├── report
│   └── images                  # графики EDA для CP1
├── src
│   ├── __init__.py
│   ├── config.py               # пути, seed, имя таргета
│   ├── download_data.py        # загрузка датасета
│   ├── eda.py                  # EDA и визуализации
│   ├── modeling.py             # модели, метрики, эксперименты
│   ├── preprocessing.py        # очистка, split, feature engineering
│   └── train.py                # основной запуск CP1
├── tests
│   └── test_pipeline.py        # smoke-тесты пайплайна
├── pyproject.toml              # настройки ruff/pytest
├── requirements.txt
└── README.md
```

## Запуск {#запуск}

### 1. Создать окружение и установить зависимости

``` bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Скачать данные

Автоматически:

``` bash
python -m src.download_data
```

Если Kaggle не скачивается из-за настроек доступа, скачайте файл вручную со страницы датасета и положите CSV сюда:

``` text
data/raw/online_shoppers_intention.csv
```

Код также умеет искать первый `.csv` в `data/raw/`, если имя файла отличается.

### 3. Построить EDA-графики и краткое описание данных

``` bash
python -m src.eda
```

После запуска появятся:

``` text
data/processed/eda_summary.json
report/images/target_distribution.png
report/images/conversion_by_month.png
report/images/conversion_by_visitor_type.png
report/images/numeric_correlation.png
```

### 4. Обучить baseline и модели CP1

Быстрый запуск без подбора гиперпараметров:

``` bash
python -m src.train
```

Запуск с небольшим подбором гиперпараметров:

``` bash
python -m src.train --tune
```

После запуска появятся:

``` text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
models/experiments.csv
models/best_model.joblib
models/test_metrics.json
```

### 5. Проверить качество кода

``` bash
ruff check .
pytest
```

## Данные {#данные}

Ожидаемые колонки исходного датасета:

-   числовые: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`;
-   категориальные/дискретные: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`;
-   таргет: `Revenue`.

В пайплайне есть нормализация альтернативных названий колонок, например `Administrative Duration` → `Administrative_Duration`.

## Что реализовано к CP1 {#что-реализовано-к-cp1}

### Обработка и подготовка данных

-   загрузка исходного CSV;
-   нормализация названий колонок;
-   приведение `Revenue` и `Weekend` к булевому типу;
-   удаление дублей;
-   проверка пропусков;
-   корректный стратифицированный split `train/valid/test = 70/15/15`;
-   предотвращение data leakage: `SimpleImputer`, `QuantileClipper`, `StandardScaler`, `OneHotEncoder` обучаются только на train внутри `Pipeline`;
-   обработка выбросов через клиппинг числовых признаков по квантилям, границы считаются только на train;
-   feature engineering:
    -   `TotalPages`;
    -   `TotalDuration`;
    -   `ProductTimePerPage`;
    -   `AdminTimePerPage`;
    -   `InfoTimePerPage`;
    -   `EngagementRate`;
    -   `IsReturningVisitor`;
    -   `HasSpecialDay`;
-   EDA-графики для баланса таргета, конверсии по месяцу, конверсии по типу посетителя и корреляций.

### Моделирование и эксперименты

Реализованы:

-   baseline: `LogisticRegression` без feature engineering;
-   `LogisticRegression` с feature engineering;
-   `KNeighborsClassifier`;
-   `RandomForestClassifier`;
-   `ExtraTreesClassifier`;
-   `GradientBoostingClassifier`;
-   `HistGradientBoostingClassifier`;
-   ablation-эксперимент без `PageValues`, потому что этот признак обычно очень сильный и его полезно отдельно проверить на риск зависимости от бизнес-логики расчёта.

Итоги всех запусков сохраняются в `models/experiments.csv`.

### Воспроизводимость

-   общий `RANDOM_STATE = 42`;
-   все split и модели с seed зафиксированы;
-   зависимости вынесены в `requirements.txt`;
-   ruff настроен в `pyproject.toml`;
-   есть smoke-тесты для preprocessing/modeling pipeline.

## Результаты {#результаты}

| Модель | F1 valid | ROC-AUC valid | PR-AUC valid | Примечание |
|----|---:|---:|---:|----|
| baseline_logreg_no_fe | 0.631 | 0.895 | 0.619 | baseline без feature engineering |
| logreg_fe | 0.626 | 0.895 | 0.617 | линейная модель + новые признаки |
| knn_fe | 0.441 | 0.851 | 0.574 | метрический baseline |
| random_forest_fe | 0.646 | 0.923 | 0.733 | bagging/ансамбль деревьев |
| extra_trees_fe | 0.615 | 0.903 | 0.645 | более рандомизированный ансамбль |
| gradient_boosting_fe | 0.645 | 0.926 | 0.730 | boosting |
| hist_gradient_boosting_fe | 0.619 | 0.922 | 0.720 | быстрый boosting |
| random_forest_no_page_values | 0.273 | 0.781 | 0.366 | ablation без PageValues |

Лучшая модель на validation по F1 — `random_forest_fe`: F1 = 0.646, ROC-AUC = 0.923, PR-AUC = 0.733. 
На test она получила F1 = 0.681, ROC-AUC = 0.929, PR-AUC = 0.757.

Ablation-эксперимент без признака `PageValues` резко ухудшил качество: F1 упал с 0.646 до 0.273, PR-AUC — с 0.733 до 0.366. Это показывает, что `PageValues` является одним из самых важных признаков для предсказания покупки.
