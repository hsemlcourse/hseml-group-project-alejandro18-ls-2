# Прогнозирование покупки в интернет-магазине

**Студент:** Яворский Александр Алексеевич БИВ238 **Чекпоинт:** CP2\
**Тип задачи:** бинарная классификация\
**Репозиторий:** `hseml-group-project-...`

## Оглавление

1.  [Краткое описание проекта](#краткое-описание-проекта)
2.  [Данные](#данные)
3.  [Метрика качества](#метрика-качества)
4.  [Структура репозитория](#структура-репозитория)
5.  [Быстрый запуск](#быстрый-запуск)
6.  [Запуск в PowerShell](#запуск-в-powershell)
7.  [Что реализовано в CP2](#что-реализовано-в-cp2)
8.  [Результаты последнего запуска](#результаты-последнего-запуска)
9.  [Артефакты после запуска](#артефакты-после-запуска)
10. [Тесты, линтер и Docker](#тесты-линтер-и-docker)
11. [Ограничения и честные замечания](#ограничения-и-честные-замечания)

## Краткое описание проекта {#краткое-описание-проекта}

Цель проекта — предсказать, завершится ли пользовательская сессия интернет-магазина покупкой.

Формально это задача бинарной классификации:

-   `Revenue = 1` — сессия завершилась покупкой;
-   `Revenue = 0` — сессия не завершилась покупкой.

Модель получает поведенческие и технические признаки сессии: количество просмотренных страниц разных типов, длительность просмотра, bounce/exit rates, page value, месяц, тип посетителя, регион, браузер, операционную систему, источник трафика и другие признаки.

Практический смысл задачи: такая модель может помогать интернет-магазину находить сессии с высокой вероятностью покупки и использовать это для аналитики, персонализации, ретаргетинга или оценки качества трафика.

## Данные {#данные}

Используется датасет **Online Shoppers Purchasing Intention Dataset**.

Изначально проект выбран на основе Kaggle-датасета:

``` text
https://www.kaggle.com/datasets/adilshamim8/online
```

Для воспроизводимого запуска в проекте добавлен скрипт загрузки из публичного UCI-зеркала, так как Kaggle обычно требует авторизацию через API-токен.

Ожидаемый путь к данным после загрузки:

``` text
data/raw/online_shoppers_intention.csv
```

Ключевые признаки датасета:

| Группа признаков | Примеры |
|------------------------------------|------------------------------------|
| Поведение на сайте | `Administrative`, `Informational`, `ProductRelated` |
| Длительность сессии | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` |
| Метрики поведения | `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| Технические признаки | `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| Контекстные признаки | `Month`, `VisitorType`, `Weekend` |
| Целевая переменная | `Revenue` |

В последнем локальном запуске использовался split:

| Split      | Количество строк |
|------------|-----------------:|
| Train      |            8 543 |
| Validation |            1 831 |
| Test       |            1 831 |

Split выполняется стратифицированно, чтобы сохранить долю положительного класса в train/validation/test.

## Метрика качества {#метрика-качества}

Основная метрика проекта — **Average Precision**, то есть площадь под Precision-Recall curve, часто обозначаемая как **PR-AUC**.

Причина выбора: покупка является более редким событием, чем отсутствие покупки. В такой ситуации обычная accuracy может быть завышена: модель может часто предсказывать класс `0` и всё равно выглядеть неплохо по accuracy. PR-AUC лучше отражает качество модели именно по положительному классу, который важен для задачи.

Дополнительно считаются:

-   `ROC-AUC` — качество ранжирования по всем порогам;
-   `F1` — баланс precision и recall при выбранном пороге;
-   `Precision` — доля верных предсказаний покупки среди всех предсказанных покупок;
-   `Recall` — доля найденных реальных покупок;
-   `Balanced Accuracy` — accuracy с учетом дисбаланса классов;
-   `Accuracy` — общая доля верных ответов.

Выбор финальной модели выполняется по validation `average_precision`. Порог классификации выбирается отдельно на validation-выборке по максимуму `F1`. Test-выборка используется только для финальной оценки.

## Структура репозитория {#структура-репозитория}

``` text
.
├── artifacts/                 # Результаты запусков: метрики, таблицы, importance, predictions
├── data/
│   ├── processed/             # Зарезервировано под обработанные данные
│   └── raw/                   # Исходный CSV датасета
├── docs/
│   └── cp2_assessment.md      # Чеклист и самооценка по CP2
├── models/                    # Сохранённая финальная модель
├── notebooks/
│   ├── 01_eda.ipynb           # EDA
│   ├── 02_baseline.ipynb      # Baseline-модель
│   └── 03_experiments.ipynb   # Эксперименты CP2
├── report/
│   ├── images/                # Графики EDA
│   └── cp2_methodology.md     # Методология CP2
├── scripts/
│   └── download_data.py       # Загрузка датасета
├── src/
│   ├── config.py              # Константы, признаки, random seed
│   ├── data.py                # Загрузка, проверка схемы и target preprocessing
│   ├── eda.py                 # EDA, графики, outlier summary
│   ├── metrics.py             # Метрики и подбор threshold
│   ├── modeling.py            # Модели, ансамбли, подбор гиперпараметров
│   ├── preprocessing.py       # Очистка, feature engineering, split
│   └── train.py               # Основной pipeline обучения и оценки
├── tests/
│   └── test_pipeline.py       # Smoke-тесты pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml             # Настройки ruff и pytest
├── requirements.txt           # Зафиксированные версии библиотек
└── README.md
```

## Быстрый запуск {#быстрый-запуск}

Команды для Linux/macOS или Git Bash:

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py --output data/raw/online_shoppers_intention.csv
python -m src.eda --data-path data/raw/online_shoppers_intention.csv --output-dir report/images
python -m src.train --data-path data/raw/online_shoppers_intention.csv --output-dir artifacts --models-dir models

pytest -q
ruff check .
ruff format --check .
```

Также доступны команды через `make`:

``` bash
make install
make download
make eda
make train
make test
make lint
```

## Запуск в PowerShell {#запуск-в-powershell}

Создать и активировать виртуальное окружение:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Если PowerShell запрещает активацию окружения:

``` powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

Установить зависимости:

``` powershell
pip install -r requirements.txt
```

Скачать данные:

``` powershell
python scripts/download_data.py --output data/raw/online_shoppers_intention.csv
```

Построить EDA-графики:

``` powershell
python -m src.eda --data-path data/raw/online_shoppers_intention.csv --output-dir report/images
```

Запустить обучение и эксперименты:

``` powershell
python -m src.train --data-path data/raw/online_shoppers_intention.csv --output-dir artifacts --models-dir models
```

Проверить, что основные артефакты созданы:

``` powershell
Test-Path .\artifacts\cp2_results.md
Test-Path .\artifacts\experiment_results.csv
Test-Path .\artifacts\test_metrics.json
Test-Path .\models\best_model.joblib
```

Ожидаемый результат для каждой команды:

``` text
True
```

Запустить тесты и линтер:

``` powershell
pytest -q
ruff check .
ruff format --check .
```

## Что реализовано в CP2 {#что-реализовано-в-cp2}

### 1. Обработка данных

Реализовано:

-   загрузка данных из CSV;
-   проверка обязательных колонок;
-   приведение target `Revenue` к бинарному формату `0/1`;
-   удаление дубликатов;
-   стратифицированный train/validation/test split;
-   обработка пропусков внутри `sklearn Pipeline`;
-   one-hot encoding категориальных признаков;
-   масштабирование числовых признаков для моделей, где это нужно;
-   анализ выбросов через IQR summary;
-   clipping выбросов по квантилям только на train-fold, чтобы не допустить data leakage;
-   EDA-визуализации: распределение target, связи признаков с target, boxplots, корреляции, SVD projection.

Feature engineering:

-   `total_pages`;
-   `total_duration`;
-   `avg_time_per_page`;
-   `product_page_share`;
-   `informational_page_share`;
-   `administrative_page_share`;
-   `bounce_exit_ratio`;
-   `page_value_log1p`;
-   `engagement_score`;
-   `is_returning_visitor`;
-   `is_new_visitor`;
-   `is_special_day`;
-   `season`;
-   `month_index`.

Главный принцип обработки: все трансформации, которые могут обучаться на данных, находятся внутри `Pipeline` и fit выполняется только на train/CV fold. Validation и test не используются для fit preprocessing-компонентов.

### 2. Моделирование и эксперименты

В проекте реализованы:

-   baseline-модель;
-   набор классических моделей;
-   градиентный бустинг;
-   ансамбли;
-   подбор гиперпараметров;
-   сравнение моделей в единой таблице;
-   финальная оценка на test;
-   интерпретация через feature importance и permutation importance.

Используемые модели и подходы:

| Тип                      | Модели                                      |
|--------------------------|---------------------------------------------|
| Baseline                 | Logistic Regression без feature engineering |
| Линейные модели          | Logistic Regression                         |
| Метрические модели       | KNN                                         |
| Деревья                  | Decision Tree                               |
| Bagging/Randomized trees | Random Forest, ExtraTrees                   |
| Boosting                 | GradientBoosting, XGBoost, LightGBM         |
| Ensembles                | Soft Voting, Stacking                       |
| Dimensionality reduction | Logistic Regression + TruncatedSVD          |

Гиперпараметры перебираются через `RandomizedSearchCV` со `StratifiedKFold`.

### 3. Интерпретируемость

После обучения сохраняются:

``` text
artifacts/feature_importance.csv
artifacts/permutation_importance.csv
```

Если финальная модель поддерживает встроенную важность признаков или коэффициенты, они сохраняются отдельно. Дополнительно считается permutation importance как model-agnostic способ оценки вклада признаков.

### 4. Качество кода и воспроизводимость

Реализовано:

-   фиксированный `RANDOM_STATE = 42`;
-   версии библиотек зафиксированы в `requirements.txt`;
-   проект разбит на модули в `src/`;
-   есть smoke-тесты в `tests/`;
-   подключен `ruff`;
-   добавлены `Dockerfile` и `docker-compose.yml`;
-   README содержит структуру проекта, команды запуска и описание результатов.

В коде используются стандартные для ML-проектов обозначения `X`, `y`, `X_train`, `X_val`, `X_test`. Для Ruff это должно быть отражено в настройках как осознанное исключение правил naming convention `N803`/`N806`, потому что такие имена являются общепринятыми в sklearn-экосистеме.

## Результаты последнего запуска {#результаты-последнего-запуска}

Последний локальный запуск завершился успешно. Лучшая модель:

``` text
gradient_boosting_fe
```

Финальные метрики на test split:

| Метрика                    | Значение |
|----------------------------|---------:|
| Average Precision / PR-AUC |   0.7612 |
| ROC-AUC                    |   0.9328 |
| F1                         |   0.6958 |
| Precision                  |   0.6699 |
| Recall                     |   0.7238 |
| Balanced Accuracy          |   0.8289 |
| Accuracy                   |   0.9011 |
| Selected threshold         |   0.3450 |

Confusion matrix на test split:

|          | Predicted 0 | Predicted 1 |
|----------|------------:|------------:|
| Actual 0 |        1443 |         102 |
| Actual 1 |          79 |         207 |

Интерпретация результата:

-   модель хорошо отделяет сессии с покупкой от сессий без покупки: `ROC-AUC = 0.9328`;
-   `PR-AUC = 0.7612` показывает, что модель полезна именно для поиска редкого положительного класса;
-   recall `0.7238` означает, что модель находит около 72% реальных покупок на test split;
-   precision `0.6699` означает, что среди сессий, которые модель относит к покупкам, около 67% действительно являются покупками;
-   accuracy `0.9011` высокая, но она не используется как главная метрика из-за дисбаланса классов.

Финальная модель выбрана не только из-за качества на validation, но и из-за практических свойств: градиентный бустинг хорошо работает с табличными данными, умеет моделировать нелинейные зависимости и взаимодействия признаков, при этом остается достаточно воспроизводимым и интерпретируемым через importance-анализ.

## Артефакты после запуска {#артефакты-после-запуска}

После запуска `src.train` создаются:

| Файл | Назначение |
|------------------------------------|------------------------------------|
| `artifacts/experiment_results.csv` | Таблица всех экспериментов и validation-метрик |
| `artifacts/test_metrics.json` | Финальные test-метрики выбранной модели |
| `artifacts/cp2_results.md` | Markdown-отчет по результатам CP2 |
| `artifacts/test_predictions.csv` | Предсказания финальной модели на test split |
| `artifacts/feature_importance.csv` | Встроенная важность признаков, если доступна |
| `artifacts/permutation_importance.csv` | Permutation importance |
| `models/best_model.joblib` | Сохраненная финальная модель |

После запуска `src.eda` создаются EDA-артефакты в `report/images/` и/или `artifacts/`, в зависимости от указанного `--output-dir`.

## Тесты, линтер и Docker {#тесты-линтер-и-docker}

### Тесты

Запуск:

``` bash
pytest -q
```

В последнем локальном запуске smoke-тесты прошли успешно:

``` text
5 passed
```

Тесты проверяют:

-   наличие ожидаемых колонок;
-   работу feature engineering;
-   clipping выбросов;
-   корректность stratified split;
-   способность pipeline обучаться и возвращать метрики.

### Ruff

Запуск:

``` bash
ruff check .
ruff format --check .
```

Автоформатирование:

``` bash
ruff check . --fix
ruff format .
```

### Docker

Сборка и запуск через Docker Compose:

``` bash
docker compose build
docker compose run --rm cp2 python -m src.train
```

Запуск тестов внутри контейнера:

``` bash
docker compose run --rm cp2 pytest -q
```

Запуск Ruff внутри контейнера:

``` bash
docker compose run --rm cp2 ruff check .
```
