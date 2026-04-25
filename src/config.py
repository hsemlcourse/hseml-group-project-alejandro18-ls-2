"""Project-level constants."""

from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "Revenue"
POSITIVE_CLASS_LABEL = 1

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "report"
REPORT_IMAGES_DIR = REPORT_DIR / "images"

RAW_DATA_PATH = RAW_DATA_DIR / "online_shoppers_intention.csv"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VALID_PATH = PROCESSED_DATA_DIR / "valid.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
EDA_SUMMARY_PATH = PROCESSED_DATA_DIR / "eda_summary.json"

EXPERIMENTS_PATH = MODELS_DIR / "experiments.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
TEST_METRICS_PATH = MODELS_DIR / "test_metrics.json"
