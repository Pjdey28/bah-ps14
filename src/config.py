from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RAW_DATA = DATA / "raw"

PROCESSED_DATA = DATA / "processed"

GOES_PROCESSED = PROCESSED_DATA / "GOES"

WIND_PROCESSED = PROCESSED_DATA / "WIND"

TRAIN_DATA = PROCESSED_DATA / "train"

ARTIFACTS = PROJECT_ROOT / "artifacts"

MODELS = ARTIFACTS / "models"

EVALUATION = ARTIFACTS / "evaluation"

SHAP_DIR = ARTIFACTS / "shap"

LOGS = PROJECT_ROOT / "logs"

RANDOM_STATE = 42

TEST_SIZE = 0.2

SEQUENCE_LENGTH = 72

FORECAST_STEPS = [6,72,144]

TARGETS = [
    "target_30min",
    "target_6hr",
    "target_12hr"
]