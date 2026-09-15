from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_NAME = "german_credit_data.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_NAME
TARGET_COLUMN = "kredit"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_FILE = MODELS_DIR / "credit_risk_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.pkl"

NUMERICAL_FEATURES = ["laufzeit", "hoehe", "alter", "Credit_per_Month"]
CATEGORICAL_FEATURES = [
    "laufkont", "moral", "verw", "sparkont", "beszeit",
    "rate", "famges", "buerge", "wohnzeit", "verm",
    "weitkred", "wohn", "bishkred", "beruf", "pers",
    "telef", "gastarb", "Longterm_Loan",
]

BAD_CREDIT_CLASS = 0
GOOD_CREDIT_CLASS = 1

RISK_THRESHOLDS = {"low": 30, "high": 70}
