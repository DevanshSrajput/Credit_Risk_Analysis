import os

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data_loader import load_data, data_info
from src.eda import eda_pipeline
from src.evaluate import evaluate_all_models
from src.preprocessing import preprocess_data
from src.train_model import build_models, train_all_models

EDA_NUMERICAL_FEATURES = ["laufzeit", "hoehe", "alter"]


def main():
    print("=" * 60)
    print("  CREDIT RISK ASSESSMENT ML PIPELINE")
    print("=" * 60)

    for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    print("\n[1/6] Loading data...")
    df = load_data()
    data_info(df)

    processed_path = PROCESSED_DATA_DIR / "processed_credit_data.csv"
    df.to_csv(processed_path, index=False)
    print(f"Raw data saved to {processed_path}")

    print("\n[2/6] Running EDA...")
    age_col = "alter"
    eda_pipeline(df, TARGET_COLUMN, EDA_NUMERICAL_FEATURES, age_col, FIGURES_DIR)

    print("\n[3/6] Preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("\n[4/6] Building models...")
    models = build_models()

    print("\n[5/6] Training models...")
    trained_models = train_all_models(models, X_train, y_train)

    print("\n[6/6] Evaluating models...")
    best_model = evaluate_all_models(trained_models, X_test, y_test)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best model: {type(best_model).__name__}")
    print(f"  Model saved to: {MODELS_DIR / 'credit_risk_model.pkl'}")
    print(f"  Reports saved to: {REPORTS_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
