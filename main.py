import os
import sys

import pandas as pd

from config import (
    DATASET_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from load_data import load_data, data_info
from eda import eda_pipeline
from evaluation import evaluate_all_models
from feature_engineering import feature_engineering_pipeline
from preprocessing import preprocess_data
from trainmodel import build_models, train_all_models


def main():
    print("=" * 60)
    print("  CREDIT RISK ASSESSMENT ML PIPELINE")
    print("=" * 60)

    for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    print("\n[1/7] Loading data...")
    df = load_data()
    data_info(df)

    print("\n[2/7] Feature engineering...")
    df = feature_engineering_pipeline(df)
    print(f"New features added. Shape: {df.shape}")

    processed_path = PROCESSED_DATA_DIR / "processed_credit_data.csv"
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

    print("\n[3/7] Running EDA...")
    age_col = "alter"
    eda_pipeline(df, TARGET_COLUMN, NUMERICAL_FEATURES, age_col, FIGURES_DIR)

    print("\n[4/7] Preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("\n[5/7] Building models...")
    models = build_models()

    print("\n[6/7] Training models...")
    trained_models = train_all_models(models, X_train, y_train)

    print("\n[7/7] Evaluating models...")
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
