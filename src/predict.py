import pandas as pd
import joblib

from src.config import MODEL_FILE, PREPROCESSOR_FILE
from src.risk_score import calculate_probability, calculate_risk_score, classify_risk


def load_artifacts():
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    return model, preprocessor


def preprocess_input(customer_data, preprocessor):
    processed_data = preprocessor.transform(customer_data)
    return processed_data


def predict_customer(customer_data):
    model, preprocessor = load_artifacts()
    preprocessed_data = preprocess_input(customer_data, preprocessor)
    probability = calculate_probability(model, preprocessed_data)
    risk_score = calculate_risk_score(probability)
    risk_category = classify_risk(risk_score)

    return {
        "Prediction": "Bad Credit" if probability >= 0.5 else "Good Credit",
        "Probability": round(probability, 4),
        "Risk Score": risk_score,
        "Risk Category": risk_category,
    }


if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "laufkont": [1],
            "laufzeit": [24],
            "moral": [2],
            "verw": [1],
            "hoehe": [3500],
            "sparkont": [2],
            "beszeit": [3],
            "rate": [2],
            "famges": [3],
            "buerge": [1],
            "wohnzeit": [2],
            "verm": [2],
            "alter": [35],
            "weitkred": [1],
            "wohn": [2],
            "bishkred": [1],
            "beruf": [3],
            "pers": [1],
            "telef": [1],
            "gastarb": [1],
        }
    )

    result = predict_customer(sample_data)
    print("\nPrediction Result:")
    print("=" * 50)
    for key, value in result.items():
        print(f"{key}: {value}")
