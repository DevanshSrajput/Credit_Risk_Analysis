import pandas as pd
import joblib
from config import MODEL_FILE, SCALER_FILE
from riskscore import calculate_probability, calculate_riskscore, classify_risk

def load_artifacts():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

def preprocess_input(customer_data, scaler):
    scaled_data = scaler.transform(customer_data)
    return scaled_data

def predict_customer(customer_data):
    model, scaler = load_artifacts()
    preprocessed_data = preprocess_input(customer_data, scaler)
    prediction = model.predict(preprocessed_data)[0]
    probability = calculate_probability(model, processed_data)  
    riskscore = calculate_riskscore(probability)
    risk_category = classify_risk(riskscore)
    
    return {
        "Prediction": prediction,
        "Probability": probability,
        "RiskScore": riskscore,
        "RiskCategory": risk_category
    }
    if __name__ == "__main__":

        sample_data = pd.DataFrame({
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
            "gastarb": [1]
        })

        result = predict_customer(sample_data)

        print("\nPrediction Result:")
        print("=" * 50)

        for key, value in result.items():
            print(f"{key}: {value}")

