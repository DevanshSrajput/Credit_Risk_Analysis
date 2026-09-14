import joblib

from config import MODEL_FILE, PREPROCESSOR_FILE, RISK_THRESHOLDS


def load_model():
    model = joblib.load(MODEL_FILE)
    return model


def load_preprocessor():
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    return preprocessor


def calculate_probability(model, customer_data):
    probability = model.predict_proba(customer_data)[:, 1]
    return probability[0] if probability.ndim > 0 else probability


def calculate_risk_score(probability):
    score = round(probability * 100, 2)
    return score


def classify_risk(score):
    if score < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    elif score < RISK_THRESHOLDS["high"]:
        return "Medium Risk"
    return "High Risk"


def generate_risk_report(customer_data):
    model = load_model()
    probability = calculate_probability(model, customer_data)
    score = calculate_risk_score(probability)
    category = classify_risk(score)
    return {
        "Prediction": "Bad Credit" if probability >= 0.5 else "Good Credit",
        "Probability": round(probability, 4),
        "Risk Score": score,
        "Risk Category": category,
    }
