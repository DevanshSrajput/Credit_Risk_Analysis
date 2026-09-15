import pandas as pd
import streamlit as st

from src.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PREPROCESSOR_FILE,
    RISK_THRESHOLDS,
)

st.set_page_config(page_title="Credit Risk Assessment", layout="wide")


@st.cache_resource
def load_model():
    import joblib
    model_path = MODELS_DIR / "credit_risk_model.pkl"
    preprocessor_path = PREPROCESSOR_FILE
    if not model_path.exists() or not preprocessor_path.exists():
        return None, None
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def main():
    st.title("Credit Risk Assessment")
    st.markdown("Enter customer details to predict credit risk.")

    model, preprocessor = load_model()
    if model is None:
        st.error(
            "Model not found. Please run `python main.py` first to train the model."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        laufkont = st.selectbox(
            "Account Status (laufkont)", [1, 2, 3, 4],
            format_func=lambda x: {1: "No checking account", 2: "< 0 DM", 3: "0 <= ... < 200 DM", 4: ">= 200 DM"}.get(x, str(x)),
        )
        laufzeit = st.slider("Loan Duration in months (laufzeit)", 1, 72, 24)
        moral = st.selectbox(
            "Credit History (moral)", [0, 1, 2, 3, 4],
            format_func=lambda x: {0: "No credits / all paid", 1: "All credits at this bank paid", 2: "Existing credits paid duly", 3: "Past payment delay", 4: "Critical account / other credits"}.get(x, str(x)),
        )
        verw = st.selectbox(
            "Purpose (verw)", list(range(10)),
            format_func=lambda x: {0: "Car (new)", 1: "Car (used)", 2: "Furniture/equipment", 3: "Radio/television", 4: "Domestic appliances", 5: "Repairs", 6: "Education", 7: "Vacation", 8: "Retraining", 9: "Other"}.get(x, str(x)),
        )
        hoehe = st.number_input("Credit Amount in DM (hoehe)", 250, 20000, 3500, step=100)
        sparkont = st.selectbox(
            "Savings Account (sparkont)", [1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Unknown / no savings", 2: "< 100 DM", 3: "100 <= ... < 500 DM", 4: "500 <= ... < 1000 DM", 5: ">= 1000 DM"}.get(x, str(x)),
        )

    with col2:
        beszeit = st.selectbox(
            "Employment Since (beszeit)", [1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Unemployed", 2: "< 1 year", 3: "1 <= ... < 4 years", 4: "4 <= ... < 7 years", 5: ">= 7 years"}.get(x, str(x)),
        )
        rate = st.selectbox(
            "Installment Rate (rate)", [1, 2, 3, 4],
            format_func=lambda x: {1: ">= 35%", 2: "25% <= ... < 35%", 3: "15% <= ... < 25%", 4: "< 15%"}.get(x, str(x)),
        )
        famges = st.selectbox(
            "Personal Status & Sex (famges)", [1, 2, 3, 4],
            format_func=lambda x: {1: "Male: divorced/separated", 2: "Male: single", 3: "Male: married/widowed", 4: "Female"}.get(x, str(x)),
        )
        buerge = st.selectbox(
            "Guarantor (buerge)", [1, 2, 3],
            format_func=lambda x: {1: "None", 2: "Co-applicant", 3: "Guarantor"}.get(x, str(x)),
        )
        wohnzeit = st.selectbox(
            "Residence (wohnzeit)", [1, 2, 3, 4],
            format_func=lambda x: {1: "< 1 year", 2: "1 <= ... < 4 years", 3: "4 <= ... < 7 years", 4: ">= 7 years"}.get(x, str(x)),
        )
        verm = st.selectbox(
            "Property (verm)", [1, 2, 3, 4],
            format_func=lambda x: {1: "Real estate", 2: "Building society savings / life insurance", 3: "Car or other", 4: "Unknown / no property"}.get(x, str(x)),
        )

    col3, col4 = st.columns(2)

    with col3:
        alter = st.slider("Age in years (alter)", 18, 75, 35)
        weitkred = st.selectbox(
            "Other Credits (weitkred)", [1, 2, 3],
            format_func=lambda x: {1: "None", 2: "Bank", 3: "Store"}.get(x, str(x)),
        )
        wohn = st.selectbox(
            "Housing (wohn)", [1, 2, 3],
            format_func=lambda x: {1: "Rent", 2: "Own", 3: "For free"}.get(x, str(x)),
        )
        bishkred = st.selectbox(
            "Number of Existing Credits (bishkred)", [1, 2, 3, 4],
            format_func=lambda x: {1: "None", 2: "1-2", 3: "3-4", 4: "5+"}.get(x, str(x)),
        )

    with col4:
        beruf = st.selectbox(
            "Job (beruf)", [1, 2, 3, 4],
            format_func=lambda x: {1: "Unskilled", 2: "Unskilled (resident)", 3: "Skilled", 4: "Professional"}.get(x, str(x)),
        )
        pers = st.selectbox(
            "Number of People Liable (pers)", [1, 2],
            format_func=lambda x: {1: "1 (single)", 2: "2+"}.get(x, str(x)),
        )
        telef = st.selectbox(
            "Telephone (telef)", [1, 2],
            format_func=lambda x: {1: "None", 2: "Yes"}.get(x, str(x)),
        )
        gastarb = st.selectbox(
            "Foreign Worker (gastarb)", [1, 2],
            format_func=lambda x: {1: "Yes", 2: "No"}.get(x, str(x)),
        )

    if st.button("Predict Credit Risk", type="primary"):
        customer_data = pd.DataFrame(
            {
                "laufkont": [laufkont],
                "laufzeit": [laufzeit],
                "moral": [moral],
                "verw": [verw],
                "hoehe": [hoehe],
                "sparkont": [sparkont],
                "beszeit": [beszeit],
                "rate": [rate],
                "famges": [famges],
                "buerge": [buerge],
                "wohnzeit": [wohnzeit],
                "verm": [verm],
                "alter": [alter],
                "weitkred": [weitkred],
                "wohn": [wohn],
                "bishkred": [bishkred],
                "beruf": [beruf],
                "pers": [pers],
                "telef": [telef],
                "gastarb": [gastarb],
            }
        )

        from src.feature_engineering import feature_engineering_pipeline
        customer_data = feature_engineering_pipeline(customer_data)
        preprocessed = preprocessor.transform(customer_data)
        prediction = model.predict(preprocessed)[0]
        probability = model.predict_proba(preprocessed)[0][1]
        risk_score = round(probability * 100, 2)

        if risk_score < RISK_THRESHOLDS["low"]:
            risk_category = "Low Risk"
        elif risk_score < RISK_THRESHOLDS["high"]:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"

        st.markdown("---")
        st.subheader("Prediction Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", "Bad Credit" if prediction == 1 else "Good Credit")
        c2.metric("Probability", f"{probability:.1%}")
        c3.metric("Risk Score", f"{risk_score}")
        c4.metric("Risk Category", risk_category)

        if risk_category == "High Risk":
            st.error(f"This customer is classified as **{risk_category}**.")
        elif risk_category == "Medium Risk":
            st.warning(f"This customer is classified as **{risk_category}**.")
        else:
            st.success(f"This customer is classified as **{risk_category}**.")


if __name__ == "__main__":
    main()
