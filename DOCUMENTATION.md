# Credit Risk Assessment ML Project - Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset Description](#dataset-description)
5. [Feature Description](#feature-description)
6. [Project Architecture](#project-architecture)
7. [Technologies Used](#technologies-used)
8. [Installation](#installation)
9. [How to Run](#how-to-run)
10. [Module Reference](#module-reference)
11. [EDA Results](#eda-results)
12. [Model Results](#model-results)
13. [Model Comparison](#model-comparison)
14. [Risk Scoring](#risk-scoring)
15. [Streamlit Application](#streamlit-application)
16. [Future Improvements](#future-improvements)

---

## Project Overview

This project builds a machine learning pipeline to assess credit risk using the German Credit dataset. It predicts whether a customer is likely to have good or bad credit based on 20 financial and demographic features. The project is designed as a teaching tool, with a modular architecture that separates data loading, feature engineering, preprocessing, model training, evaluation, and deployment.

---

## Problem Statement

Banks and financial institutions need to assess the creditworthiness of loan applicants. Manual assessment is time-consuming and inconsistent. An ML-based system can:

- Automate credit risk evaluation
- Provide consistent, data-driven decisions
- Quantify risk with probability scores
- Reduce default rates by identifying high-risk applicants

---

## Objectives

1. Load and explore the German Credit dataset
2. Perform Exploratory Data Analysis (EDA) to understand patterns
3. Engineer meaningful features from raw data
4. Build a preprocessing pipeline with proper train/test splitting
5. Train and compare multiple ML algorithms (Logistic Regression, Decision Tree, Random Forest)
6. Evaluate models using accuracy, precision, recall, F1, and ROC-AUC
7. Select the best model and save it for inference
8. Build a risk scoring system that converts probabilities to understandable scores
9. Deploy via a Streamlit web application

---

## Dataset Description

**Source:** UCI Machine Learning Repository - German Credit Dataset

| Property | Value |
|----------|-------|
| Total Samples | 1,000 |
| Features | 20 (all integer-encoded) |
| Target | `kredit` (0 = Bad Credit, 1 = Good Credit) |
| Class Distribution | 30% Bad (300), 70% Good (700) |
| Missing Values | None |
| Feature Types | All integer (numerical and categorical encoded as integers) |

---

## Feature Description

### Numerical Features

| Feature | German Name | Description | Unique Values |
|---------|-------------|-------------|---------------|
| Duration | `laufzeit` | Loan duration in months | 33 |
| Credit Amount | `hoehe` | Credit amount in DM | 923 |
| Age | `alter` | Age in years | 53 |

### Categorical Features (Integer-Encoded)

| Feature | German Name | Description | Values |
|---------|-------------|-------------|--------|
| Account Status | `laufkont` | Status of checking account | 1=No checking, 2=<0 DM, 3=0-200 DM, 4=>=200 DM |
| Credit History | `moral` | Credit history of applicant | 0=No credits, 1=All paid, 2=Existing paid duly, 3=Past delay, 4=Critical |
| Purpose | `verw` | Purpose of credit | 0-9 (Car new, Car used, Furniture, Radio/TV, Appliances, Repairs, Education, Vacation, Retraining, Other) |
| Savings Account | `sparkont` | Savings account/bonds | 1=Unknown, 2=<100, 3=100-500, 4=500-1000, 5=>=1000 |
| Employment | `beszeit` | Present employment since | 1=Unemployed, 2=<1yr, 3=1-4yr, 4=4-7yr, 5=>=7yr |
| Installment Rate | `rate` | Installment rate as % of disposable income | 1=>=35%, 2=25-35%, 3=15-25%, 4=<15% |
| Personal Status | `famges` | Personal status and sex | 1=Male div/separated, 2=Male single, 3=Male married, 4=Female |
| Guarantor | `buerge` | Other debtors/guarantors | 1=None, 2=Co-applicant, 3=Guarantor |
| Residence | `wohnzeit` | Present residence since | 1=<1yr, 2=1-4yr, 3=4-7yr, 4=>=7yr |
| Property | `verm` | Property | 1=Real estate, 2=Building society, 3=Car/other, 4=Unknown |
| Other Credits | `weitkred` | Other installment banks | 1=None, 2=Bank, 3=Store |
| Housing | `wohn` | Housing | 1=Rent, 2=Own, 3=Free |
| Existing Credits | `bishkred` | Number of existing credits | 1=None, 2=1-2, 3=3-4, 4=5+ |
| Job | `beruf` | Job | 1=Unskilled, 2=Unskilled resident, 3=Skilled, 4=Professional |
| People Liable | `pers` | Number of people providing maintenance | 1=Single, 2=2+ |
| Telephone | `telef` | Telephone | 1=None, 2=Yes |
| Foreign Worker | `gastarb` | Foreign worker | 1=Yes, 2=No |

### Engineered Features

| Feature | Description | Formula |
|---------|-------------|---------|
| `Credit_per_Month` | Monthly credit obligation | `hoehe / laufzeit` |
| `Longterm_Loan` | Whether loan duration exceeds average | `laufzeit > mean(laufzeit)` (binary) |
| `Age_Group` | Categorical age bracket | Bins: 18-25, 25-40, 40-60, 60+ |

### Target Variable

| Value | Meaning | Count |
|-------|---------|-------|
| 0 | Bad Credit (default risk) | 300 (30%) |
| 1 | Good Credit | 700 (70%) |

---

## Project Architecture

```
Credit_Risk_Analysis/
├── data/
│   ├── raw/
│   │   └── german_credit_data.csv      # Original dataset
│   └── processed/
│       └── processed_credit_data.csv    # Cleaned output
├── models/
│   ├── credit_risk_model.pkl            # Trained model
│   ├── preprocessor.pkl                 # Fitted ColumnTransformer
│   └── model_metadata.pkl               # Best model info
├── notebooks/                           # Jupyter notebooks
├── reports/
│   ├── figures/                         # Generated plots
│   ├── model_comparison.csv             # All model metrics
│   └── model_report.txt                 # Best model summary
├── src/
│   ├── __init__.py                      # Package marker
│   ├── config.py                        # Central configuration
│   ├── data_loader.py                   # Load & inspect data
│   ├── feature_engineering.py           # Create new features
│   ├── preprocessing.py                 # ColumnTransformer pipeline
│   ├── eda.py                           # Exploratory data analysis
│   ├── train_model.py                   # Train ML models
│   ├── evaluate.py                      # Evaluate & compare models
│   ├── risk_score.py                    # Probability to risk score
│   ├── predict.py                       # Inference pipeline
│   └── utils.py                         # Reusable helpers
├── app.py                               # Streamlit dashboard
├── main.py                              # Pipeline orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

### Pipeline Flow

```
main.py
  │
  ├── [1] Load Data ──────────── data_loader.py
  │
  ├── [2] EDA ────────────────── eda.py
  │
  ├── [3] Preprocess ─────────── preprocessing.py
  │     ├── Remove duplicates
  │     ├── Feature engineering (feature_engineering.py)
  │     ├── Train/test split
  │     └── ColumnTransformer (StandardScaler + OneHotEncoder)
  │
  ├── [4] Build Models ───────── train_model.py
  │
  ├── [5] Train Models ───────── train_model.py
  │
  └── [6] Evaluate ───────────── evaluate.py
        ├── Metrics (accuracy, precision, recall, F1, ROC-AUC)
        ├── Confusion matrices
        ├── ROC curves
        ├── Save best model
        └── Generate reports
```

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.14 | Programming language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models, preprocessing, metrics |
| matplotlib | Basic plotting |
| seaborn | Statistical visualizations |
| joblib | Model serialization |
| streamlit | Web application framework |
| plotly | Interactive charts |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/A-zbot/Credit_Risk_Analysis.git
cd Credit_Risk_Analysis

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Train the Model

```bash
python main.py
```

This runs the complete pipeline:
1. Loads the raw dataset
2. Runs EDA and generates plots
3. Engineers features
4. Preprocesses data with ColumnTransformer
5. Trains Logistic Regression, Decision Tree, Random Forest
6. Evaluates and selects the best model
7. Saves model, preprocessor, and reports

### Run the Streamlit App

```bash
streamlit run app.py
```

Opens a web interface where you can input customer details and get real-time credit risk predictions.

### Run Individual Modules

```bash
python -m src.data_loader      # Test data loading
python -m src.predict           # Test predictions
```

---

## EDA Results

The EDA generates the following visualizations in `reports/figures/`:

| Plot | Description |
|------|-------------|
| `class_distribution.png` | Target variable distribution (30/70 split) |
| `numerical_distribution.png` | Histograms of laufzeit, hoehe, alter |
| `correlation_heatmap.png` | Correlations between numerical features |
| `boxplots.png` | Boxplots of numerical features |
| `age_vs_risk.png` | Age distribution by credit risk class |
| `target_vs_numerical.png` | Numerical features by target class |

Key findings:
- Dataset is imbalanced: 70% good credit, 30% bad credit
- Credit amount (hoehe) and duration (laufzeit) are positively correlated
- Younger applicants tend to have higher default rates
- Higher credit amounts correlate with higher risk

---

## Model Results

### Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 0.8100 |
| Precision (Bad) | 0.7200 |
| Recall (Bad) | 0.6000 |
| F1 Score (Bad) | 0.6545 |
| ROC-AUC | 0.8274 |

### Decision Tree

| Metric | Value |
|--------|-------|
| Accuracy | 0.7000 |
| Precision (Bad) | 0.5000 |
| Recall (Bad) | 0.5667 |
| F1 Score (Bad) | 0.5312 |
| ROC-AUC | 0.8381 |

### Random Forest

| Metric | Value |
|--------|-------|
| Accuracy | 0.7900 |
| Precision (Bad) | 0.7647 |
| Recall (Bad) | 0.4333 |
| F1 Score (Bad) | 0.5532 |
| ROC-AUC | 0.8196 |

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.8100 | 0.7200 | 0.6000 | 0.6545 | **0.8274** |
| Decision Tree | 0.7000 | 0.5000 | 0.5667 | 0.5312 | 0.8381 |
| Random Forest | 0.7900 | 0.7647 | 0.4333 | 0.5532 | 0.8196 |

**Best Model:** Logistic Regression (highest ROC-AUC)

Note: Metrics are reported for the bad credit class (class 0) as the positive class, since identifying defaults is the primary goal.

---

## Risk Scoring

The risk scoring system converts model probability into an understandable score:

```
Model Probability (bad credit)
        │
        ▼
  Score = probability × 100
        │
        ├── 0–29   → Low Risk
        ├── 30–69  → Medium Risk
        └── 70–100 → High Risk
```

Example:
- Probability of bad credit: 0.174
- Risk Score: 17.4
- Category: Low Risk

---

## Streamlit Application

The web app provides:

- 20 input fields matching all dataset features
- Human-readable labels for all categorical values
- Real-time prediction with probability and risk score
- Color-coded risk category display (green/yellow/red)

Run with: `streamlit run app.py`

---

## Future Improvements

1. **XGBoost/LightGBM** - Add gradient boosting models
2. **SMOTE** - Handle class imbalance with oversampling
3. **Hyperparameter Tuning** - GridSearch/RandomSearch for optimal parameters
4. **SHAP Values** - Add model interpretability
5. **API Endpoint** - FastAPI for production deployment
6. **Database Integration** - Store predictions and track model performance
7. **CI/CD Pipeline** - Automated testing and deployment
8. **A/B Testing** - Compare model versions in production
