# Complete Credit Risk Assessment ML Project Structure

Since you want to **teach the project to a student**, I recommend using the following final structure. It keeps the ML workflow modular while still being simple enough to explain file by file.

```text
Credit-Risk-Assessment-ML/
│
├── data/
│   │
│   ├── raw/
│   │   └── german_credit_data.csv
│   │
│   └── processed/
│       └── processed_credit_data.csv
│
├── models/
│   ├── credit_risk_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.pkl
│
├── notebooks/
│   └── Credit_Risk_EDA.ipynb
│
├── reports/
│   │
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── numerical_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── boxplot_laufzeit.png
│   │   ├── boxplot_hoehe.png
│   │   ├── boxplot_alter.png
│   │   ├── age_vs_risk.png
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   │
│   ├── model_comparison.csv
│   └── model_report.txt
│
├── src/
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── risk_score.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# What Every File Does

## 1. `data/raw/german_credit_data.csv`

**Purpose:** Original dataset.

This is the dataset you provided.

```text
1000 customers
      ↓
20 features
      ↓
kredit = target
```

We keep this file untouched.

---

# 2. `data/processed/processed_credit_data.csv`

**Purpose:** Stores the cleaned dataset after preprocessing/feature engineering.

Flow:

```text
Raw Dataset
     ↓
Cleaning
     ↓
Feature Engineering
     ↓
Processed Dataset
```

This gives students a clear distinction between **raw data** and **processed data**.

---

# 3. `models/credit_risk_model.pkl`

**Purpose:** Stores the final trained ML model.

For example:

```text
Logistic Regression
        ↓
Decision Tree
        ↓
Random Forest
        ↓
Best Model
        ↓
credit_risk_model.pkl
```

The application loads this file instead of retraining the model.

---

# 4. `models/preprocessor.pkl`

**Purpose:** Stores the preprocessing pipeline.

It handles things such as:

```text
Categorical Features
       ↓
Encoding

Numerical Features
       ↓
Scaling
```

This is important because the exact same transformations must be applied to new customers.

---

# 5. `models/model_metadata.pkl`

**Purpose:** Stores information about the trained model.

Example:

```python
{
    "model_name": "Random Forest",
    "accuracy": 0.82,
    "precision": 0.78,
    "recall": 0.76,
    "f1_score": 0.77,
    "roc_auc": 0.89
}
```

This isn't mandatory for prediction, but it's useful for tracking experiments.

---

# 6. `notebooks/Credit_Risk_EDA.ipynb`

**Purpose:** Interactive experimentation.

Students can use it to explore:

```python
df.head()
df.info()
df.describe()
```

and experiment with visualizations.

The important distinction:

```text
Notebook
   ↓
Learning + experimentation

src/
   ↓
Actual reusable application code
```

We don't want the entire project living inside a gigantic notebook.

---

# 7. `src/__init__.py`

**Purpose:** Makes `src` a Python package.

It allows imports such as:

```python
from src.config import TARGET_COLUMN
```

It can remain very simple:

```python
"""
Credit Risk Assessment ML package.
"""
```

---

# 8. `src/config.py`

**Purpose:** Central configuration.

Contains things such as:

```text
Dataset path
Target column
Test size
Random state
Model path
Preprocessor path
Risk thresholds
```

Example:

```python
TARGET_COLUMN = "kredit"

TEST_SIZE = 0.20

RANDOM_STATE = 42
```

Instead of scattering these values throughout the project, we maintain them here.

---

# 9. `src/data_loader.py`

**Purpose:** Load and inspect the dataset.

Main responsibilities:

```text
CSV
 ↓
DataFrame
 ↓
Basic validation
 ↓
Return data
```

Functions will include things like:

```python
load_data()
dataset_info()
get_feature_target()
```

It **does not train models**.

---

# 10. `src/preprocessing.py`

**Purpose:** Prepare the dataset for Machine Learning.

This is where we'll use the corrected architecture:

```text
                 Dataset
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Numerical             Categorical
       Features               Features
          │                     │
          ▼                     ▼
   StandardScaler          OneHotEncoder
          │                     │
          └─────────┬───────────┘
                    ▼
             ColumnTransformer
                    │
                    ▼
              ML-ready data
```

It will also handle the train/test split.

This is one of the most important files in the project because it prevents **data leakage** and ensures training and prediction use identical transformations.

---

# 11. `src/feature_engineering.py`

**Purpose:** Create useful new features.

For example:

```text
Credit Amount
      +
Loan Duration
      ↓
Credit Per Month
```

Other possible features:

```text
Age Group
Long-Term Loan
Credit-to-Income-like ratios
```

The goal is to give the model useful information derived from existing data.

---

# 12. `src/eda.py`

**Purpose:** Exploratory Data Analysis.

It generates:

* Class distribution
* Histograms
* Correlation heatmap
* Box plots
* Feature distributions
* Risk comparisons

Example:

```text
Dataset
   ↓
EDA
   ↓
"What does our data actually look like?"
```

EDA happens **before serious model training** because otherwise we're essentially asking the model to solve a mystery while refusing to look at the clues.

---

# 13. `src/train_model.py`

**Purpose:** Train multiple Machine Learning algorithms.

We'll train:

```text
Logistic Regression
Decision Tree
Random Forest
```

Potentially later:

```text
XGBoost
```

The workflow:

```text
Training Data
      │
      ├── Logistic Regression
      │
      ├── Decision Tree
      │
      └── Random Forest
```

Each trained model is returned for evaluation.

---

# 14. `src/evaluate.py`

**Purpose:** Evaluate and compare models.

Metrics:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

Visualizations:

```text
Confusion Matrix
ROC Curve
```

Then:

```text
All Models
    ↓
Compare Performance
    ↓
Best Model
```

The best model will eventually be saved as:

```text
models/credit_risk_model.pkl
```

---

# 15. `src/risk_score.py`

**Purpose:** Convert the model's probability into an understandable risk score.

Example:

```text
Model Probability
       ↓
      0.86
       ↓
Risk Score
       ↓
      86
       ↓
High Risk
```

We'll define thresholds such as:

```text
0–29    Low Risk
30–69   Medium Risk
70–100  High Risk
```

These are **educational thresholds**, not actual banking policy.

---

# 16. `src/predict.py`

**Purpose:** Make predictions for new customers.

The inference pipeline becomes:

```text
New Customer
     ↓
Preprocessor
     ↓
Encoded / Scaled Data
     ↓
Trained Model
     ↓
Prediction
     ↓
Probability
     ↓
Risk Score
     ↓
Risk Category
```

Example result:

```text
Prediction      : Bad Credit
Probability     : 0.86
Risk Score      : 86
Risk Category   : High Risk
```

This module will eventually be used directly by `app.py`.

---

# 17. `src/utils.py`

**Purpose:** Generic reusable helper functions.

For example:

```python
create_project_directories()
print_section()
save_metrics()
check_file_exists()
```

The rule is:

> If a helper is generic and useful across multiple modules, it belongs here.

---

# 18. `main.py`

**Purpose:** Control the entire ML pipeline.

This becomes the project's main entry point.

The eventual flow:

```text
                main.py
                   │
                   ▼
            Create Directories
                   │
                   ▼
              Load Data
                   │
                   ▼
             Clean Data
                   │
                   ▼
         Feature Engineering
                   │
                   ▼
                 EDA
                   │
                   ▼
            Preprocessing
                   │
                   ▼
            Train Models
                   │
                   ▼
          Evaluate Models
                   │
                   ▼
            Select Best
                   │
                   ▼
             Save Model
```

The student should eventually be able to run:

```bash
python main.py
```

and execute the complete training workflow.

---

# 19. `app.py`

**Purpose:** Streamlit web application.

This is where the project becomes interactive.

The user enters information such as:

```text
Age
Loan Duration
Credit Amount
Employment
Savings
Housing
Purpose
...
```

Then:

```text
        [ Predict Credit Risk ]
                  ↓
           Machine Learning
                  ↓
       ┌─────────────────────┐
       │ Prediction: Bad     │
       │ Probability: 86%    │
       │ Risk Score: 86      │
       │ Category: High Risk │
       └─────────────────────┘
```

Run it with:

```bash
streamlit run app.py
```

---

# 20. `reports/figures/`

Contains graphs produced by the project.

For example:

```text
class_distribution.png
```

Shows the number of customers in each target class.

```text
correlation_heatmap.png
```

Shows relationships between numerical variables.

```text
confusion_matrix.png
```

Shows correct and incorrect predictions.

```text
roc_curve.png
```

Shows model discrimination performance.

These can be used in the student's project report and presentation.

---

# 21. `reports/model_comparison.csv`

Contains the performance of all models.

Example:

```text
Model                    Accuracy    Precision    Recall    F1    ROC-AUC
Logistic Regression      0.78        0.72         0.69      0.70  0.84
Decision Tree             0.73        0.66         0.64      0.65  0.76
Random Forest             0.82        0.78         0.76      0.77  0.89
```

This provides an easy way to demonstrate why the final model was selected.

---

# 22. `reports/model_report.txt`

A human-readable model summary.

For example:

```text
Credit Risk Assessment Model
============================

Best Model: Random Forest

Accuracy: 0.82
Precision: 0.78
Recall: 0.76
F1 Score: 0.77
ROC-AUC: 0.89
```

Useful for project documentation.

---

# 23. `requirements.txt`

Contains Python dependencies.

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
plotly
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 24. `.gitignore`

Prevents unnecessary files from being committed to Git.

Example:

```text
__pycache__/
*.pyc
venv/
.env
.ipynb_checkpoints/
models/*.pkl
data/processed/
```

This keeps the GitHub repository clean.

---

# 25. `README.md`

This is the project's documentation.

It should contain:

```text
1. Project Overview
2. Problem Statement
3. Objectives
4. Dataset Description
5. Feature Description
6. Project Architecture
7. Technologies Used
8. Installation
9. How to Run
10. EDA Results
11. Model Results
12. Model Comparison
13. Risk Scoring
14. Streamlit Application
15. Screenshots
16. Future Improvements
```

For a student project, this is important because someone should be able to clone the repository and understand what you've built without requiring a personal guided tour.

---

# Final Responsibility Map

The entire project can be remembered with this simple sequence:

```text
┌───────────────────────┐
│       DATA            │
│   german_credit.csv   │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│    data_loader.py     │
│      Load Data        │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│feature_engineering.py │
│    Create Features    │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│   preprocessing.py    │
│   Prepare Features    │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│       eda.py          │
│   Understand Data     │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│    train_model.py     │
│    Train Models       │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│     evaluate.py       │
│ Compare Models        │
└───────────┬───────────┘
            ↓
       BEST MODEL
            ↓
┌───────────────────────┐
│ credit_risk_model.pkl │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│      predict.py       │
│   New Customer        │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│    risk_score.py      │
│ Probability → Score   │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│        app.py         │
│   Streamlit Dashboard │
└───────────────────────┘
```

## The four layers students should remember

| Layer                | Files                                                          | Question it answers                                |
| -------------------- | -------------------------------------------------------------- | -------------------------------------------------- |
| **Data**             | `data_loader.py`, `feature_engineering.py`, `preprocessing.py` | What data do we have and how do we prepare it?     |
| **Analysis**         | `eda.py`                                                       | What patterns exist in the data?                   |
| **Machine Learning** | `train_model.py`, `evaluate.py`                                | Which model learns the problem best?               |
| **Deployment**       | `predict.py`, `risk_score.py`, `app.py`                        | How do we use the trained model on a new customer? |

### One architectural change from our earlier files

Before we continue generating files, we should **rewrite `preprocessing.py`, `predict.py`, and the related configuration around a single `ColumnTransformer`/`Pipeline`**. That's the correct way to handle your German Credit dataset's numerical and categorical variables and avoids the mismatch created by the earlier all-column `StandardScaler` approach.

So the next actual code file should be the **final corrected `src/config.py`**, followed by the corrected `preprocessing.py`. This will give us a stable foundation for the remaining files rather than stacking more code on top of an approach we already know needs changing.
