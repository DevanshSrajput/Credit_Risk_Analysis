# Know.md - Project Decisions & Rationale

Every architectural and implementation decision in this project, why it was made, and what alternatives were considered.

---

## 1. Why the German Credit Dataset?

**Decision:** Use the UCI German Credit dataset (1,000 samples, 20 features).

**Why:**
- Small enough to train quickly and debug easily
- Well-documented with clear feature definitions
- Contains both numerical and categorical features (good for teaching preprocessing)
- Real-world problem that students can relate to
- Imbalanced classes (70/30) which is typical in credit risk

**Alternatives considered:**
- Kaggle Home Credit Default Risk: Too large (300K+ rows), too many tables
- Lending Club: Requires API access, data changes over time
- Synthetic data: Doesn't teach real-world data quirks

---

## 2. Why Integer-Encoded Categorical Features?

**Decision:** Keep the original integer encoding for categorical features rather than converting to strings.

**Why:**
- The dataset is designed this way (UCI standard)
- All features are already integers, making it easy to overlook that they're categorical
- Teaches an important lesson: **integer encoding doesn't mean numerical relationship**
- A value of `2` for `laufkont` isn't "more" than `1` — it's a different category

**The trap:** Beginners often treat all integer columns as numerical. This project explicitly addresses this by separating `NUMERICAL_FEATURES` (laufzeit, hoehe, alter) from `CATEGORICAL_FEATURES` (everything else with low cardinality).

---

## 3. Why ColumnTransformer Instead of Separate Scaling/Encoding?

**Decision:** Use sklearn's `ColumnTransformer` with `StandardScaler` for numerical and `OneHotEncoder` for categorical features.

**Why:**
- Single object that handles all transformations
- Prevents data leakage (fit on train, transform on test)
- Saves/loads as one pickle file for inference
- sklearn's recommended approach

**The wrong approach (which we initially had):**
```python
# BAD: Scales everything including categorical codes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
This treats categorical codes as if they have meaningful numerical relationships, which they don't.

**The correct approach:**
```python
ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features)
])
```

---

## 4. Why `drop="first"` in OneHotEncoder?

**Decision:** Use `drop="first"` to avoid the dummy variable trap.

**Why:**
- With N categories, N-1 dummy variables are sufficient
- Prevents multicollinearity (perfect correlation between dummy variables)
- Reduces dimensionality without losing information
- Standard practice for linear models like Logistic Regression

**Trade-off:** Decision Trees and Random Forests can handle all N dummies, but using N-1 works for all models.

---

## 5. Why Feature Engineering Before Preprocessing?

**Decision:** Create `Credit_per_Month`, `Longterm_Loan`, and `Age_Group` before the ColumnTransformer.

**Why:**
- Engineered features need to be part of the same preprocessing pipeline
- `Credit_per_Month` is numerical → goes through StandardScaler
- `Longterm_Loan` is binary categorical → goes through OneHotEncoder
- Feature engineering must happen at both training AND inference time

**The consistency problem:**
If feature engineering happens in `main.py` but not in `predict.py`, the model receives different feature sets at training vs inference. By doing it inside the preprocessing flow, we ensure consistency.

**Inference path:**
```python
# predict.py
customer_data = feature_engineering_pipeline(customer_data)  # Must create features
preprocessed = preprocessor.transform(customer_data)          # Then transform
prediction = model.predict(preprocessed)
```

---

## 6. Why `handle_unknown="ignore"` in OneHotEncoder?

**Decision:** Set `handle_unknown="ignore"` for categorical encoding.

**Why:**
- At inference time, a new customer might have a category not seen during training
- Without this, the encoder would crash
- With `"ignore"`, unknown categories get all-zero encoding
- Safe fallback that doesn't break the pipeline

---

## 7. Why Class 0 is "Bad Credit" and Class 1 is "Good Credit"?

**Decision:** Map `kredit=0` → Bad Credit, `kredit=1` → Good Credit.

**Why:**
- The dataset documentation specifies this mapping
- 30% have `kredit=0` (bad credit) — typical default rate
- 70% have `kredit=1` (good credit) — majority class
- We confirmed by checking `model.classes_` which returns `[0, 1]`

**Critical lesson:** Always verify which class is which before interpreting probabilities. The model's `predict_proba(X)[:, 0]` gives the probability of class 0 (bad credit), which is what we want for risk scoring.

---

## 8. Why ROC-AUC Uses Negated Probability?

**Decision:** Pass `-probability` to `roc_auc_score()` when probability is for the bad credit class.

**Why:**
- `roc_auc_score(y_true, y_score)` expects higher `y_score` to indicate the positive class
- We want to measure how well the model identifies BAD credit (class 0)
- `predict_proba(X)[:, 0]` gives P(bad credit) — higher = worse
- But `roc_auc_score` assumes higher = more likely positive
- Negating makes higher values correspond to GOOD credit, which aligns with the metric's expectation

**Alternative considered:** Relabel the target (0→1, 1→0) — but this adds complexity and potential for errors.

---

## 9. Why Precision/Recall/F1 Use `pos_label=0`?

**Decision:** Set `pos_label=BAD_CREDIT_CLASS` (0) in precision, recall, and F1 calculations.

**Why:**
- Default `pos_label=1` would measure performance for GOOD credit
- We care about detecting BAD credit (defaults)
- A bank wants to catch as many potential defaults as possible (high recall for bad credit)
- Precision for bad credit tells us: "Of those we flagged as risky, how many actually defaulted?"

**Business impact:**
- High recall for bad credit = fewer missed defaults (good for bank)
- High precision for bad credit = fewer false alarms (good for customer experience)

---

## 10. Why Logistic Regression Was Chosen as Best Model?

**Decision:** Select Logistic Regression based on highest ROC-AUC (0.8274).

**Why:**
- ROC-AUC is the most important metric for imbalanced datasets
- It measures discrimination ability across all thresholds
- Logistic Regression has the highest ROC-AUC among the three models
- It's also the most interpretable (coefficients show feature importance)

**Counter-arguments:**
- Random Forest has higher recall for class 1 (0.94) — but we care about class 0
- Decision Tree is most interpretable — but has lowest accuracy
- XGBoost might perform better — but wasn't included in initial scope

---

## 11. Why Risk Thresholds are 30/70?

**Decision:** Low Risk < 30, Medium Risk 30-69, High Risk >= 70.

**Why:**
- These are educational thresholds, not actual banking policy
- They provide clear separation between categories
- A score of 30 means 30% probability of default — that's significant
- A score of 70 means 70% probability of default — very high risk

**Real-world context:**
- Actual banks use more complex scoring systems
- Regulatory requirements (Basel III) mandate specific risk classification
- These thresholds are simplified for teaching purposes

---

## 12. Why `src/` Package Structure?

**Decision:** Move all source modules into `src/` with proper Python package imports.

**Why:**
- Clean separation between source code and project root
- Imports use `from src.config import ...` — explicit and unambiguous
- Prevents accidental imports from the wrong directory
- Standard Python project structure
- `src/__init__.py` makes it a proper package

**Alternatives considered:**
- Flat structure (all .py in root): Simpler but messy as project grows
- `lib/` directory: Less common in ML projects
- `credit_risk/` named package: Too specific, harder to reuse

---

## 13. Why Separate `data_loader.py` from `preprocessing.py`?

**Decision:** Data loading and preprocessing are separate modules.

**Why:**
- Single Responsibility Principle
- `data_loader.py`: Only handles reading CSV and basic inspection
- `preprocessing.py`: Handles duplicates, feature engineering, splitting, encoding
- Can test data loading independently
- Can swap data sources without changing preprocessing

---

## 14. Why Save the Preprocessor as a Separate File?

**Decision:** Save `preprocessor.pkl` alongside `credit_risk_model.pkl`.

**Why:**
- The model expects preprocessed input
- The same transformations must be applied at inference time
- Can't rely on the application remembering the exact training transformations
- One file per artifact = cleaner deployment

**Inference flow:**
```python
model = joblib.load("credit_risk_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Same transformations as training
X = preprocessor.transform(new_data)
prediction = model.predict(X)
```

---

## 15. Why Streamlit Over Flask/FastAPI?

**Decision:** Use Streamlit for the web application.

**Why:**
- Zero frontend code needed (pure Python)
- Built-in widgets (sliders, dropdowns, buttons)
- Automatic layout and styling
- `st.cache_resource` for model loading
- Perfect for ML demos and Prototypes

**Trade-offs:**
- Less customizable than Flask/FastAPI
- Not suitable for high-traffic production
- But ideal for a teaching project

---

## 16. Why No Cross-Validation?

**Decision:** Use a single train/test split instead of k-fold cross-validation.

**Why:**
- Dataset is small (1,000 samples) — CV would leave very small validation sets
- Single split is simpler to understand for students
- 80/20 split with stratification gives reasonable evaluation
- Results are more reproducible (single random state)

**Future improvement:** Add StratifiedKFold for more robust evaluation.

---

## 17. Why `skipinitialspace=True` in CSV Reading?

**Decision:** Set `skipinitialspace=True` when reading the CSV.

**Why:**
- The German Credit CSV has spaces after delimiters
- Without this, column names become `" laufzeit"` instead of `"laufzeit"`
- Also applied `data.columns = data.columns.str.strip()` as defense-in-depth
- Prevents subtle bugs where column names don't match

---

## 18. Why `stratify=y` in Train/Test Split?

**Decision:** Use stratified splitting to maintain class proportions.

**Why:**
- Dataset is imbalanced (70/30)
- Random split might create unrepresentative subsets
- Stratification ensures both train and test have ~70% good / 30% bad credit
- More reliable evaluation metrics

---

## 19. Why `RANDOM_STATE = 42`?

**Decision:** Use 42 as the random state everywhere.

**Why:**
- Ensures reproducibility
- Same train/test split every run
- Same model initialization every run
- Students can verify results match
- 42 is the "Answer to the Ultimate Question of Life, the Universe, and Everything" (Hitchhiker's Guide)

---

## 20. Why Feature Scaling for Logistic Regression but Not Decision Trees?

**Decision:** Apply StandardScaler to numerical features for all models.

**Why:**
- Logistic Regression is sensitive to feature scales (gradient descent convergence)
- Decision Trees and Random Forests are scale-invariant
- But using the same preprocessing for all models:
  - Simplifies the pipeline
  - Prevents confusion when swapping models
  - Doesn't hurt tree-based models
  - Required for fair comparison

---

## 21. Why `max_iter=1000` for Logistic Regression?

**Decision:** Set maximum iterations to 1000.

**Why:**
- Default (100) might not converge with 20+ features after one-hot encoding
- The dataset has 20 original features → ~40+ after encoding
- More iterations ensure convergence
- Warning-free output

---

## 22. Why n_estimators=100 for Random Forest?

**Decision:** Use 100 trees in the Random Forest.

**Why:**
- Default in sklearn
- Good balance between performance and training time
- More trees (500+) give marginal improvement
- 100 is sufficient for this dataset size

---

## 23. Why the Evaluation Reports are Generated as Files?

**Decision:** Save `model_comparison.csv` and `model_report.txt` to disk.

**Why:**
- Students can include them in project reports
- Easy to compare experiments over time
- Machine-readable (CSV) and human-readable (TXT)
- Part of the reproducibility workflow

---

## 24. Why `Age_Group` is Categorical Not Numerical?

**Decision:** Create `Age_Group` as a categorical feature (pd.cut with labels).

**Why:**
- Age has a non-linear relationship with credit risk
- Grouping captures this: Young (high risk), Adult (low risk), Senior (medium risk)
- As categorical, it gets one-hot encoded
- Alternative: Polynomial features — but more complex and harder to interpret

---

## 25. Why We Don't Use All Engineered Features in EDA?

**Decision:** EDA runs on original features only (laufzeit, hoehe, alter), not engineered ones.

**Why:**
- EDA should understand the RAW data before any transformations
- Engineered features are derived and would show derived relationships
- Clean separation: EDA on raw data → Feature engineering → Preprocessing → Training

---

## Summary of Key Principles

1. **Data Leakage Prevention:** Never fit preprocessing on test data
2. **Consistency:** Same transformations at training and inference
3. **Reproducibility:** Fixed random states, saved artifacts
4. **Modularity:** Each file has one responsibility
5. **Documentation:** Every decision is traceable
6. **Simplicity:** Start simple, add complexity only when needed
