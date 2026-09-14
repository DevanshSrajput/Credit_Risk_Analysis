import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config import MODEL_FILE, RANDOM_STATE


def build_models():
    models = {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=100
        ),
    }
    return models


def train_single_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def train_all_models(models, X_train, y_train):
    trained_models = {}
    print("Training models...")
    print("=" * 60)
    for name, model in models.items():
        print(f"Training {name}...")
        trained_model = train_single_model(model, X_train, y_train)
        trained_models[name] = trained_model
        print(f"{name} trained successfully.")
    print("=" * 60)
    return trained_models


def save_model(model):
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
