import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR, MODEL_FILE, MODEL_METADATA_FILE, REPORTS_DIR
from src.train_model import save_model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }
    return predictions, metrics


def plot_confusion_matrix(y_test, predictions, model_name, output_dir):
    matrix = sk_confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()


def plot_roc_curve(model, X_test, y_test, model_name, output_dir):
    from sklearn.metrics import RocCurveDisplay

    display = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - ROC Curve")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
    plt.close()


def save_model_report(best_model_name, best_metrics):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = REPORTS_DIR / "model_report.txt"
    with open(report_path, "w") as f:
        f.write("Credit Risk Assessment Model\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best Model: {best_model_name}\n\n")
        for metric, value in best_metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
    print(f"Model report saved to {report_path}")


def save_model_comparison(all_metrics):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.DataFrame(all_metrics).T
    csv_path = REPORTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path)
    print(f"Model comparison saved to {csv_path}")


def evaluate_all_models(trained_models, X_test, y_test):
    best_model = None
    best_score = 0
    best_name = ""
    all_metrics = {}

    print("\nModel Evaluation")
    print("=" * 60)

    for name, model in trained_models.items():
        print(f"\n--- {name} ---")
        predictions, metrics = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics

        for metric, value in metrics.items():
            print(f"  {metric:<12}: {value:.4f}")

        print("\n  Classification Report:")
        print(classification_report(y_test, predictions))

        plot_confusion_matrix(y_test, predictions, name, FIGURES_DIR)
        plot_roc_curve(model, X_test, y_test, name, FIGURES_DIR)

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model
            best_name = name

    print("=" * 60)
    print(f"\nBest Model: {best_name} (ROC-AUC: {best_score:.4f})")
    save_model(best_model)
    save_model_comparison(all_metrics)
    save_model_report(best_name, all_metrics[best_name])

    joblib.dump(
        {"model_name": best_name, "metrics": all_metrics[best_name]},
        MODEL_METADATA_FILE,
    )
    print(f"Model metadata saved to {MODEL_METADATA_FILE}")

    return best_model
