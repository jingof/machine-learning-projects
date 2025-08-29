# src/train.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import xgboost as xgb
import shap
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from src.fairness import get_fairness_metrics

def train_and_evaluate():
    """
    Trains, evaluates, and saves baseline and bias-mitigated models.
    """
    print("Starting model training and evaluation...")

    # --- Setup Paths and Load Data ---
    base_path = Path(__file__).parent.parent
    processed_data_path = base_path / "data/processed/processed_loan_data.parquet"
    model_path = base_path / "models/"
    model_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_data_path)
    # --- Data Splitting ---
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    sensitive_features = X['race_groupb'] # Using the one-hot encoded column
    X = X.drop('race_groupb', axis=1) # Drop sensitive feature from main features

    X_train, X_test, y_train, y_test, sensitive_features_train, sensitive_features_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y
    )
    print(X.columns)
    joblib.dump(X.columns, model_path / "data_features.joblib")
    print(f"Data split into {len(X_train)} training and {len(X_test)} test samples.")

    # --- 1. Baseline Model (XGBoost) ---
    print("\n--- Training Baseline XGBoost Model ---")
    baseline_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    
    # Evaluate Baseline Model
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline)
    baseline_fairness = get_fairness_metrics(y_test, y_pred_baseline, sensitive_features_test)
    
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline F1-Score: {baseline_f1:.4f}")
    # print(f"Baseline Demographic Parity Difference: {baseline_fairness['difference']['demographic_parity_difference']:.4f}")
    print(f"Baseline Demographic Parity Difference: {baseline_fairness['difference']:.4f}")

    # --- 2. Bias Mitigated Model (Fairlearn) ---
    print("\n--- Training Bias Mitigated Model (ExponentiatedGradient) ---")
    mitigator = ExponentiatedGradient(
        estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        constraints=DemographicParity()
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    y_pred_mitigated = mitigator.predict(X_test)

    # Evaluate Mitigated Model
    mitigated_accuracy = accuracy_score(y_test, y_pred_mitigated)
    mitigated_f1 = f1_score(y_test, y_pred_mitigated)
    mitigated_fairness = get_fairness_metrics(y_test, y_pred_mitigated, sensitive_features_test)

    print(f"Mitigated Accuracy: {mitigated_accuracy:.4f}")
    print(f"Mitigated F1-Score: {mitigated_f1:.4f}")
    # print(f"Mitigated Demographic Parity Difference: {mitigated_fairness['difference']['demographic_parity_difference']:.4f}")
    print(f"Mitigated Demographic Parity Difference: {mitigated_fairness['difference']:.4f}")

    # --- Save Artifacts ---
    print("\n--- Saving Models and Evaluation Results ---")
    # Save models
    joblib.dump(baseline_model, model_path / "baseline_model.joblib")
    joblib.dump(mitigator, model_path / "mitigated_model.joblib")
    print("Models saved.")

    # Save SHAP explainer
    explainer = shap.TreeExplainer(baseline_model)
    joblib.dump(explainer, model_path / "shap_explainer.joblib")
    print("SHAP explainer saved.")

    # Save evaluation results for the app
    results = {
        "baseline": {
            "accuracy": baseline_accuracy,
            "f1_score": baseline_f1,
            # "demographic_parity_difference": baseline_fairness['difference']['demographic_parity_difference'],
            "demographic_parity_difference": baseline_fairness['difference'],
            "approval_rates_by_group": baseline_fairness['by_group'].to_dict()
        },
        "mitigated": {
            "accuracy": mitigated_accuracy,
            "f1_score": mitigated_f1,
            # "demographic_parity_difference": mitigated_fairness['difference']['demographic_parity_difference'],
            "demographic_parity_difference": baseline_fairness['difference'],
            "approval_rates_by_group": mitigated_fairness['by_group'].to_dict()
        }
    }
    with open(model_path / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    print("Evaluation results saved.")


if __name__ == "__main__":
    train_and_evaluate()