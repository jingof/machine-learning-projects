# src/predict.py

import pandas as pd
import joblib
from pathlib import Path

# --- Load Models and Explainer ---
BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "models"

BASELINE_MODEL = joblib.load(MODEL_PATH / "baseline_model.joblib")
MITIGATED_MODEL = joblib.load(MODEL_PATH / "mitigated_model.joblib")
SHAP_EXPLAINER = joblib.load(MODEL_PATH / "shap_explainer.joblib")
MODEL_COLUMNS = ['credit_score', 'annual_income', 'loan_amount', 'loan_term_months', 'income_to_loan_ratio']

def make_prediction(input_data, use_mitigated_model=False):
    """
    Makes a prediction for a single data point.

    Args:
        input_data (dict): A dictionary with applicant data.
        use_mitigated_model (bool): Flag to use the fairness-aware model.

    Returns:
        A dictionary containing the prediction and SHAP values.
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Feature Engineering (must match pipeline.py)
    df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']

    # Ensure columns are in the correct order
    df = df[MODEL_COLUMNS]

    # Select model
    model = MITIGATED_MODEL if use_mitigated_model else BASELINE_MODEL

    # Make prediction
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]

    # Get SHAP values
    shap_values = SHAP_EXPLAINER.shap_values(df)

    return {
        "prediction": int(prediction),
        "prediction_label": "Default" if prediction == 1 else "Paid Off",
        "confidence_score": float(prediction_proba[prediction]),
        "shap_values": shap_values[0].tolist(),
        "base_value": SHAP_EXPLAINER.expected_value,
        "feature_names": MODEL_COLUMNS,
    }