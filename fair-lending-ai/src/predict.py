# src/predict.py

import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# --- Load Models and Explainer ---
BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "models"

BASELINE_MODEL = joblib.load(MODEL_PATH / "baseline_model.joblib")
MITIGATED_MODEL = joblib.load(MODEL_PATH / "mitigated_model.joblib")
SHAP_EXPLAINER = joblib.load(MODEL_PATH / "shap_explainer.joblib")
MODEL_COLUMNS = joblib.load(MODEL_PATH / "data_features.joblib")


def make_prediction(input_data, use_mitigated_model=False):
    """
    Makes a prediction for a single data point.

    Args:
        input_data (dict): A dictionary with applicant data.
        use_mitigated_model (bool): Flag to use the fairness-aware model.

    Returns:
        A dictionary containing the prediction and SHAP values.
    """

    # Ensure columns are in the correct order
    df = predict_pipeline(input_data)
    df = df[MODEL_COLUMNS]
    # Select model
    model = MITIGATED_MODEL if use_mitigated_model else BASELINE_MODEL

    # Make prediction
    prediction = model.predict(df)[0]
    if not use_mitigated_model:
        prediction_proba = model.predict_proba(df)[0]
    else:
        # ExponentiatedGradient trains an ensemble of classifiers to satisfy fairness constraints.
        # After fitting, it doesn’t return a single model 
        prediction_proba = get_mitigated_model_probs(model, df)
    # Get SHAP values
    shap_values = SHAP_EXPLAINER.shap_values(df)
    return {
        "prediction": int(prediction),
        "prediction_label": "Default" if prediction == 1 else "Paid Off",
        "confidence_score": float(prediction_proba[prediction]),
        "shap_values": shap_values[0],
        "base_value": SHAP_EXPLAINER.expected_value,
        "feature_names": df.columns.to_list(),
    }


def predict_pipeline(input_data):
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    # Feature Engineering (must match pipeline.py)
    df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']
    df['race_GroupA'] = 0
    df['race_GroupB'] = 0
    df['race_GroupC'] = 0
    df['race_GroupD'] = 0
    race_group = input_data['race_group'][-1]
    df[f'race_Group{race_group}'] = 1
    df = df.rename(columns={"race_GroupB": "race_groupb"})
    df.drop(['race_group'], axis=1, inplace=True)
    return df


def get_mitigated_model_probs(model, df):
    probs = np.zeros((df.shape[0], 2))
    for predictor, weight in zip(model.predictors_, model.weights_):
        probs += weight * predictor.predict_proba(df)
    # final probability estimates
    probs /= np.sum(model.weights_)
    return probs.flatten()
