# app/main.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Import the prediction function from our source code
from src.predict import make_prediction, predict_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Risk & Fairness Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
except:
    pass


# --- Load Evaluation Results ---
@st.cache_data
def load_results():
    results_path = Path(__file__).parent.parent / "models/evaluation_results.json"
    with open(results_path, 'r') as f:
        return json.load(f)

results = load_results()
baseline_results = results['baseline']
mitigated_results = results['mitigated']

# --- Sidebar ---
st.sidebar.title("Applicant Information")
st.sidebar.header("Enter Applicant Details:")

credit_score = st.sidebar.slider("Credit Score", 500, 850, 650)
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=15000, max_value=500000, value=60000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=5000, max_value=100000, value=20000, step=500)
loan_term_months = st.sidebar.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72], index=1)
race_group = st.sidebar.selectbox("Race", ['GroupA', 'GroupB', 'GroupC', 'GroupD'], index=1)

st.sidebar.header("Model Selection")
use_mitigated = st.sidebar.checkbox("Use Bias Mitigated Model", value=True)
model_name = "Bias Mitigated Model" if use_mitigated else "Baseline Model"

# --- Main Page ---
st.title("Credit Risk & Fairness Analysis Dashboard")
st.write(f"Showing results for: **{model_name}**")
st.markdown("---")

# --- Prediction Section ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Loan Prediction")
    input_data = {
        'credit_score': credit_score,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'loan_term_months': loan_term_months,
        'race_group': race_group
    }
    
    if st.button("Get Prediction", type="primary"):
        prediction_output = make_prediction(input_data, use_mitigated_model=use_mitigated)
        
        pred_label = prediction_output['prediction_label']
        confidence = prediction_output['confidence_score']
        
        if pred_label == "Default":
            st.error(f"Prediction: **{pred_label}**")
        else:
            st.success(f"Prediction: **{pred_label}**")
        
        st.metric(label="Confidence Score", value=f"{confidence:.2%}")

        # Store prediction output in session state to display SHAP plot
        st.session_state['prediction_output'] = prediction_output
    
with col2:
    st.header("Prediction Explanation (SHAP)")
    if 'prediction_output' in st.session_state:
        pred_out = st.session_state['prediction_output']
        
        # Create a SHAP explanation object for the plot
        shap_explanation = shap.Explanation(
            values=pred_out['shap_values'],
            base_values=pred_out['base_value'],
            data=predict_pipeline(input_data)[pred_out['feature_names']],
            feature_names=pred_out['feature_names']
        )
        # Create SHAP Force Plot
        fig = shap.force_plot(
            base_value=shap_explanation.base_values,
            shap_values=shap_explanation.values,
            features=shap_explanation.data,
            feature_names=shap_explanation.feature_names,
            matplotlib=True,
            show=False
        )
        st.pyplot(fig, bbox_inches='tight')
        st.info("The plot above shows features pushing the prediction higher (red) or lower (blue). A higher model output corresponds to a higher risk of default.")
    else:
        st.write("Click 'Get Prediction' to see the explanation.")

st.markdown("---")

# --- Model Performance & Fairness Section ---
st.header("Overall Model Performance & Fairness")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Baseline Model")
    st.metric("Accuracy", f"{baseline_results['accuracy']:.2%}")
    st.metric("F1-Score", f"{baseline_results['f1_score']:.3f}")
    st.metric("Demographic Parity Diff.", f"{baseline_results['demographic_parity_difference']:.3f}", 
              help="Difference in approval rates between groups. Closer to 0 is fairer.")

with c2:
    st.subheader("Bias Mitigated Model")
    st.metric("Accuracy", f"{mitigated_results['accuracy']:.2%}")
    st.metric("F1-Score", f"{mitigated_results['f1_score']:.3f}")
    st.metric("Demographic Parity Diff.", f"{mitigated_results['demographic_parity_difference']:.3f}",
              help="Difference in approval rates between groups. Closer to 0 is fairer.")

with c3:
    st.subheader("Comparison")
    acc_change = mitigated_results['accuracy'] - baseline_results['accuracy']
    fairness_change = mitigated_results['demographic_parity_difference'] - baseline_results['demographic_parity_difference']
    st.metric("Accuracy Change", f"{acc_change:.2%}")
    st.metric("Fairness Improvement", f"{abs(fairness_change):.3f}")
