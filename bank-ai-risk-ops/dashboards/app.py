import streamlit as st, pandas as pd, numpy as np, joblib
from pathlib import Path
st.set_page_config(page_title="BankAI RiskOps", layout="wide")
st.title("🏦 BankAI RiskOps Dashboard")
data_dir = Path("data"); artifacts_dir = Path("artifacts")
st.sidebar.header("Data")
if not data_dir.exists():
    st.warning("No data directory found. Run `python main.py generate-data`.")
else:
    tx = pd.read_csv(data_dir / "transactions.csv", parse_dates=["timestamp"])
    st.metric("Transactions", len(tx))
    st.metric("Fraud rate", f"{tx['is_fraud'].mean()*100:.2f}%")
    st.dataframe(tx.sample(min(100, len(tx))))
st.sidebar.header("Fraud Scoring")
model_path = artifacts_dir / "fraud_model.joblib"
if model_path.exists():
    model = joblib.load(model_path)
    st.success("Fraud model loaded.")
    st.subheader("Score a transaction")
    amount = st.number_input("Amount", value=120.0, step=10.0)
    hour = st.slider("Hour", 0, 23, 13)
    dayofweek = st.slider("Day of week (0=Mon)", 0, 6, 2)
    is_weekend = 1 if dayofweek>=5 else 0
    time_since_prev = st.number_input("Time since prev (s)", value=3600)
    amount_log = np.log1p(amount)
    X = pd.DataFrame([{"amount":amount,"hour":hour,"dayofweek":dayofweek,"is_weekend":is_weekend,"time_since_prev":time_since_prev,"amount_log":amount_log}])
    score = model.predict_proba(X)[0,1]
    st.metric("Fraud score", f"{score:.3f}")
else:
    st.info("Train the fraud model first.")
