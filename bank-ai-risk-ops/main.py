import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse, os
from pathlib import Path
import pandas as pd
from src.bankai.utils.io import load_yaml, ensure_dir
from src.bankai.data.synth_data import generate_all
from src.bankai.features.transactions import add_tx_time_features
from src.bankai.models.fraud import train_fraud, score_fraud
from src.bankai.models.credit import train_credit
from src.bankai.models.complaints import train_complaints

def load_paths():
    cfg = load_yaml("configs/paths.yaml")
    data_dir = ensure_dir(cfg.get("data_dir","data"))
    artifacts_dir = ensure_dir(cfg.get("artifacts_dir","artifacts"))
    seed = cfg.get("seed", 42)
    return {"data_dir": data_dir, "artifacts_dir": artifacts_dir, "seed": seed}

def cmd_generate(args):
    paths = load_paths()
    print("Generating synthetic data...")
    generate_all(paths, seed=paths["seed"], n_customers=args.customers, days=args.days)
    print("Done. Files saved to:", paths["data_dir"])

def _load_tx(paths):
    df = pd.read_csv(Path(paths["data_dir"])/"transactions.csv", parse_dates=["timestamp"])
    df = add_tx_time_features(df)
    return df

def cmd_train_fraud(_args):
    paths = load_paths()
    df = _load_tx(paths)
    cfg = load_yaml("configs/fraud.yaml")
    print("Training fraud model...")
    metrics = train_fraud(df, cfg, paths["artifacts_dir"])
    print("Fraud metrics:", metrics)

def cmd_score_fraud(_args):
    paths = load_paths()
    df = _load_tx(paths)
    print("Scoring latest transactions...")
    score_fraud(df, paths["artifacts_dir"])
    print("Top scores saved to artifacts/fraud_top500_scores.csv")

def cmd_train_credit(_args):
    paths = load_paths()
    loans = pd.read_csv(Path(paths["data_dir"])/"loans.csv")
    cfg = load_yaml("configs/credit.yaml")
    print("Training credit models...")
    metrics = train_credit(loans, cfg, paths["artifacts_dir"])
    print("Credit metrics:", metrics)

def cmd_train_complaints(_args):
    paths = load_paths()
    df = pd.read_csv(Path(paths["data_dir"])/"complaints.csv")
    cfg = load_yaml("configs/complaints.yaml")
    print("Training complaint classifier...")
    metrics = train_complaints(df, cfg, paths["artifacts_dir"])
    print("Complaint metrics:", metrics)

def cmd_run_dashboard(_args):
    os.system("streamlit run dashboards/app.py")

def make_parser():
    p = argparse.ArgumentParser(description="BankAI RiskOps Suite CLI")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-data", help="Generate synthetic datasets")
    g.add_argument("--customers", type=int, default=6000)
    g.add_argument("--days", type=int, default=1200)
    g.set_defaults(func=cmd_generate)
    sub.add_parser("train-fraud", help="Train fraud model").set_defaults(func=cmd_train_fraud)
    sub.add_parser("score-fraud", help="Score fraud using latest model").set_defaults(func=cmd_score_fraud)
    sub.add_parser("train-credit", help="Train credit PD/LGD").set_defaults(func=cmd_train_credit)
    sub.add_parser("train-complaints", help="Train complaint classifier").set_defaults(func=cmd_train_complaints)
    sub.add_parser("dashboard", help="Launch Streamlit dashboard").set_defaults(func=cmd_run_dashboard)
    return p

def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
