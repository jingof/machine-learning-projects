import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def train_fraud(df: pd.DataFrame, config: dict, artifacts_dir: str):
    print(f"-- Training fraud model.")
    y = df[config["target"]].astype(int)
    drop_cols = config["features"]["drop"]
    X = df.drop(columns=drop_cols + [config["target"]], errors="ignore")
    cat_cols = [c for c in X.columns if X[c].dtype=="object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", "passthrough", num_cols)])
    model = LogisticRegression(**config["model"]["params"])
    pipe = Pipeline([("pre", pre), ("clf", model)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte); proba = pipe.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba); report = classification_report(yte, preds, output_dict=True)
    joblib.dump(pipe, f"{artifacts_dir}/fraud_model.joblib")
    pd.DataFrame(report).to_csv(f"{artifacts_dir}/fraud_classification_report.csv")
    with open(f"{artifacts_dir}/fraud_auc.txt","w") as f: f.write(str(auc))
    print(f"- Done with training fraud model.")
    return {"auc": auc, "report": report}

def score_fraud(df: pd.DataFrame, artifacts_dir: str):
    print(f"-- Training score model.")
    model = joblib.load(f"{artifacts_dir}/fraud_model.joblib")
    drop_cols = ["transaction_id","account_id","customer_id","timestamp","merchant_category","device_id","ip_address","is_fraud"]
    X = df.drop(columns=drop_cols, errors="ignore")
    df = df.copy()
    df["fraud_score"] = model.predict_proba(X)[:,1]
    df.sort_values("fraud_score", ascending=False, inplace=True)
    df.head(500).to_csv(f"{artifacts_dir}/fraud_top500_scores.csv", index=False)
    print(f"- Done with training score model.")
    return df
