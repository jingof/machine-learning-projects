import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

def train_credit(loans: pd.DataFrame, config: dict, artifacts_dir: str):
    print(f"-- Training credit model.")
    pd_y = loans[config["pd_target"]].astype(int)
    pd_X = loans.drop(columns=[config["pd_target"], config["lgd_target"], "loan_id","customer_id"], errors="ignore")
    Xtr, Xte, ytr, yte = train_test_split(pd_X, pd_y, test_size=config["test_size"], random_state=config["random_state"], stratify=pd_y)
    pd_model = LogisticRegression(**config["pd_model"]["params"]).fit(Xtr, ytr)
    pd_proba = pd_model.predict_proba(Xte)[:,1]
    pd_auc = roc_auc_score(yte, pd_proba)
    joblib.dump(pd_model, f"{artifacts_dir}/credit_pd_model.joblib")
    lgd_df = loans[loans[config["pd_target"]]==1].copy()
    if len(lgd_df) > 10:
        lgd_y = lgd_df[config["lgd_target"]]
        lgd_X = lgd_df.drop(columns=[config["pd_target"], config["lgd_target"], "loan_id","customer_id"], errors="ignore")
        Xtr2, Xte2, ytr2, yte2 = train_test_split(lgd_X, lgd_y, test_size=config["test_size"], random_state=config["random_state"])
        lgd_model = LinearRegression(**config["lgd_model"]["params"]).fit(Xtr2, ytr2)
        preds = lgd_model.predict(Xte2).clip(0,1)
        lgd_mae = mean_absolute_error(yte2, preds); lgd_r2 = r2_score(yte2, preds)
        joblib.dump(lgd_model, f"{artifacts_dir}/credit_lgd_model.joblib")
    else:
        lgd_mae, lgd_r2 = None, None
    with open(f"{artifacts_dir}/credit_metrics.txt","w") as f:
        f.write(f"PD AUC: {pd_auc}\nLGD MAE: {lgd_mae}\nLGD R2: {lgd_r2}\n")
    print(f"- Done with training credit model.")
    return {"pd_auc": pd_auc, "lgd_mae": lgd_mae, "lgd_r2": lgd_r2}
