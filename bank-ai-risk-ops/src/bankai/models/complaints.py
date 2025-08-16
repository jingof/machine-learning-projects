import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_complaints(df: pd.DataFrame, config: dict, artifacts_dir: str):
    print(f"-- Training complaints model.")
    y = df[config["target"]]; X = df[config["text_col"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y)
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)), ("clf", LogisticRegression(max_iter=600))])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte); acc = accuracy_score(yte, preds)
    rep = classification_report(yte, preds, output_dict=True)
    joblib.dump(pipe, f"{artifacts_dir}/complaint_model.joblib")
    pd.DataFrame(rep).to_csv(f"{artifacts_dir}/complaints_report.csv")
    with open(f"{artifacts_dir}/complaints_acc.txt","w") as f: f.write(str(acc))
    print(f"- Done with training complaints model.")
    return {"accuracy": acc, "report": rep}
