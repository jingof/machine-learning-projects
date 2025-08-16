import pandas as pd, numpy as np
def add_tx_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"]>=5).astype(int)
    df["amount_log"] = np.log1p(df["amount"])
    df = df.sort_values(["account_id","timestamp"])
    df["ts_unix"] = pd.to_datetime(df["timestamp"]).astype("int64")//10**9
    df["time_since_prev"] = df.groupby("account_id")["ts_unix"].diff().fillna(0)
    df.drop(columns=["ts_unix"], inplace=True)
    return df
