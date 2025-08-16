import numpy as np, pandas as pd
def psi(expected: pd.Series, actual: pd.Series, buckets:int=10)->float:
    expected = expected.replace([np.inf,-np.inf], np.nan).dropna()
    actual = actual.replace([np.inf,-np.inf], np.nan).dropna()
    cuts = np.quantile(expected, np.linspace(0,1,buckets+1)[1:-1])
    e_counts = np.histogram(expected, bins=np.concatenate(([-np.inf],cuts,[np.inf])))[0]
    a_counts = np.histogram(actual, bins=np.concatenate(([-np.inf],cuts,[np.inf])))[0]
    e_perc = (e_counts/e_counts.sum()).clip(1e-6,1); a_perc = (a_counts/a_counts.sum()).clip(1e-6,1)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))
def ks(expected: pd.Series, actual: pd.Series)->float:
    try:
        from scipy.stats import ks_2samp
        d,_ = ks_2samp(expected.dropna(), actual.dropna()); return float(d)
    except Exception:
        x=np.sort(expected.dropna()); y=np.sort(actual.dropna()); xi=yi=0; d=0.0
        while xi<len(x) and yi<len(y):
            if x[xi]<=y[yi]: xi+=1
            else: yi+=1
            d=max(d, abs(xi/len(x)-yi/len(y)))
        return float(d)
