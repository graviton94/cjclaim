import pickle, numpy as np, pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io_utils import ART, write_parquet, log_jsonl

def fit_sarimax(y: pd.Series, order=(0,1,1), seasonal_order=(0,1,1,52)):
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def seasonal_naive(y: pd.Series, horizon=52):
    if len(y) < 52:
        raise ValueError("Not enough history for seasonal naive")
    fc = pd.Series(y.iloc[-52:].values, index=range(1, horizon+1))
    return fc

def save_artifacts(series: str, train_until: int, model):
    path = ART/f"models/{series}/{train_until}"
    path.mkdir(parents=True, exist_ok=True)
    with open(path/"model.pkl","wb") as f:
        pickle.dump(model, f)
    meta = {"series_id": series, "train_until": train_until, "model_ver": "sarimax_v1"}
    (path/"meta.yaml").write_text("\n".join([f"{k}: {v}" for k,v in meta.items()]), encoding="utf-8")

def load_model(series: str, train_until: int):
    with open(ART/f"models/{series}/{train_until}/model.pkl","rb") as f:
        return pickle.load(f)
