# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

def _naive_seasonal(y: pd.Series, horizon: int = 26, season: int = 52):
    if len(y) < season + 1:
        level = np.mean(y[-8:]) if len(y) >= 8 else np.mean(y)
        yhat = np.full(horizon, level)
        sd = np.std(y[-8:]) if len(y) >= 8 else level * 0.2
        return yhat, yhat - sd, yhat + sd
    template = y[-season:].values
    seq = np.tile(template, int(np.ceil(horizon/season)))[:horizon]
    return seq, seq * 0.9, seq * 1.1

def fit_forecast(y: pd.Series, horizon: int = 26, seasonal_order=(0,1,1,52)):
    y = y.astype(float).fillna(0.0)
    model_tag = "sarimax"
    try:
        if SARIMAX is None:
            raise ImportError("statsmodels SARIMAX not available")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(y, order=(0,1,1),
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=200)
            fc = res.get_forecast(steps=horizon)
            yhat = fc.predicted_mean.values
            conf = fc.conf_int(alpha=0.2).values
            lo, hi = conf[:,0], conf[:,1]
            if np.isnan(yhat).any():
                raise RuntimeError("nan forecast")
    except Exception:
        model_tag = "naive_seasonal"
        yhat, lo, hi = _naive_seasonal(y, horizon=horizon, season=seasonal_order[-1])
    return {"yhat": pd.Series(yhat), "yhat_lower": pd.Series(lo), "yhat_upper": pd.Series(hi), "model": model_tag}
