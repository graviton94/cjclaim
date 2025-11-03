# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

def _naive_seasonal(y: pd.Series, horizon: int = 26, season: int = 52, ci: float = 0.95):
    """Return naive seasonal forecast with interval width scaled to the requested CI.

    We use a normal-critical-value approximation for the interval width. If the
    series is too short, fallback to a constant level with an estimated sd.
    """
    # map common CI to z-score; fallback to normal quantile approximation when possible
    ci_to_z = {
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.96,
        0.99: 2.5758,
        0.999: 3.2905
    }
    z = ci_to_z.get(ci, None)
    if z is None:
        # simple approximation using inverse error function if scipy not available
        try:
            from math import sqrt
            # approximate normal quantile via inverse error function is not available here
            # fall back to 1.96 for safety
            z = 1.96
        except Exception:
            z = 1.96

    if len(y) < season + 1:
        level = np.mean(y[-8:]) if len(y) >= 8 else np.mean(y)
        yhat = np.full(horizon, level)
        sd = np.std(y[-8:]) if len(y) >= 8 else max(1.0, abs(level) * 0.2)
        lo = yhat - z * sd
        hi = yhat + z * sd
        return yhat, lo, hi

    template = y[-season:].values
    seq = np.tile(template, int(np.ceil(horizon/season)))[:horizon]
    # estimate seasonal variability per-segment
    sd = np.std(template) if template.size > 1 else max(1.0, np.mean(template) * 0.2)
    lo = seq - z * sd
    hi = seq + z * sd
    return seq, lo, hi

def fit_forecast(y: pd.Series, horizon: int = 26, seasonal_order=(0,1,1,52), ci: float = 0.95):
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
            alpha = 1.0 - float(ci)
            conf = fc.conf_int(alpha=alpha).values
            lo, hi = conf[:,0], conf[:,1]
            if np.isnan(yhat).any():
                raise RuntimeError("nan forecast")
    except Exception:
        model_tag = "naive_seasonal"
        yhat, lo, hi = _naive_seasonal(y, horizon=horizon, season=seasonal_order[-1], ci=ci)
    return {"yhat": pd.Series(yhat), "yhat_lower": pd.Series(lo), "yhat_upper": pd.Series(hi), "model": model_tag}


def safe_forecast(y: pd.Series, horizon: int = 26, seasonal_order=(0,1,1,52), ci: float = 0.95):
    """A wrapper around fit_forecast that guards against degenerate or extreme forecasts.

    If the underlying model produces NaNs, infinities, or predictions that are
    wildly outside the historical range (or produce absurdly wide intervals),
    this function falls back to the naive seasonal predictor and clips outputs
    to a reasonable range based on historical data.
    """
    # ensure numeric series
    y = y.astype(float).fillna(0.0)
    # historical stats
    if len(y) == 0:
        return fit_forecast(y, horizon=horizon, seasonal_order=seasonal_order)

    hist_min = float(np.min(y))
    hist_max = float(np.max(y))
    hist_mean = float(np.mean(y))
    hist_std = float(np.std(y)) if len(y) > 1 else max(1.0, abs(hist_mean) * 0.1)

    # call primary forecast with requested CI
    try:
        res = fit_forecast(y, horizon=horizon, seasonal_order=seasonal_order, ci=ci)
    except Exception:
        res = None

    def _is_bad_output(res_dict):
        if res_dict is None:
            return True
        yhat = np.asarray(res_dict.get('yhat', np.array([])), dtype=float)
        lo = np.asarray(res_dict.get('yhat_lower', np.array([])), dtype=float)
        hi = np.asarray(res_dict.get('yhat_upper', np.array([])), dtype=float)

        if yhat.size == 0:
            return True
        if np.isnan(yhat).any() or np.isinf(yhat).any():
            return True
        # extremely large magnitudes
        if np.any(np.abs(yhat) > max(1e6, hist_max * 100 + 10)):
            return True
        # absurdly wide intervals relative to historical volatility
        avg_width = float(np.mean(np.abs(hi - lo))) if (lo.size and hi.size) else 0.0
        if avg_width > max(1e6, hist_std * 1000 + 10):
            return True
        return False

    if _is_bad_output(res):
        # fallback to naive seasonal predictor using the requested CI
        naive = _naive_seasonal(y, horizon=horizon, season=seasonal_order[-1], ci=ci)
        yhat, lo, hi = naive
        model_tag = 'fallback_naive'
    else:
        yhat = np.asarray(res.get('yhat', []), dtype=float)
        lo = np.asarray(res.get('yhat_lower', []), dtype=float)
        hi = np.asarray(res.get('yhat_upper', []), dtype=float)
        model_tag = res.get('model', 'sarimax')

    # Clip predictions to reasonably near historical range
    lower_bound = max(0.0, hist_min * 0.5)
    upper_bound = max(hist_max * 1.5, hist_mean + 10 * hist_std)

    yhat_clipped = np.clip(yhat, lower_bound, upper_bound)
    lo_clipped = np.clip(lo, lower_bound, upper_bound)
    hi_clipped = np.clip(hi, lower_bound, upper_bound)

    # ensure coherency lo <= yhat <= hi
    lo_final = np.minimum(lo_clipped, yhat_clipped)
    hi_final = np.maximum(hi_clipped, yhat_clipped)

    return {
        "yhat": pd.Series(yhat_clipped),
        "yhat_lower": pd.Series(lo_final),
        "yhat_upper": pd.Series(hi_final),
        "model": model_tag
    }
