"""Monthly SARIMAX helpers with sqrt-transform and enforced stationarity/invertibility.

Functions:
- fit_monthly_sarimax(y_m: pd.Series) -> SARIMAXResults
- forecast_and_inverse(res, steps: int) -> np.ndarray
- loss_with_bias(y_true, y_pred, lam=0.5) -> float

Design: transform y -> sqrt(y) before fitting; after forecast, square back and clip >=0.
"""
from typing import Optional
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def fit_with_retries(model, maxiter_list=(50, 200), methods=('lbfgs', 'bfgs', 'powell')):
    """Attempt to fit a statsmodels model with multiple optimizers and iteration budgets.

    - model: a statsmodels model object with .fit(method=..., maxiter=...)
    - maxiter_list: iterable of maxiter values to attempt in order
    - methods: iterable of method names to try for each maxiter

    Raises RuntimeError with last exception if all attempts fail.
    Performs basic checks for convergence and AR-root stability when available.
    """
    last_exc = None
    for it in maxiter_list:
        for m in methods:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', ConvergenceWarning)
                    res = model.fit(disp=False, method=m, maxiter=it)
                # check mle_retvals for convergence flag when present
                if hasattr(res, 'mle_retvals') and isinstance(res.mle_retvals, dict):
                    conv = res.mle_retvals.get('converged', True)
                    if not conv:
                        raise RuntimeError(f'converged=False ({m},{it})')
                # AR-root stability check: require |arroots| > 1 for stability in our policy
                if hasattr(res, 'arroots'):
                    import numpy as _np
                    if _np.any(_np.abs(res.arroots) <= 1):
                        raise RuntimeError(f'unstable AR roots ({m},{it})')
                return res
            except Exception as e:
                last_exc = e
                continue
    raise RuntimeError(f'All fit attempts failed: {last_exc}')


def fit_monthly_sarimax(y_m: pd.Series, order=(2, 0, 0), seasonal_order=None):
    """Fit SARIMAX to monthly series after sqrt transform.

    - y_m: pandas Series indexed by DatetimeIndex with freq='MS' (month-start)
    - order: non-seasonal (p,d,q)
    - seasonal_order: if None, try seasonal candidates [(1,1,1,12),(0,1,1,12)]

    Raises ValueError on input problems and RuntimeError if all fits fail.
    """
    if not isinstance(y_m, pd.Series):
        raise ValueError("y_m must be a pandas Series")
    if y_m.isnull().any():
        raise ValueError("y_m contains NaN values")
    if (y_m < 0).any():
        raise ValueError("y_m contains negative values")

    # require a minimum number of months to fit seasonal model (e.g. 24 months)
    min_months = 24
    if len(y_m) < min_months:
        raise ValueError(f"Insufficient data for monthly SARIMAX fit: need >= {min_months} months, got {len(y_m)}")

    # transform
    y_t = np.sqrt(y_m.astype(float))

    # seasonal_order may be provided as a single 4-tuple (p,d,q,12) or as an
    # iterable of candidate tuples. If a single 4-tuple is provided we must
    # wrap it as a single-element list; otherwise iterating directly over a
    # 4-tuple would yield ints and later code will fail with "object of type
    # 'int' has no len()". Default to two seasonal candidates when None.
    if seasonal_order is None:
        seasonal_candidates = [(1, 1, 1, 12), (0, 1, 1, 12)]
    else:
        # if seasonal_order looks like a single 4-tuple (e.g. (1,1,1,12)),
        # wrap it into a list; if it's already a list/iterable of tuples use it
        # directly.
        if isinstance(seasonal_order, (list, tuple)) and len(seasonal_order) == 4 and all(isinstance(x, int) for x in seasonal_order):
            seasonal_candidates = [tuple(seasonal_order)]
        else:
            # coerce to list to support e.g. generator or other iterable
            seasonal_candidates = list(seasonal_order)
    best_res = None
    best_aic = np.inf
    last_exc = None

    for so in seasonal_candidates:
        try:
            mod = SARIMAX(
                endog=y_t,
                order=order,
                seasonal_order=so,
                trend='n',
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            # use fit_with_retries for robust fitting
            res = fit_with_retries(mod, maxiter_list=(50, 200), methods=('lbfgs', 'bfgs', 'powell'))
            # choose by AIC
            if hasattr(res, 'aic') and res.aic < best_aic:
                best_res = res
                best_aic = res.aic
        except Exception as e:
            last_exc = e
            continue

    if best_res is None:
        raise RuntimeError(f"All seasonal fits failed: {last_exc}")
    # mark that this fit used the sqrt transform so callers can compare
    # transform metadata across candidates when making selection decisions.
    try:
        setattr(best_res, '_transform', 'sqrt')
    except Exception:
        pass
    return best_res


def forecast_and_inverse(res, steps: int = 6) -> np.ndarray:
    """Forecast the next `steps` values from a fitted SARIMAX result, inverse the sqrt transform.

    Returns numpy array of length `steps` with non-negative floats.
    """
    if steps <= 0:
        return np.array([])
    yhat_t = res.forecast(steps=steps)
    # inverse transform
    yhat = np.square(yhat_t)
    # clip to avoid tiny negative rounding
    yhat = np.maximum(yhat, 0.0)
    return np.asarray(yhat)


def loss_with_bias(y_true: np.ndarray, y_pred: np.ndarray, lam: float = 0.5) -> float:
    """Custom loss: MAPE + lam * |mean_error|. y_true and y_pred are numpy arrays.

    Used for model selection to penalize systematic over-forecasting.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    # avoid division by zero by using max(y_true,1)
    denom = np.maximum(y_true, 1.0)
    mape = np.mean(np.abs((y_true - y_pred) / denom))
    me = np.mean(y_pred - y_true)
    return float(mape + lam * abs(me))
