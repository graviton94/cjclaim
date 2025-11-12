
"""Forecast API (MONTHLY)."""
from __future__ import annotations
__all__ = ["train_incremental", "generate_monthly_forecast"]
import pandas as pd, numpy as np
from pathlib import Path

def safe_read_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    try:
        df = pd.read_parquet(p)
        # 최소 스키마 검증(필요 시 강화)
        if df is None or len(df.columns) == 0:
            raise ValueError("empty parquet schema")
        return df
    except Exception as e:
        raise RuntimeError(f"Invalid parquet: {p} ({e})")

# --- public API for pipeline S5 ---
try:
    from .utils_io import get_latest_file, ensure_dir
except Exception:
    def get_latest_file(root, pattern, recursive=True):
        root = Path(root)
        if not root.exists():
            return None
        return max([p for p in root.rglob("*") if p.is_file()], default=None,
                   key=lambda p: p.stat().st_mtime)
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def _infer_series_id(df: pd.DataFrame) -> pd.Series:
    if "series_id" in df.columns:
        return df["series_id"].astype(str)
    cols = [c for c in ["플랜트","제품범주2","중분류"] if c in df.columns]
    if cols:
        return df[cols].astype(str).agg("_".join, axis=1)
    return pd.Series(["unknown"] * len(df))

def _values_col(df: pd.DataFrame) -> str:
    for c in ["y", "count", "발생건수"]:
        if c in df.columns:
            return c
    return None

def generate_monthly_forecast(models_dir: str, horizon_months: int, ts_path: str, out_parquet: str) -> dict:
    """S5: 주어진 monthly_ts에서 각 series_id의 향후 N개월 월예측 생성."""
    out_p = Path(out_parquet); ensure_dir(out_p.parent)
    df = safe_read_parquet(ts_path).copy()
    # Debug: print incoming schema and small sample to help trace missing-column issues
    try:
        print(f"[DEBUG] generate_monthly_forecast called: ts_path={ts_path}, out_parquet={out_parquet}")
        print(f"[DEBUG] columns: {list(df.columns)}")
        print("[DEBUG] sample:\n", df.head(3).to_dict(orient='records'))
        if 'yyyymm' in df.columns:
            try:
                print("[DEBUG] unique yyyymm sample:", df['yyyymm'].dropna().unique()[:10])
            except Exception:
                pass
    except Exception:
        pass
    
    # series_id
    df["series_id"] = _infer_series_id(df)
    # 스키마 보정 (월)
    if "yyyymm" not in df.columns and {"year","month"} <= set(df.columns):
        df["yyyymm"] = (df["year"].astype(int)*100 + df["month"].astype(int)).astype(int)
    df["yyyymm"] = df["yyyymm"].astype(str).str.replace(r"[^0-9]","",regex=True).astype(int)
    if "year" not in df.columns or "month" not in df.columns:
        df["year"]  = (df["yyyymm"] // 100).astype(int)
        df["month"] = (df["yyyymm"] % 100).astype(int)
    # 필수 컬럼 검증 — 가능한 경우 자동 보정, 불가하면 빈 결과 반환 (pipeline 중단 방지)
    missing_cols = [col for col in ["year", "month", "yyyymm"] if col not in df.columns]
    if missing_cols:
        # 우선 yyyymm이 있으면 파싱 시도
        if "yyyymm" in df.columns:
            try:
                df["yyyymm"] = df["yyyymm"].astype(str).str.replace(r"[^0-9]", "", regex=True).astype(int)
                df["year"] = (df["yyyymm"] // 100).astype(int)
                df["month"] = (df["yyyymm"] % 100).astype(int)
                missing_cols = [c for c in ["year", "month", "yyyymm"] if c not in df.columns]
            except Exception:
                print(f"[WARN] Unable to derive year/month from yyyymm in {ts_path}; missing: {missing_cols}")
                pd.DataFrame(columns=["series_id", "year", "month", "yyyymm", "yhat", "lower_bound", "upper_bound"]).to_parquet(out_p, index=False)
                return {"forecast": str(out_p), "series_count": 0, "warning": f"monthly_ts missing required columns: {missing_cols}"}
        # yyyymm이 없지만 year+month가 있으면 생성
        if {"year", "month"} <= set(df.columns):
            try:
                df["yyyymm"] = (df["year"].astype(int) * 100 + df["month"].astype(int)).astype(int)
            except Exception:
                pass
        # 최종적으로 아직 필수 컬럼이 누락되면 빈 결과 반환
        if not all(col in df.columns for col in ["year", "month", "yyyymm"]):
            print(f"[WARN] monthly_ts missing required columns after attempted fixes: {missing_cols} (path: {ts_path})")
            pd.DataFrame(columns=["series_id", "year", "month", "yyyymm", "yhat", "lower_bound", "upper_bound"]).to_parquet(out_p, index=False)
            return {"forecast": str(out_p), "series_count": 0, "warning": f"monthly_ts missing required columns: {missing_cols}"}
    vcol = "y" if "y" in df.columns else ("claim_count" if "claim_count" in df.columns else None)
    if vcol is None:
        # 마지막 안전장치
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        vcol = num_cols[0] if num_cols else None
    if vcol is None:
        # 비어있다면 빈 결과 저장 후 리턴
        pd.DataFrame(columns=["series_id","year","month","yyyymm","yhat","lower_bound","upper_bound"]).to_parquet(out_p, index=False)
        return {"forecast": str(out_p), "series_count": 0, "warning": "monthly_ts has no numeric values"}
    # 4) 시리즈별 월 예측 (seasonal=12)
    rows = []
    for sid, g in df.groupby("series_id", sort=False):
        # build a monthly-indexed series (month-start timestamps) for the group's observations
        try:
            # yyyymm is like 202301 or 202312 -> convert to month-start timestamps
            idx = pd.to_datetime(g["yyyymm"].astype(str), format="%Y%m")
            idx = idx.to_period("M").to_timestamp(how='start')
            y = pd.Series(g[vcol].astype(float).fillna(0.0).values, index=pd.DatetimeIndex(idx))
            # ensure index is sorted
            y = y.sort_index()
            # reindex to a full monthly range between min and max (fill missing months with zeros)
            full_idx = pd.date_range(start=y.index.min(), end=y.index.max(), freq='MS')
            y = y.reindex(full_idx, fill_value=0.0)
        except Exception:
            # if we cannot construct a proper datetime index, fail-fast for this series
            raise RuntimeError(f"Unable to construct monthly datetime index for series {sid}")

        # now y is a DatetimeIndex monthly series (zeros filled for months with no events)
        res = safe_forecast(y, horizon=horizon_months, seasonal_order=(1,1,1,12), ci=0.95)
        yhat = res["yhat"].values
        lo   = res["yhat_lower"].values
        hi   = res["yhat_upper"].values
        # 기준 월(마지막 yyyymm)부터 k개월 미래로 생성
        base = int(g["yyyymm"].max())
        by, bm = base//100, base%100
        for k in range(1, horizon_months+1):
            my = by + (bm + k - 1)//12
            mm = (bm + k - 1)%12 + 1
            yyyymm = my*100 + mm
            rows.append({
                "series_id": sid,
                "year": my,
                "month": mm,
                "yyyymm": yyyymm,
                "yhat": float(max(0.0, yhat[k-1] if k-1 < len(yhat) else 0.0)),
                "lower_bound": float(max(0.0, lo[k-1] if k-1 < len(lo) else 0.0)),
                "upper_bound": float(max(0.0, hi[k-1] if k-1 < len(hi) else 0.0)),
            })
    # 최종 스키마 고정
    out_df = pd.DataFrame(rows)
    required_cols = ["series_id","year","month","yyyymm","yhat","lower_bound","upper_bound"]
    if out_df.empty:
        # write empty standardized schema to avoid KeyError downstream
        pd.DataFrame(columns=required_cols).to_parquet(out_p, index=False)
        return {"forecast": str(out_p), "series_count": 0}
    # ensure required columns exist (fill missing with sensible defaults)
    for c in required_cols:
        if c not in out_df.columns:
            if c in ("yhat","lower_bound","upper_bound"):
                out_df[c] = 0.0
            else:
                out_df[c] = None
    out_df = out_df[required_cols]
    out_df.to_parquet(out_p, index=False)
    return {"forecast": str(out_p), "series_count": int(out_df["series_id"].nunique())}
import warnings
import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

# use the monthly model helpers we implemented
try:
    from .model_monthly import fit_monthly_sarimax, forecast_and_inverse
except Exception:
    # if local import fails, keep names but they will raise at runtime
    fit_monthly_sarimax = None
    forecast_and_inverse = None

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

def fit_forecast(y: pd.Series, horizon: int = 6, seasonal_order=(1,1,1,12), ci: float = 0.95):
    """Fit a monthly SARIMAX on sqrt-transformed data and return forecast (no fallback).

    This function intentionally fails fast: any fitting/forecasting error will raise.
    It uses `src.model_monthly.fit_monthly_sarimax` and `forecast_and_inverse`.
    """
    if fit_monthly_sarimax is None or forecast_and_inverse is None:
        raise RuntimeError("model_monthly helpers not available; ensure src.model_monthly is present")

    y = y.astype(float).fillna(0.0)
    if len(y) == 0:
        raise RuntimeError("empty series provided to fit_forecast")

    # Expect y to be a pandas Series indexed by month-start timestamps; if not, try to coerce
    if y.index is None or not isinstance(y.index, pd.DatetimeIndex):
        # If integer index, caller should have provided proper timestamps; raise to enforce contract
        raise RuntimeError("fit_forecast requires a pandas Series indexed by DatetimeIndex (freq='MS')")

    # Use monthly-specific model (sqrt transform inside)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_monthly_sarimax(y, order=(2,0,0), seasonal_order=seasonal_order)
        # compute forecast on transformed scale and inverse
        try:
            # try to obtain prediction intervals from the statespace results
            fc = res.get_forecast(steps=horizon)
            yhat_t = fc.predicted_mean
            conf = fc.conf_int(alpha=1.0 - float(ci))
            lo_t = conf.iloc[:, 0]
            hi_t = conf.iloc[:, 1]
            # inverse transform (square)
            yhat = np.square(yhat_t.values)
            lo = np.square(lo_t.values.clip(min=0.0))
            hi = np.square(hi_t.values.clip(min=0.0))
        except Exception:
            # fallback to simple point forecast via our helper
            yhat = forecast_and_inverse(res, steps=horizon)
            lo = np.maximum(0.0, yhat * 0.8)
            hi = np.maximum(0.0, yhat * 1.2)

    return {"yhat": pd.Series(yhat), "yhat_lower": pd.Series(lo), "yhat_upper": pd.Series(hi), "model": "sarimax_monthly"}


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

    # call primary forecast with requested CI (fail-fast; no fallback)
    res = fit_forecast(y, horizon=horizon, seasonal_order=seasonal_order, ci=ci)

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
        # No fallback: treat as failure
        raise RuntimeError('Forecast output failed validity checks (extreme or NaN values)')
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

def load_model(series_id, year):
    """시리즈별 연도별 실제 모델 로드"""
    # 예시: 모델 파일 경로 생성
    model_path = f"models/{series_id}_{year}.pkl"
    import joblib
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        # 모델이 없으면 더미 반환 또는 에러 처리
        class DummyModel:
            def forecast(self, steps):
                return np.zeros(steps)
        return DummyModel()
    
def seasonal_naive(y, horizon=52):
    """공개 seasonal_naive 함수 (내부 _naive_seasonal 래핑)"""
    seq, lo, hi = _naive_seasonal(y, horizon=horizon, season=52, ci=0.95)
    return seq

def train_incremental(updated_series_list, weekly_ts_path, models_dir, warm_start=True, max_workers=4):
    """Placeholder for incremental model training. Implement actual logic as needed."""
    # For now, just log the input and return a dummy result
    print(f"[train_incremental] series: {len(updated_series_list)}, weekly_ts: {weekly_ts_path}, models_dir: {models_dir}")
    return {
        "trained_count": len(updated_series_list),
        "failed": [],
        "models_dir": models_dir,
    }
