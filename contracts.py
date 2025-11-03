from typing import Literal, Optional
import pandas as pd
import numpy as np

CURATED_COLUMNS = {
    "series_id": "string",
    "year": "int64",
    "week": "int64",
    "claim_count": "float64",
    "plant": "string",
    "product_cat2": "string",
    "mid_category": "string",
}

def validate_curated(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CURATED_COLUMNS if c not in df.columns]
    assert not missing, f"curated missing cols: {missing}"
    df = df.astype({k:v for k,v in CURATED_COLUMNS.items() if k in df.columns})
    assert df["week"].between(1, 53).all(), "week must be 1..53"
    assert (df["claim_count"] >= 0).all(), "claim_count must be >= 0"
    full = (
        df.groupby(["series_id","year"])["week"]
          .apply(lambda s: set(s.tolist()) == set(range(1, 53+1)))
    )
    assert full.all(), "weekly padding incomplete in curated layer"
    return df

FEATURES_COLUMNS = {
    "series_id": "string",
    "year": "int64",
    "week": "int64",
    "psi": "float64",
    "peak_flag": "int64",
    "cp_flag": "int64",
    "amplitude": "float64",
}
def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES_COLUMNS if c not in df.columns]
    assert not missing, f"features missing cols: {missing}"
    df = df.astype(FEATURES_COLUMNS)
    assert df["peak_flag"].isin([0,1]).all()
    assert df["cp_flag"].isin([0,1]).all()
    return df

FORECAST_COLS = {
    "series_id":"string","year":"int64","week":"int64",
    "y_pred":"float64","y_lo":"float64","y_hi":"float64",
    "model_ver":"string","train_until":"int64"
}
METRIC_COLS = {
    "series_id":"string","year":"int64",
    "MAPE":"float64","MASE":"float64","Bias":"float64","RMSE":"float64","n_points":"int64"
}
ADJUSTMENT_KEYS = {"bias_intercept","seasonal_adj","changepoint_flags"}

def validate_forecast(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FORECAST_COLS if c not in df.columns]
    assert not missing, f"forecast missing cols: {missing}"
    df = df.astype(FORECAST_COLS)
    return df

def validate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in METRIC_COLS if c not in df.columns]
    assert not missing, f"metrics missing cols: {missing}"
    df = df.astype(METRIC_COLS)
    return df
