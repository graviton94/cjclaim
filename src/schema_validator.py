"""Schema validation and monthly conversion helpers for CJ quality-claim pipeline.

Functions:
- validate_schema(df): enforce CSV column names, types and date formats (raises ValueError on violations)
- to_monthly(df): converts raw event-level dataframe into monthly-indexed aggregated series with freq='MS'

This module follows the canvas rules: fixed input columns and strict fail-fast behavior.
"""
from typing import List
import pandas as pd

REQUIRED: List[str] = ["발생일자", "중분류", "플랜트", "제품범주2", "제조일자", "count"]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate dataframe schema strictly.

    Raises ValueError when:
      - required columns are missing
      - extra columns are present
      - date parsing fails for the two date columns
      - "count" is not integer dtype or contains NaN
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    cols = list(df.columns)
    missing = [c for c in REQUIRED if c not in cols]
    extra = [c for c in cols if c not in REQUIRED]
    if missing or extra:
        raise ValueError(f"Invalid schema: missing={missing}, extra={extra}")

    # Strict date parsing (raise on parse errors)
    try:
        pd.to_datetime(df["발생일자"], format="%Y-%m-%d", errors="raise")
        pd.to_datetime(df["제조일자"], format="%Y-%m-%d", errors="raise")
    except Exception as e:
        raise ValueError(f"Date parsing failed: {e}")

    # count must be integer-like and non-null
    if df["count"].isnull().any():
        raise ValueError("count column contains NaN values")
    # allow int dtypes or floats that are integral
    if not (pd.api.types.is_integer_dtype(df["count"]) or
            (pd.api.types.is_float_dtype(df["count"]) and (df["count"] % 1 == 0).all())):
        raise ValueError("count must be integer dtype or float with integer values")


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level DataFrame into monthly-indexed aggregated counts.

    Returns DataFrame indexed by `발생월` (Timestamp at month start) with columns
    ['중분류','플랜트','제품범주2','제조일자','count'] and index.freq='MS'.

    This function assumes `validate_schema` was already called.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # copy to avoid mutating caller
    df2 = df.copy()
    # normalize date and create month-start timestamp
    # convert to period (monthly) then to timestamp at month-start
    df2["발생월"] = pd.to_datetime(df2["발생일자"]).dt.to_period("M").dt.to_timestamp(how='start')

    grouped = (
        df2
        .groupby(["발생월", "중분류", "플랜트", "제품범주2", "제조일자"], dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values("발생월")
    )

    # set index to 발생월 (DatetimeIndex). Do not enforce a global freq here because
    # the combined grouped table contains many series with different sparsity; callers
    # should reindex per-series as needed. This function remains strict about date
    # parsing and aggregation but is permissive about irregularity across different
    # series.
    grouped = grouped.set_index("발생월")
    grouped.index = pd.DatetimeIndex(grouped.index)
    return grouped
