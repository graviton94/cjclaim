# -*- coding: utf-8 -*-
import pandas as pd
from typing import List
from .constants import WEEK_FREQ

GROUP_COLS = ["플랜트", "제품범주2", "중분류"]

def weekly_agg_from_counts(df: pd.DataFrame,
                           date_col: str = "제조일자",
                           value_col: str = "count",
                           group_cols: List[str] = GROUP_COLS) -> pd.DataFrame:
    """
    이미 count가 포함된 레코드를 '제조일자' 기준 주간 합계로 집계.
    반환: series_id, group_cols, week, y
    """
    df = df.copy()
    df["week"] = pd.to_datetime(df[date_col]).dt.to_period(WEEK_FREQ).dt.start_time
    g = (df.groupby(group_cols + ["week"], dropna=False)[value_col]
            .sum()
            .rename("y")
            .reset_index())
    g["series_id"] = g[group_cols].astype(str).agg("|".join, axis=1)
    return g[["series_id", *group_cols, "week", "y"]].sort_values(["series_id", "week"])
