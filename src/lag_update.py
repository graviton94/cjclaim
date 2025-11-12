from __future__ import annotations
import pandas as pd
from pathlib import Path
from .utils_io import ensure_dir

def update_lag_stats_from_df(df: pd.DataFrame, out_csv: str) -> dict:
    df = df.copy()
    df['발생일자'] = pd.to_datetime(df['발생일자'], errors='coerce')
    df['제조일자'] = pd.to_datetime(df['제조일자'], errors='coerce')
    df = df.dropna(subset=['발생일자','제조일자'])
    df['lag_days'] = (df['발생일자'] - df['제조일자']).dt.days.clip(lower=0)
    grp = df.groupby('제품범주2')['lag_days']
    stat = grp.quantile([0.5, 0.9]).unstack().reset_index().rename(columns={0.5:'lag_p50', 0.9:'lag_p90'})
    stat['normal_low'] = (stat['lag_p50'] - (stat['lag_p90'] - stat['lag_p50'])*1.5).clip(lower=0).round().astype(int)
    stat['normal_high'] = (stat['lag_p90']).round().astype(int)
    out = Path(out_csv)
    ensure_dir(out.parent)
    stat.to_csv(out, index=False, encoding='utf-8-sig')
    return {"lag_stats": str(out), "rows": len(stat)}
