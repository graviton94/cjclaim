# -*- coding: utf-8 -*-
"""
CSV -> 주간 집계 파켓/CSV 산출 (제조일자 기준).
DuckDB 적재는 후속 단계에서 추가.
"""
import argparse
from pathlib import Path
import pandas as pd

from src.constants import DATA_DIR
from src.io_utils import load_min_csv
from src.preprocess import weekly_agg_from_counts

def main(csv_path: str, out_stem: str = "weekly_agg"):
    df = load_min_csv(csv_path)
    g = weekly_agg_from_counts(df)  # series_id, group_cols, week, y
    out_parquet = DATA_DIR / f"{out_stem}.parquet"
    out_csv = DATA_DIR / f"{out_stem}.csv"
    g.to_parquet(out_parquet, index=False)
    g.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 주간 집계 저장: {out_parquet} , {out_csv}")
    # 간단 요약
    s = g.groupby("series_id")["y"].sum().sort_values(ascending=False).head(10)
    print("TOP10 series by total count:")
    print(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DATA_DIR / "claims.csv"))
    parser.add_argument("--out_stem", default="weekly_agg")
    args = parser.parse_args()
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    main(args.csv, args.out_stem)
