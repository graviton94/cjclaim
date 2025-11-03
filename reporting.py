from pathlib import Path
import pandas as pd
from io_utils import ART

def report_year(year: int, outpath: Path):
    met = pd.read_parquet(ART/f"metrics/metrics_{year}.parquet")
    tbl = met.groupby("year")[["MAPE","MASE","Bias","RMSE"]].mean().round(3).reset_index()
    md = [f"# 예측 성능 리포트 ({year})", "", "| 연도 | MAPE | MASE | Bias | RMSE |",
          "|-----|------|------|------|------|"]
    for _,r in tbl.iterrows():
        md.append(f"| {int(r['year'])} | {r['MAPE']} | {r['MASE']} | {r['Bias']} | {r['RMSE']} |")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(md), encoding="utf-8")
