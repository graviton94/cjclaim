import pandas as pd
from pathlib import Path
from io_utils import read_parquet, write_parquet, log_jsonl, ART
from forecasting import load_model, seasonal_naive

def forecast_year(curated_path, year):
    df = read_parquet(curated_path)
    out = []
    for series, g in df[df["year"]<=year-1].groupby("series_id"):
        y = g.sort_values(["year","week"])["claim_count"].reset_index(drop=True)
        try:
            model = load_model(series, year-1)
            fc = model.forecast(steps=52)
            y_pred = pd.DataFrame({"series_id":series,"year":year,"week":range(1,53),"y_pred":fc})
        except Exception as e:
            fc = seasonal_naive(y, 52)
            y_pred = pd.DataFrame({"series_id":series,"year":year,"week":range(1,53),"y_pred":fc.values})
            log_jsonl({"event":"forecast_fallback","series":series,"year":year,"reason":str(e)})
        y_pred["y_lo"] = (y_pred["y_pred"]*0.9)
        y_pred["y_hi"] = (y_pred["y_pred"]*1.1)
        y_pred["model_ver"] = "sarimax_v1"
        y_pred["train_until"] = year-1
        out.append(y_pred)
    outdf = pd.concat(out, ignore_index=True)
    write_parquet(outdf, ART/f"forecasts/{year}.parquet")
    log_jsonl({"event":"forecast","year":year,"ok":True,"n_series":outdf['series_id'].nunique()})
