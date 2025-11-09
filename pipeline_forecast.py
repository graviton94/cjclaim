import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from io_utils import write_parquet, log_jsonl, ART
import pandas as pd
from forecasting import load_model, seasonal_naive

def forecast_year(curated_path, year, output_path=None):
    df = pd.read_parquet(curated_path)
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
    if output_path:
        write_parquet(outdf, output_path)
    else:
        write_parquet(outdf, ART/f"forecasts/{year}.parquet")
    log_jsonl({"event":"forecast","year":year,"ok":True,"n_series":outdf['series_id'].nunique()})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=False)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    forecast_year("data/curated/claims_monthly.parquet", args.year, args.output)
