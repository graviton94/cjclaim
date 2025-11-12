from __future__ import annotations
from pathlib import Path
import pandas as pd

def reconcile_month(forecast_parquet: str, weekly_ts_parquet: str,
                    out_csv: str, bias_map_path: str | None = None) -> dict:
    f = Path(forecast_parquet); w = Path(weekly_ts_parquet)
    if not f.exists() or not w.exists():
        raise FileNotFoundError("Forecast or weekly_ts not found for reconcile.")
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"series_id": [], "bias_adj": []}).to_csv(out, index=False)
    return {"reconcile": str(out)}
