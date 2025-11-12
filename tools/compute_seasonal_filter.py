"""Compute seasonal strength (lag-12 ACF) per series and emit ranked candidates.
Writes artifacts/seasonal_strength.json and artifacts/seasonal_candidates.json

Rules used:
 - only consider series with n_train >= 36
 - compute acf12 = pandas.Series.autocorr(lag=12) on y_train (drop last 6 months)
 - seasonal_score = abs(acf12)
 - candidate if seasonal_score >= 0.2

Output keys: series_id, n_train, acf12, seasonal_score, candidate (bool)
"""
import json
from pathlib import Path
import math
import pandas as pd

TRAINED = Path('artifacts/trained_models_v3.json')
PARQUET = Path('data/curated/claims_monthly.parquet')
OUT1 = Path('artifacts/seasonal_strength.json')
OUT2 = Path('artifacts/seasonal_candidates.json')

if not TRAINED.exists():
    raise SystemExit('trained_models_v3.json not found')
if not PARQUET.exists():
    raise SystemExit('Parquet data missing')

with open(TRAINED, 'r', encoding='utf-8') as f:
    trained = json.load(f)
series = trained.get('series_models', {}) or {}

df = pd.read_parquet(PARQUET)

rows = []
for sid, info in series.items():
    try:
        df_s = df[df['series_id'] == sid][['year','month','claim_count']].copy()
        if df_s.empty:
            continue
        df_s = df_s.groupby(['year','month'], as_index=False)['claim_count'].sum().sort_values(['year','month'])
        idx = pd.to_datetime(df_s[['year','month']].assign(day=1))
        y = pd.Series(df_s['claim_count'].values, index=idx).sort_index()
        if len(y) <= 6:
            continue
        y_train = y[:-6]
        n_train = len(y_train)
        acf12 = None
        try:
            if n_train >= 13:
                acf12 = float(y_train.autocorr(lag=12))
            else:
                acf12 = None
        except Exception:
            acf12 = None
        seasonal_score = abs(acf12) if acf12 is not None and not math.isnan(acf12) else 0.0
        candidate = (n_train >= 36) and (seasonal_score >= 0.2)
        rows.append({'series_id': sid, 'n_train': int(n_train), 'acf12': acf12, 'seasonal_score': seasonal_score, 'candidate': candidate})
    except Exception:
        continue

# write full strength file and a ranked candidate list
rows_sorted = sorted(rows, key=lambda r: (r['candidate'], r['seasonal_score']), reverse=True)
Path('artifacts').mkdir(parents=True, exist_ok=True)
with open(OUT1, 'w', encoding='utf-8') as f:
    json.dump(rows_sorted, f, ensure_ascii=False, indent=2)

cands = [r['series_id'] for r in rows_sorted if r['candidate']]
with open(OUT2, 'w', encoding='utf-8') as f:
    json.dump({'candidates': cands, 'count': len(cands)}, f, ensure_ascii=False, indent=2)

print('Wrote', OUT1, 'and', OUT2, 'candidates=', len(cands))
