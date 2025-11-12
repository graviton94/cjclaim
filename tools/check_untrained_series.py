"""Compare claims_monthly.parquet series vs trained_models.json keys and write untrained_series.json

Usage:
    python tools/check_untrained_series.py --parquet data/curated/claims_monthly.parquet --trained artifacts/trained_models.json
"""
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--parquet', required=True)
parser.add_argument('--trained', required=True)
args = parser.parse_args()

p = Path(args.parquet)
if not p.exists():
    raise SystemExit(f"Parquet not found: {p}")

t = Path(args.trained)
if not t.exists():
    raise SystemExit(f"Trained models not found: {t}")

# load data
df = pd.read_parquet(p)
if 'series_id' not in df.columns:
    raise SystemExit('parquet missing series_id column')
all_series = sorted(df['series_id'].unique())

with open(t, 'r', encoding='utf-8') as f:
    trained = json.load(f)

trained_keys = set(trained.get('series_models', {}).keys())

untrained = [s for s in all_series if s not in trained_keys]

out_path = Path('artifacts') / 'untrained_series.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({'untrained_count': len(untrained), 'untrained_series': untrained}, f, indent=2, ensure_ascii=False)

print(f"Wrote {out_path} ({len(untrained)} series untrained)")