"""Generate artifacts/fit_summary_log.json from artifacts/trained_models.json

Produces a JSON array with one record per series containing:
 - series_id, seasonal_order, aic, n_train_points, arroots (if present)

Usage:
  python tools/gen_fit_summary.py
"""
import json
from pathlib import Path

trained_path = Path('artifacts/trained_models.json')
out_path = Path('artifacts/fit_summary_log.json')
if not trained_path.exists():
    raise SystemExit('trained_models.json not found')

with open(trained_path, 'r', encoding='utf-8') as f:
    trained = json.load(f)

series = trained.get('series_models', {})
rows = []
for sid, info in series.items():
    spec = info.get('model_spec', {}) or {}
    seasonal = spec.get('seasonal_order') if isinstance(spec, dict) else None
    aic = info.get('aic')
    n_train = info.get('n_train_points') or info.get('n_train') or info.get('n_train_points')
    arroots = info.get('arroots')
    rows.append({
        'series_id': sid,
        'seasonal_order': seasonal,
        'aic': aic,
        'n_train_points': n_train,
        'arroots': arroots,
        'model_spec': spec
    })

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f'Wrote fit summary: {out_path} ({len(rows)} rows)')
