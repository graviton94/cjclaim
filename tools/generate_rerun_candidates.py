"""Generate rerun candidate list from trained_models_v3.json.
Select series with n_train_points >= N (default 36) and with seasonal_order == (0,0,0,0)
and write artifacts/rerun_candidates.json with key 'untrained_series'.
"""
import json
from pathlib import Path

MIN_POINTS = 36
MAX_CANDIDATES = 1000

trained_path = Path('artifacts/trained_models_v3.json')
if not trained_path.exists():
    raise SystemExit('trained_models_v3.json not found')

with open(trained_path, 'r', encoding='utf-8') as f:
    trained = json.load(f)

series = trained.get('series_models', {}) or {}
cands = []
for sid, info in series.items():
    n = info.get('n_train_points') or info.get('n_train') or 0
    ms = info.get('model_spec') or {}
    seasonal = tuple(ms.get('seasonal_order') or ())
    if n >= MIN_POINTS and seasonal == (0,0,0,0):
        cands.append(sid)

cands = cands[:MAX_CANDIDATES]
out = {'untrained_series': cands, 'generated_count': len(cands)}
Path('artifacts').mkdir(parents=True, exist_ok=True)
with open('artifacts/rerun_candidates.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Wrote artifacts/rerun_candidates.json ({len(cands)} candidates)")
