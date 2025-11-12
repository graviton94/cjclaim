import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
orig_p = root / 'artifacts' / 'trained_models_v3.json'
new_p = root / 'artifacts' / 'trained_models_v3_rerun_500.json'
out_json = root / 'artifacts' / 'rerun_changed_details.json'
out_csv = root / 'artifacts' / 'rerun_changed_details.csv'

if not orig_p.exists() or not new_p.exists():
    print('missing snapshots:', orig_p.exists(), new_p.exists())
    raise SystemExit(1)

with open(orig_p,'r',encoding='utf-8') as f:
    orig = json.load(f)
with open(new_p,'r',encoding='utf-8') as f:
    new = json.load(f)

# trained_models_v3.json layout: {"metadata":..., "series_models": {series_id: {..}}}
if isinstance(orig, dict) and 'series_models' in orig:
    orig_map = orig['series_models']
else:
    # fallback: assume list of records
    orig_map = {r['series_id']: r for r in orig}

if isinstance(new, dict) and 'series_models' in new:
    new_map = new['series_models']
else:
    new_map = {r['series_id']: r for r in new}

changed = []
for sid in sorted(set(orig_map) | set(new_map)):
    o = orig_map.get(sid)
    n = new_map.get(sid)
    if not o or not n:
        continue
    # compare model_spec and aic
    o_spec = o.get('model_spec')
    n_spec = n.get('model_spec')
    o_aic = o.get('aic')
    n_aic = n.get('aic')
    o_ar = o.get('arroots')
    n_ar = n.get('arroots')
    if o_spec != n_spec or o_aic != n_aic or o_ar != n_ar:
        changed.append({
            'series_id': sid,
            'orig_model_spec': o_spec,
            'rerun_model_spec': n_spec,
            'orig_aic': o_aic,
            'rerun_aic': n_aic,
            'orig_arroots': o_ar,
            'rerun_arroots': n_ar,
        })

with open(out_json,'w',encoding='utf-8') as f:
    json.dump({'n_changed': len(changed), 'changed': changed}, f, ensure_ascii=False, indent=2)

# write CSV header
import csv
with open(out_csv,'w',encoding='utf-8',newline='') as f:
    w = csv.writer(f)
    w.writerow(['series_id','orig_aic','rerun_aic','aic_delta','orig_model_spec','rerun_model_spec'])
    for c in changed:
        delta = None
        try:
            delta = None if c['orig_aic'] is None or c['rerun_aic'] is None else (c['rerun_aic'] - c['orig_aic'])
        except Exception:
            delta = ''
        w.writerow([c['series_id'], c['orig_aic'], c['rerun_aic'], delta, json.dumps(c['orig_model_spec'], ensure_ascii=False), json.dumps(c['rerun_model_spec'], ensure_ascii=False)])

print('wrote', out_json, out_csv, 'changed=', len(changed))
