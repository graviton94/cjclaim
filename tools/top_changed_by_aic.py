import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
inp = root / 'artifacts' / 'rerun_changed_details.json'
out = root / 'artifacts' / 'top_changed_by_aic.csv'
if not inp.exists():
    print('missing', inp)
    raise SystemExit(1)
with open(inp,'r',encoding='utf-8') as f:
    data = json.load(f)
rows = []
for c in data.get('changed',[]):
    oa = c.get('orig_aic')
    ra = c.get('rerun_aic')
    if oa is None or ra is None:
        continue
    try:
        delta = ra - oa
    except Exception:
        continue
    rows.append((delta, c['series_id'], oa, ra, c.get('orig_model_spec'), c.get('rerun_model_spec')))
rows.sort(key=lambda x: x[0])
import csv
with open(out,'w',encoding='utf-8',newline='') as f:
    w = csv.writer(f)
    w.writerow(['aic_delta','series_id','orig_aic','rerun_aic','orig_model_spec','rerun_model_spec'])
    for r in rows[:30]:
        w.writerow([r[0], r[1], r[2], r[3], json.dumps(r[4], ensure_ascii=False), json.dumps(r[5], ensure_ascii=False)])
print('wrote', out, 'entries=', len(rows))
