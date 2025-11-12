import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'artifacts' / 'seasonal_strength.json'
if not p.exists():
    print('missing', p)
    raise SystemExit(1)

with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

# sort by seasonal_score desc
data_sorted = sorted(data, key=lambda x: x.get('seasonal_score', 0), reverse=True)

print('total_series:', len(data))

# top 10
print('\nTop 10 seasonal_score:')
for row in data_sorted[:10]:
    print(row['series_id'][:80].ljust(80), 'n_train=', row.get('n_train'), 'score=', round(row.get('seasonal_score',0), 4), 'candidate=', row.get('candidate'))

# counts by thresholds
for th in [0.8, 0.6, 0.5, 0.4, 0.3]:
    cnt = sum(1 for r in data if r.get('seasonal_score', 0) >= th and r.get('n_train',0) >= 36)
    print(f'count score>={th} and n_train>=36: {cnt}')

# distribution of n_train
from collections import Counter
cnts = Counter(r.get('n_train', 0) for r in data)
print('\nTop n_train counts:')
for n, c in cnts.most_common()[:10]:
    print(n, c)

# best candidate with n_train>=36 if any
candidates = [r for r in data_sorted if r.get('n_train',0) >= 36]
print('\nTop 5 series with n_train>=36:')
for r in candidates[:5]:
    print(r['series_id'][:80].ljust(80), 'n_train=', r.get('n_train'), 'score=', round(r.get('seasonal_score',0),4))
