import json
from pathlib import Path

art = Path('artifacts')
# load b1 and b2
b1 = []
b2 = []
for name in ['retry_list_shell_b1.json','retry_list_shell_b2.json']:
    p=art/name
    if p.exists():
        try:
            data=json.loads(p.read_text(encoding='utf-8'))
            if isinstance(data, dict) and 'untrained_series' in data:
                data=data['untrained_series']
            globals()[name.replace('.json','')] = data
        except Exception:
            globals()[name.replace('.json','')] = []
    else:
        globals()[name.replace('.json','')] = []
b1 = globals().get('retry_list_shell_b1', [])
b2 = globals().get('retry_list_shell_b2', [])

candidates = []
def _load_json(pth):
    try:
        return json.loads(pth.read_text(encoding='utf-8'))
    except Exception:
        return json.loads(pth.read_text(encoding='utf-8-sig'))

# add retry_list_missing_aic_top120
p = art / 'retry_list_missing_aic_top120.json'
if p.exists():
    candidates += _load_json(p)
# add retry_list_sample120
p = art / 'retry_list_sample120.json'
if p.exists():
    data = _load_json(p)
    if isinstance(data, dict) and 'untrained_series' in data:
        candidates += data['untrained_series']
    else:
        candidates += data
# add retry_list_sample10 if present
p = art / 'retry_list_sample10.json'
if p.exists():
    data = _load_json(p)
    if isinstance(data, dict) and 'untrained_series' in data:
        candidates += data['untrained_series']
    else:
        candidates += data
# add retry_list_shell_top200
p = art / 'retry_list_shell_top200.json'
if p.exists():
    data = json.loads(p.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'untrained_series' in data:
        candidates += data['untrained_series']
    else:
        candidates += data

# dedupe preserving order
seen = set()
uniq = []
for s in candidates:
    if s not in seen:
        seen.add(s)
        uniq.append(s)

# remove already attempted in b1/b2
for s in list(uniq):
    if s in b1 or s in b2:
        uniq.remove(s)

# prepare b3,b4,b5 (up to 100 each)
b3 = uniq[:100]
b4 = uniq[100:200]
b5 = uniq[200:300]

# write files
(art / 'retry_list_shell_b3.json').write_text(json.dumps(b3, ensure_ascii=False, indent=2), encoding='utf-8')
(art / 'retry_list_shell_b4.json').write_text(json.dumps(b4, ensure_ascii=False, indent=2), encoding='utf-8')
(art / 'retry_list_shell_b5.json').write_text(json.dumps(b5, ensure_ascii=False, indent=2), encoding='utf-8')
print('Wrote b3/b4/b5 counts:', len(b3), len(b4), len(b5))
