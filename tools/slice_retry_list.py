import json
from pathlib import Path
p=Path('artifacts/retry_list_shell_top200.json')
if not p.exists():
    raise SystemExit('retry_list_shell_top200.json not found')
with p.open('r', encoding='utf-8') as f:
    data=json.load(f)
# data may be list or dict with key 'untrained_series'
if isinstance(data, dict) and 'untrained_series' in data:
    series=data['untrained_series']
else:
    series=data
b1=series[:100]
b2=series[100:200]
(Path('artifacts')/ 'retry_list_shell_b1.json').write_text(json.dumps(b1, ensure_ascii=False, indent=2), encoding='utf-8')
(Path('artifacts')/ 'retry_list_shell_b2.json').write_text(json.dumps(b2, ensure_ascii=False, indent=2), encoding='utf-8')
print('Wrote artifacts/retry_list_shell_b1.json and b2.json', len(b1), len(b2))
