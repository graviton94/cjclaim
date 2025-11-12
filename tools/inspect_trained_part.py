import json
from pathlib import Path

def main():
    p = Path('artifacts/trained_models_part.json')
    if not p.exists():
        print('Missing', p); return 2
    j = json.loads(p.read_text(encoding='utf-8'))
    s = j.get('series_models', {})
    print('series_models count:', len(s))
    for i, (sid, info) in enumerate(s.items()):
        if i >= 10:
            break
        print(i+1, sid, info.get('model_spec'), 'aic=', info.get('aic'))

if __name__ == '__main__':
    main()
