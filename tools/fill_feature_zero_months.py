import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FEATURES = ROOT / 'data' / 'features'
OUT_DIR = ART / 'features_filled'


def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    snap = ART / 'trained_models_v3_rerun_500.json'
    if not snap.exists():
        print('rerun snapshot missing; using trained_models_v3.json if present')
        snap = ART / 'trained_models_v3.json'
        if not snap.exists():
            raise SystemExit('no snapshot found in artifacts')

    meta = load_json(snap).get('metadata', {})
    ru_year = int(meta.get('train_until_year', 2023))
    ru_month = int(meta.get('train_until_month', 12))
    train_until = pd.Timestamp(year=ru_year, month=ru_month, day=1)
    fh = int(meta.get('forecast_horizon', 6))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(FEATURES.glob('*.json'))
    print(f'Found {len(files)} feature files; writing filled copies to {OUT_DIR}')

    for f in files:
        try:
            feat = load_json(f)
        except Exception:
            continue

        rows = feat.get('data', [])
        if rows:
            min_row = min(rows, key=lambda r: (r.get('year', 9999), r.get('month', 99)))
            # try to find n_train from snapshot if exists
            sid = Path(f).stem
            # snapshot keys may have different quoting; try to match by starting substring
            # prefer exact match
            filled_start = pd.Timestamp(year=int(min_row['year']), month=int(min_row['month']), day=1)
        else:
            # no rows -> skip
            continue

        # attempt to find n_train in snapshot (if available) to derive start covering full training window
        # else, use filled_start from feature
        # build full index from start to train_until
        full_idx = pd.date_range(start=filled_start, end=train_until, freq='MS')
        # build a dict of existing rows
        row_map = {(r['year'], r['month']): r.get('claim_count', 0.0) for r in rows}
        new_data = []
        for d in full_idx:
            y = d.year
            m = d.month
            cnt = row_map.get((y, m), 0.0)
            new_data.append({'year': int(y), 'month': int(m), 'claim_count': float(cnt)})

        feat_filled = dict(feat)
        feat_filled['data'] = new_data
        outp = OUT_DIR / f.name
        write_json(outp, feat_filled)

    print('done')


if __name__ == '__main__':
    main()
