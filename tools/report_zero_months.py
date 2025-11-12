import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
FILLED = OUT / 'features_filled'
OUT_CSV = OUT / 'zero_months_report.csv'


def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    files = list(FILLED.glob('*.json'))
    rows = []
    for f in files:
        try:
            feat = load_json(f)
        except Exception:
            continue
        data = feat.get('data', [])
        n_total = len(data)
        n_nonzero = sum(1 for r in data if r.get('claim_count', 0) != 0)
        n_zero = n_total - n_nonzero
        pct_nonzero = (n_nonzero / n_total * 100) if n_total else 0
        rows.append({
            'series_id': Path(f).stem,
            'n_total': n_total,
            'n_nonzero': n_nonzero,
            'n_zero': n_zero,
            'pct_nonzero': round(pct_nonzero, 2)
        })

    df = pd.DataFrame(rows).sort_values(['pct_nonzero', 'n_total'], ascending=[True, False])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_CSV} ({len(df)} rows)')


if __name__ == '__main__':
    main()
