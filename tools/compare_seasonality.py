"""Compare seasonal models in trained_models_v3.json vs stability_report.csv and print mismatches.
Run: python tools/compare_seasonality.py
"""
from pathlib import Path
import json, ast, csv


def main():
    base = Path(__file__).resolve().parents[1]
    trained_p = base / 'artifacts' / 'trained_models_v3.json'
    csv_p = base / 'artifacts' / 'stability_report.csv'
    if not trained_p.exists():
        print('Missing', trained_p); return
    if not csv_p.exists():
        print('Missing', csv_p); return

    obj = json.loads(trained_p.read_text(encoding='utf-8'))
    series = obj.get('series_models', {})
    seasonal_trained = [k for k, v in series.items() if isinstance(v.get('model_spec'), dict) and any(int(x) != 0 for x in v['model_spec'].get('seasonal_order', []))]

    print('seasonal_in_trained:', len(seasonal_trained))

    csv_map = {}
    with csv_p.open(encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            csv_map[row['series_id']] = row

    mismatches = []
    for sid in seasonal_trained:
        v = series[sid]
        json_spec = v.get('model_spec')
        row = csv_map.get(sid)
        csv_spec = row.get('model_spec') if row else None
        mismatches.append({'series_id': sid, 'json_seasonal': json_spec.get('seasonal_order') if json_spec else None, 'csv_model_spec': csv_spec})

    if not mismatches:
        print('No seasonal-trained series found.')
        return

    print('\nMismatches (seasonal in trained_models_v3):')
    for m in mismatches:
        print('\n---')
        print('series_id:', m['series_id'])
        print('json seasonal_order:', m['json_seasonal'])
        print('csv model_spec:', m['csv_model_spec'])


if __name__ == '__main__':
    main()
