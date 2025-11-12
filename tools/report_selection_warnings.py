"""Summarize selection_warnings from trained_models snapshots into CSV for review.

Reads artifacts/trained_models_v3.json (or trained_models.json) and writes
artifacts/selection_warnings_summary.csv containing series_id, message, and
candidate_summaries. The CSV is written with utf-8-sig to be friendly on Windows.
"""
import json
from pathlib import Path
import csv


def main():
    root = Path(__file__).resolve().parents[1]
    art = root / 'artifacts'
    v3 = art / 'trained_models_v3.json'
    v1 = art / 'trained_models.json'
    snap = v3 if v3.exists() else v1
    if not snap or not snap.exists():
        print('No trained_models snapshot found in artifacts/')
        return 1

    j = json.loads(snap.read_text(encoding='utf-8'))
    series = j.get('series_models', {})
    rows = []
    for sid, info in series.items():
        sw = info.get('selection_warnings') or {}
        # older snapshots stored selection_warnings at top-level artifacts; handle both
        # if present as dict with sid keys
        if isinstance(sw, dict) and sw:
            # record message and candidates if provided
            rows.append({'series_id': sid, 'message': sw.get('message', ''), 'candidates': sw.get('candidates', '')})
        else:
            # check top-level selection_warnings path
            pass

    out = art / 'selection_warnings_summary.csv'
    keys = ['series_id', 'message', 'candidates']
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print('Wrote selection warnings summary:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
