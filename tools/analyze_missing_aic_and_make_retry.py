#!/usr/bin/env python3
"""
Analyze fit_summary_log.json for missing AICs and create a retry list.

Outputs:
 - artifacts/missing_aic_analysis.csv  (summary rows)
 - artifacts/retry_list_missing_aic_top120.json  (series list to retry)

"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIT = ART / "fit_summary_log.json"


def load_fit_summary():
    if not FIT.exists():
        raise SystemExit(f"Fit summary not found: {FIT}")
    with FIT.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    rows = load_fit_summary()
    out_csv = ART / 'missing_aic_analysis.csv'
    analysis = []
    for r in rows:
        sid = r.get('series_id')
        aic = r.get('aic')
        seasonal = r.get('seasonal_order')
        n = r.get('n_train_points') or 0
        analysis.append({'series_id': sid, 'has_aic': aic is not None, 'aic': aic, 'n_train_points': n, 'seasonal': seasonal})

    # write csv
    with out_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['series_id','has_aic','aic','n_train_points','seasonal'])
        writer.writeheader()
        for a in analysis:
            writer.writerow(a)

    # select missing aic with n>=24, sort by n desc, take top 120
    candidates = [a for a in analysis if (not a['has_aic']) and (a['n_train_points'] and a['n_train_points'] >= 24)]
    candidates_sorted = sorted(candidates, key=lambda x: (-x['n_train_points'], x['series_id']))
    topN = candidates_sorted[:120]
    retry_path = ART / 'retry_list_missing_aic_top120.json'
    with retry_path.open('w', encoding='utf-8') as f:
        json.dump([t['series_id'] for t in topN], f, ensure_ascii=False, indent=2)

    print(f"Wrote analysis CSV: {out_csv} (total rows={len(analysis)})")
    print(f"Wrote retry list: {retry_path} (count={len(topN)})")


if __name__ == '__main__':
    main()
