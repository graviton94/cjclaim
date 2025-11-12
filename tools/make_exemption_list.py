#!/usr/bin/env python3
"""
Create an exemptions JSON file listing series to exclude from gate denominators.

Logic (defaults):
 - Exempt if n_train_points < --n-min (default 6)
 - Exempt if nonzero_pct <= --nonzero-min-pct (default 5.0)

Reads `artifacts/fit_summary_log.json` and `data/curated/claims_monthly.parquet` (to compute nonzero%)
Writes `artifacts/exemptions.json` containing {"exempt": [series_ids...]}
Also writes artifacts/exemptions_details.json with per-series diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
DATA = ROOT / 'data' / 'curated' / 'claims_monthly.parquet'


def load_fit_summary(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise SystemExit(f"fit_summary not found: {p}")
    return {r['series_id']: r for r in json.loads(p.read_text(encoding='utf-8-sig'))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-min', type=int, default=6, help='Minimum n_train points to consider (default 6)')
    p.add_argument('--nonzero-min-pct', type=float, default=5.0, help='Minimum non-zero percentage (as percent) to not exempt (default 5.0)')
    p.add_argument('--out', default=str(ART / 'exemptions.json'))
    p.add_argument('--details', default=str(ART / 'exemptions_details.json'))
    args = p.parse_args()

    fit_path = ART / 'fit_summary_log.json'
    fit = load_fit_summary(fit_path)

    # Compute nonzero percentage from parquet if available
    nonzero_map = {}
    if DATA.exists():
        df = pd.read_parquet(DATA)
        # compute nonzero% per series over available months
        grp = df.groupby('series_id')['claim_count']
        nonzero_map = (grp.apply(lambda s: 100.0 * (s.astype(bool).sum() / max(1, len(s))))).to_dict()
    else:
        print(f'Warning: parquet data not found: {DATA}. Nonzero% not computed; defaulting to 100 for series with n_train>=1')

    exemptions = []
    details = []
    for sid, rec in fit.items():
        n_train = rec.get('n_train_points') or rec.get('n_train') or 0
        nonzero_pct = nonzero_map.get(sid, 100.0)
        exempt_reasons = []
        if n_train < args.n_min:
            exempt_reasons.append('n_train_too_small')
        if nonzero_pct <= args.nonzero_min_pct:
            exempt_reasons.append('low_nonzero_pct')

        if exempt_reasons:
            exemptions.append(sid)

        details.append({'series_id': sid, 'n_train_points': n_train, 'nonzero_pct': nonzero_pct, 'exempt_reasons': exempt_reasons})

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({'exempt': sorted(list(set(exemptions)))}, ensure_ascii=False, indent=2), encoding='utf-8')

    detp = Path(args.details)
    detp.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Wrote exemptions: {outp} count={len(exemptions)} details={detp}')


if __name__ == '__main__':
    main()
