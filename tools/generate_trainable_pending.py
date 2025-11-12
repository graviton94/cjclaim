#!/usr/bin/env python3
"""
Generate trainable pending list and untrained_series grouping.

Writes:
 - artifacts/trainable_pending.json  (list of series meeting pending thresholds)
 - artifacts/untrained_series.json   (summary with trainable and not_trainable lists)

Usage:
  python tools/generate_trainable_pending.py --parquet data/curated/claims_monthly.parquet --trained artifacts/trained_models_part_normalized.json

Thresholds (Pending -> trainable):
 - n_months >= 24
 - zero_fraction <= 0.9
 - variance > 0
 - max_zero_run < 12

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def max_zero_run(arr) -> int:
    # arr is a 1d iterable of numbers; treat zeros exactly equal to 0
    max_run = 0
    cur = 0
    for v in arr:
        if v == 0:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


def analyze_series(sdf: pd.DataFrame) -> dict[str, Any]:
    # sdf expected to have year, month, claim_count
    y = sdf.sort_values(['year', 'month'])['claim_count'].dropna().astype(float).to_numpy()
    n = int(len(y))
    if n == 0:
        return dict(n=0, zero_frac=1.0, var=0.0, max_zero_run=0)
    zero_frac = float((y == 0).sum()) / n
    var = float(y.var())
    mzr = int(max_zero_run(y))
    return dict(n=n, zero_frac=zero_frac, var=var, max_zero_run=mzr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', required=True)
    p.add_argument('--trained', required=True)
    args = p.parse_args()

    pq = Path(args.parquet)
    if not pq.exists():
        raise SystemExit(f"Parquet not found: {pq}")

    trained_path = Path(args.trained)
    if not trained_path.exists():
        raise SystemExit(f"Trained snapshot not found: {trained_path}")

    df = pd.read_parquet(pq)
    if 'series_id' not in df.columns:
        raise SystemExit('parquet missing series_id column')
    if 'claim_count' not in df.columns:
        raise SystemExit('parquet missing claim_count column')
    series_list = sorted(df['series_id'].unique())

    with trained_path.open('r', encoding='utf-8') as f:
        trained = json.load(f)
    trained_keys = set(trained.get('series_models', {}).keys())

    untrained = [s for s in series_list if s not in trained_keys]

    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    trainable = []
    not_trainable = []

    for s in untrained:
        sdf = df[df['series_id'] == s][['year', 'month', 'claim_count']]
        stats = analyze_series(sdf)
        # thresholds
        if stats['n'] >= 24 and stats['zero_frac'] <= 0.9 and stats['var'] > 0 and stats['max_zero_run'] < 12:
            trainable.append({'series_id': s, 'n': stats['n'], 'zero_frac': stats['zero_frac'], 'var': stats['var'], 'max_zero_run': stats['max_zero_run']})
        else:
            not_trainable.append({'series_id': s, 'n': stats['n'], 'zero_frac': stats['zero_frac'], 'var': stats['var'], 'max_zero_run': stats['max_zero_run']})

    trainable_path = artifacts_dir / 'trainable_pending.json'
    with trainable_path.open('w', encoding='utf-8') as f:
        json.dump({'trainable_count': len(trainable), 'trainable': trainable}, f, ensure_ascii=False, indent=2)

    untrained_path = artifacts_dir / 'untrained_series.json'
    with untrained_path.open('w', encoding='utf-8') as f:
        json.dump({'untrained_count': len(untrained), 'trainable_count': len(trainable), 'trainable': [t['series_id'] for t in trainable], 'not_trainable': [t['series_id'] for t in not_trainable]}, f, ensure_ascii=False, indent=2)

    print(f"Total series: {len(series_list)}")
    print(f"Trained snapshot series: {len(trained_keys)}")
    print(f"Untrained series: {len(untrained)}")
    print(f"Trainable (pending) series: {len(trainable)} (wrote {trainable_path})")
    print(f"Not trainable (kept pending/normal): {len(not_trainable)} (wrote {untrained_path})")


if __name__ == '__main__':
    main()
