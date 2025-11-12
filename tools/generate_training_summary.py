"""Generate a small training summary JSON from artifacts/stability_report.csv.

Writes: artifacts/training_summary.json

Usage: python tools/generate_training_summary.py
"""
from __future__ import annotations
import ast
import json
import os
from typing import Any

import pandas as pd


def parse_model_spec(spec_str: Any) -> dict | None:
    if not spec_str or (isinstance(spec_str, float) and pd.isna(spec_str)):
        return None
    try:
        # specs in CSV are single-quoted dicts, safe to literal_eval
        return ast.literal_eval(spec_str)
    except Exception:
        return None


def main():
    csv_path = os.path.join("artifacts", "stability_report.csv")
    out_path = os.path.join("artifacts", "training_summary.json")

    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)

    total_series = len(df)

    # Normalize columns
    df['error'] = df.get('error')
    df['model_spec'] = df.get('model_spec')
    df['n_train'] = pd.to_numeric(df.get('n_train'), errors='coerce')
    df['stable'] = df.get('stable')

    too_short_count = int(df['error'].fillna('').str.strip().eq('too_short').sum())
    fit_failed_count = int(df['error'].fillna('').str.startswith('fit_failed').sum())

    # Trained where model_spec exists (non-null/empty)
    trained_mask = df['model_spec'].notna() & (df['model_spec'].str.strip() != '')
    trained_count = int(trained_mask.sum())

    # Parse seasonal orders
    seasonal_count = 0
    for s in df.loc[trained_mask, 'model_spec'].values:
        spec = parse_model_spec(s)
        if spec and 'seasonal_order' in spec:
            so = spec.get('seasonal_order')
            # treat any non-zero entry as seasonal
            try:
                if any(int(x) != 0 for x in so):
                    seasonal_count += 1
            except Exception:
                # ignore parse errors
                pass

    # AR-root stability counts among trained
    stable_vals = df.loc[trained_mask, 'stable'].fillna('')
    stable_true = int((stable_vals == 'True').sum())
    stable_false = int((stable_vals == 'False').sum())

    n_train_series = df['n_train'].dropna().astype(int)
    n_train_stats = {
        'median': int(n_train_series.median()) if not n_train_series.empty else None,
        'min': int(n_train_series.min()) if not n_train_series.empty else None,
        'max': int(n_train_series.max()) if not n_train_series.empty else None,
    }

    summary = {
        'total_series': int(total_series),
        'trained_count': trained_count,
        'untrained_count': int(total_series - trained_count),
        'too_short_count': too_short_count,
        'fit_failed_count': fit_failed_count,
        'seasonal_count': seasonal_count,
        'seasonal_ratio': float(seasonal_count / trained_count) if trained_count else 0.0,
        'arroot_stable_count': stable_true,
        'arroot_unstable_count': stable_false,
        'n_train_stats': n_train_stats,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
