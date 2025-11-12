"""Recompute training_summary.json from artifacts/trained_models.json.

This writes a small summary that mirrors the fields produced by the main
pipeline but is tolerant when `series_models` is empty.

Usage: python tools/recompute_training_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ART = Path("artifacts")
TRAINED = ART / "trained_models.json"
OUT = ART / "training_summary.json"

if not TRAINED.exists():
    raise SystemExit(f"Missing {TRAINED}")

with open(TRAINED, 'r', encoding='utf-8') as f:
    data = json.load(f)

meta = data.get('metadata', {})
series = data.get('series_models', {}) or {}

nonzero_vals = [float(m.get('nonzero_pct', 0.0)) for m in series.values()]
avg_nonzero = float(np.mean(nonzero_vals)) if len(nonzero_vals) > 0 else 0.0

summary = {
    'train_date': meta.get('train_date'),
    'n_series': len(series),
    'n_samples': meta.get('n_samples', 0),
    'train_period': meta.get('train_period', f"until {meta.get('train_until_year')}-{meta.get('train_until_month'):02d}"),
    'forecast_horizon': meta.get('forecast_horizon'),
    'model_distribution': {},
    'avg_nonzero_pct': avg_nonzero,
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Wrote summary to {OUT}")
print(json.dumps(summary, indent=2, ensure_ascii=False))
