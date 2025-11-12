#!/usr/bin/env python3
"""
Normalize a trained snapshot to ensure each series_models entry contains
the canonical fields expected by downstream tools: model_spec.order,
model_spec.seasonal_order, aic, arroots (list), n_train_points.

Usage:
  python tools/normalize_snapshot.py artifacts/trained_models.json --out artifacts/trained_models.json

This script will back up the input file before overwriting the output path.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def normalize_entry(src: dict[str, Any]) -> dict[str, Any]:
    out = dict(src) if isinstance(src, dict) else {}
    spec = out.get('model_spec') or {}
    # ensure order and seasonal_order are present and are lists
    order = spec.get('order') if isinstance(spec, dict) else None
    seasonal = spec.get('seasonal_order') if isinstance(spec, dict) else None
    if order is None:
        spec['order'] = [0, 0, 0]
    else:
        spec['order'] = list(order)
    if seasonal is None:
        spec['seasonal_order'] = [0, 0, 0, 0]
    else:
        spec['seasonal_order'] = list(seasonal)
    out['model_spec'] = spec

    # ensure aic exists (null if unknown)
    if 'aic' not in out:
        out['aic'] = None

    # ensure arroots present as list
    ar = out.get('arroots')
    if ar is None:
        out['arroots'] = []
    else:
        try:
            out['arroots'] = [float(x) for x in ar]
        except Exception:
            out['arroots'] = []

    # ensure n_train_points present
    if 'n_train_points' not in out:
        # allow alternative keys
        out['n_train_points'] = out.get('n_train') or out.get('n_train_points') or None

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input snapshot path')
    p.add_argument('--out', required=True, help='Output snapshot path (can overwrite input)')
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"Input snapshot not found: {inp}")

    with inp.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)

    series = data.get('series_models') or {}
    new_series = {}
    for sid, info in series.items():
        new_series[sid] = normalize_entry(info)

    data['series_models'] = new_series

    # backup
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    bak = outp.with_suffix(outp.suffix + f'.bak.{ts}')
    shutil.copy2(inp, bak)
    print(f"Backup written: {bak}")

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote normalized snapshot: {outp} ({len(new_series)} series)")


if __name__ == '__main__':
    main()
