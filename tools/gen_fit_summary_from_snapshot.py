#!/usr/bin/env python3
"""
Generate artifacts/fit_summary_log.json from available trained snapshot.

Search order:
 - artifacts/trained_models.json
 - artifacts/trained_models_part_normalized.json
 - artifacts/trained_models_part.json

Also prints simple metrics: total, seasonal_count, seasonal_ratio, missing_aic_count, missing_aic_ratio
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Set

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

REQUIRED_FIELDS = ["model_spec", "aic"]


def load_snapshot(p: str) -> Dict[str, Any]:
    if not os.path.exists(p):
        raise SystemExit(f"Snapshot not found: {p}")
    with open(p, 'r', encoding='utf-8-sig') as f:
        d = json.load(f)
    return d.get('series_models', d)


def load_exemptions(path: str | None) -> Set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8-sig') as f:
        x = json.load(f)
    if isinstance(x, dict) and 'exempt' in x:
        x = x['exempt']
    return set(x)


def is_seasonal(seasonal_order) -> bool:
    if seasonal_order is None:
        return False
    try:
        return any(int(x) != 0 for x in seasonal_order)
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', default=str(ART / 'trained_models_v3.json'),
                   help='Path to trained snapshot (default: artifacts/trained_models_v3.json)')
    p.add_argument('--exclude', default=None,
                   help='Exemption JSON file (array or {exempt:[...]}) to exclude from denominator')
    args = p.parse_args()

    snap = load_snapshot(args.snapshot)
    if not snap or len(snap) == 0:
        print(f"ERROR: empty snapshot -> {os.path.abspath(args.snapshot)}")
        raise SystemExit(2)

    exempt = load_exemptions(args.exclude)

    rows = []
    for sid, rec in snap.items():
        if sid in exempt:
            continue
        aic = rec.get('aic', None)
        spec = (rec.get('model_spec') or {})
        seas = spec.get('seasonal_order', (0, 0, 0, 0))
        n_train = rec.get('n_train_points') or rec.get('n_train') or None
        arroots = rec.get('arroots')
        rows.append({'series_id': sid, 'aic': aic, 'seasonal_order': seas, 'n_train_points': n_train, 'arroots': arroots})

    outp = ART / "fit_summary_log.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    total = len(rows)
    seasonal = sum(1 for r in rows if is_seasonal(r.get('seasonal_order')))
    missing = sum(1 for r in rows if (r.get('aic') in (None, '', 'NA')))

    metrics = {
        'total': total,
        'seasonal_count': seasonal,
        'seasonal_ratio': round(seasonal / max(1, total), 4),
        'missing_aic_count': missing,
        'missing_aic_ratio': round(missing / max(1, total), 4),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
