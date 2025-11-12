#!/usr/bin/env python3
"""
Generate diagnostics CSV for 'shell' entries in the trained snapshot.

Writes:
 - artifacts/shell_entries_diagnostics.csv

Columns include: series_id, has_params, has_aic, n_train_points, selection_loss, seasonal_order, model_spec, note

Usage: python tools/generate_shell_diagnostics.py
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SNAP_CANDIDATES = [
    ART / "trained_models.json",
    ART / "trained_models_part_normalized.json",
    ART / "trained_models_part.json",
]


def load_nonempty_snapshot() -> dict[str, Any]:
    for p in SNAP_CANDIDATES:
        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get('series_models'):
                    print(f"Using snapshot: {p}")
                    return data
    raise SystemExit('No non-empty snapshot found')


def inspect_snapshot(snap: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    series = snap.get('series_models', {}) or {}
    for sid, info in series.items():
        params = info.get('params')
        aic = info.get('aic')
        n_train = info.get('n_train_points') or info.get('n_train') or None
        sel_loss = info.get('selection_loss')
        spec = info.get('model_spec') or {}
        seasonal = spec.get('seasonal_order') if isinstance(spec, dict) else None
        note = []
        has_params = params is not None
        has_aic = aic is not None
        if not has_params:
            note.append('missing_params')
        if not has_aic:
            note.append('missing_aic')
        if isinstance(params, list) and len(params) == 0:
            note.append('empty_params')
        rows.append({
            'series_id': sid,
            'has_params': bool(has_params),
            'has_aic': bool(has_aic),
            'n_train_points': n_train,
            'selection_loss': sel_loss,
            'seasonal_order': seasonal,
            'model_spec': json.dumps(spec, ensure_ascii=False),
            'note': ';'.join(note) if note else ''
        })
    return rows


def write_csv(rows: list[dict[str, Any]]) -> Path:
    out = ART / 'shell_entries_diagnostics.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['series_id','has_params','has_aic','n_train_points','selection_loss','seasonal_order','model_spec','note']
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out


def main():
    snap = load_nonempty_snapshot()
    rows = inspect_snapshot(snap)
    out = write_csv(rows)
    total = len(rows)
    missing_params = sum(1 for r in rows if not r['has_params'])
    missing_aic = sum(1 for r in rows if not r['has_aic'])
    print(f"Wrote diagnostics CSV: {out} (total={total})")
    print(f"Missing params: {missing_params}, Missing aic: {missing_aic}")

    # also write a small prioritized retry list: missing aic & n_train>=24 sorted by n_train desc
    candidates = [r for r in rows if (not r['has_aic']) and (r['n_train_points'] and r['n_train_points'] >= 24)]
    candidates_sorted = sorted(candidates, key=lambda x: (-int(x['n_train_points'] or 0), x['series_id']))
    retry_path = ART / 'retry_list_shell_top200.json'
    with retry_path.open('w', encoding='utf-8') as f:
        json.dump([r['series_id'] for r in candidates_sorted[:200]], f, ensure_ascii=False, indent=2)
    print(f"Wrote prioritized retry list: {retry_path} (count={min(200, len(candidates_sorted))})")


if __name__ == '__main__':
    main()
