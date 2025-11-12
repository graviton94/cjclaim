#!/usr/bin/env python3
"""
Analyze recent pipeline run results: counts, distributions of n_total, model types, and failures.
Writes a short summary to stdout and to artifacts/run_analysis.json
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / 'artifacts' / 'trained_models.json'
FAILED = ROOT / 'artifacts' / 'failed_series.json'
OUT = ROOT / 'artifacts' / 'run_analysis.json'

def load_json(p):
    if not p.exists():
        return None
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

data = load_json(SNAP)
failed_list = load_json(FAILED) or []

if data is None:
    print(f"Snapshot not found: {SNAP}")
    raise SystemExit(2)

series_models = data.get('series_models', {}) or {}

counts = {
    'n_series_saved': len(series_models),
    'n_failed_entries': len(failed_list) if isinstance(failed_list, list) else len(data.get('failed_series', {})),
}

dist = Counter()
model_types = Counter()
by_n_total = defaultdict(int)

for sid, info in series_models.items():
    mt = info.get('model_type', 'unknown')
    model_types[mt] += 1
    n_total = info.get('n_total') or info.get('n_train_points')
    try:
        if n_total is None:
            # try to infer from hist fields
            n_total = info.get('n_train_points')
        n_total = int(n_total) if n_total is not None else None
    except Exception:
        n_total = None
    if n_total is not None:
        by_n_total[n_total] += 1
        if n_total < 24:
            dist['lt24'] += 1
        elif n_total < 36:
            dist['24_35'] += 1
        else:
            dist['ge36'] += 1

failed_reasons = Counter()
failed_by_n = defaultdict(int)
if isinstance(failed_list, list):
    for entry in failed_list:
        msg = entry.get('error') or entry.get('message') or str(entry)
        failed_reasons[msg] += 1
else:
    # failed_series may be a dict of sid -> info
    for sid, info in (data.get('failed_series') or {}).items():
        msg = info.get('message') or info.get('error') or str(info)
        failed_reasons[msg] += 1
        n_total = info.get('n_train_points') or None
        if n_total is not None:
            failed_by_n[int(n_total)] += 1

out = {
    'counts': counts,
    'model_types': dict(model_types),
    'n_total_distribution': dict(by_n_total),
    'buckets': dict(dist),
    'failed_reasons_top': failed_reasons.most_common(20),
    'failed_by_n_sample': dict(list(failed_by_n.items())[:20])
}

with OUT.open('w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    pass
