#!/usr/bin/env python3
"""
Simple inspector for artifacts/trained_models_v3.json (UTF-8 safe).
Prints series_models length, metadata keys, and first 3 series ids.
"""
import json
from pathlib import Path
p = Path(__file__).resolve().parent.parent / 'artifacts' / 'trained_models_v3.json'
if not p.exists():
    print(f'NOT_FOUND:{p}')
    raise SystemExit(2)
with p.open('r', encoding='utf-8') as f:
    data = json.load(f)
series = data.get('series_models', {}) or {}
print('v3_series_models_len', len(series))
meta = data.get('metadata', {}) or {}
print('meta_keys', list(meta.keys()))
# print first 5 series ids
sids = list(series.keys())
print('sample_series_ids_count', min(5, len(sids)))
for sid in sids[:5]:
    print('-', sid)
