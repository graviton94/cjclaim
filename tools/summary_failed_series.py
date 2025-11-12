#!/usr/bin/env python3
"""
Summarize failed_series entries from artifacts/trained_models.json.
Writes a JSON summary to artifacts/failed_series_summary.json and prints a short table to stdout.
"""
import json
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "artifacts" / "trained_models.json"
OUT = ROOT / "artifacts" / "failed_series_summary.json"

def main():
    if not SNAP.exists():
        print(f"Snapshot not found: {SNAP}")
        return
    with SNAP.open('r', encoding='utf-8') as f:
        data = json.load(f)

    failed = data.get('failed_series') or {}
    # failed may be a dict mapping series_id -> {message:..., ...} or a list
    items = []
    if isinstance(failed, dict):
        for sid, info in failed.items():
            info = info or {}
            msg = info.get('message') or info.get('error') or str(info)
            items.append((sid, msg, info))
    elif isinstance(failed, list):
        for entry in failed:
            sid = entry.get('series_id') if isinstance(entry, dict) else None
            msg = entry.get('message') if isinstance(entry, dict) else str(entry)
            items.append((sid, msg, entry))
    else:
        print(f"Unknown failed_series type: {type(failed)}")

    counter = collections.Counter([m for (_s, m, _i) in items])
    top = counter.most_common(30)

    out = {
        'total_failed_entries': len(items),
        'top_messages': [{'message': m, 'count': c} for m, c in top],
    }
    # attach samples for top messages (up to 5 each)
    samples = {}
    for msg, _c in top:
        samples[msg] = []
    for sid, msg, info in items:
        if msg in samples and len(samples[msg]) < 5:
            samples[msg].append({'series_id': sid, 'info': info})
    out['samples'] = samples

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # print summary
    print(f"Wrote summary to: {OUT}")
    print(f"Total failed entries: {len(items)}")
    print("Top failure messages:")
    for m, c in top[:20]:
        print(f" - {c:6d}  {m}")

if __name__ == '__main__':
    main()
