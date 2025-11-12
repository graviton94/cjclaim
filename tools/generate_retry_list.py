#!/usr/bin/env python3
"""
Generate a prioritized retry list from artifacts/fit_summary_log.json.
Writes artifacts/retry_list_top120.json (array of series_id strings).
Selection: aic is missing and n_train_points >= 36, sorted by n_train_points desc.
"""
from pathlib import Path
import json
import argparse

ART = Path(__file__).resolve().parents[1] / 'artifacts'

def load_exempt(path: str | None):
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Exemptions file not found: {p}")
    j = json.loads(p.read_text(encoding='utf-8-sig'))
    if isinstance(j, dict) and 'exempt' in j:
        return set(j['exempt'])
    if isinstance(j, list):
        return set(j)
    return set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=120)
    p.add_argument('--min-n-train', type=int, default=36)
    p.add_argument('--exclude', default=None, help='Exemptions JSON to exclude from retry list')
    p.add_argument('--fit-summary', default=str(ART / 'fit_summary_log.json'))
    p.add_argument('--out', default=str(ART / 'retry_list_top120.json'))
    args = p.parse_args()

    fs = Path(args.fit_summary)
    if not fs.exists():
        raise SystemExit(f"fit_summary not found: {fs}")

    rows = json.loads(fs.read_text(encoding='utf-8-sig'))
    exempt = load_exempt(args.exclude)

    candidates = [r for r in rows if (r.get('aic') is None) and ((r.get('n_train_points') or 0) >= args.min_n_train) and (r['series_id'] not in exempt)]
    cand_sorted = sorted(candidates, key=lambda r: r.get('n_train_points', 0), reverse=True)
    top = cand_sorted[: args.top]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r['series_id'] for r in top], ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote retry list: {out} entries:{len(top)}")
    print('sample:', [r['series_id'] for r in top[:10]])


if __name__ == '__main__':
    main()
