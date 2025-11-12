#!/usr/bin/env python3
"""
Evaluate gates G1/G2/G3 and produce retry_list_remaining.json.
- Reads artifacts/fit_summary_log.json and artifacts/stability_report.csv (if present)
- Writes artifacts/gates_report.json and artifacts/retry_list_remaining.json
"""
from pathlib import Path
import json
import csv
import argparse

ART = Path(__file__).resolve().parents[1] / 'artifacts'

parser = argparse.ArgumentParser()
parser.add_argument('--fit-summary', default=str(ART / 'fit_summary_log.json'))
parser.add_argument('--stability-csv', default=str(ART / 'stability_report.csv'))
parser.add_argument('--exclude', default=None, help='Exemptions JSON file to exclude from denominator and retry lists')
parser.add_argument('--out-report', default=str(ART / 'gates_report.json'))
parser.add_argument('--out-retry', default=str(ART / 'retry_list_remaining.json'))
args = parser.parse_args()

fit_path = Path(args.fit_summary)
stab_path = Path(args.stability_csv)
report_path = Path(args.out_report)
retry_out = Path(args.out_retry)

if not fit_path.exists():
    raise SystemExit(f"Missing fit summary: {fit_path}")

with fit_path.open('r', encoding='utf-8-sig') as f:
    rows = json.load(f)

total = len(rows)

# load exemptions
exempt = set()
if args.exclude:
    ex_p = Path(args.exclude)
    if ex_p.exists():
        j = json.loads(ex_p.read_text(encoding='utf-8-sig'))
        if isinstance(j, dict) and 'exempt' in j:
            exempt = set(j['exempt'])
        elif isinstance(j, list):
            exempt = set(j)

# filter rows for denominator (exclude exemptions)
rows_filtered = [r for r in rows if r['series_id'] not in exempt]
total_filtered = len(rows_filtered)

missing_aic = [r for r in rows_filtered if r.get('aic') is None]
missing_count = len(missing_aic)
seasonal_count = sum(1 for r in rows_filtered if r.get('seasonal_order') and any(int(x) != 0 for x in r.get('seasonal_order')))
seasonal_ratio = seasonal_count / total_filtered if total_filtered else 0.0
missing_ratio = missing_count / total_filtered if total_filtered else 0.0

# prepare retry list: only series with n_train_points >=36
candidates = [r for r in missing_aic if (r.get('n_train_points') or 0) >= 36]
candidates_sorted = sorted(candidates, key=lambda r: r.get('n_train_points', 0), reverse=True)
retry_series = [r['series_id'] for r in candidates_sorted]

# read stability report if present
stable_count = None
total_stab_checked = None
unstable_series = []
if stab_path.exists():
    with stab_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows_stab = [r for r in reader if r.get('series_id') not in exempt]
    total_stab_checked = len(rows_stab)
    # accept either 'status' == 'stable' or stable=='True'; treat 'unknown' as unknown and exclude from denom
    def is_stable(rec):
        st = rec.get('status') or rec.get('stable')
        return str(st).lower() in ('stable', 'true', '1')

    def is_unknown(rec):
        st = rec.get('status') or rec.get('stable')
        return str(st).lower() in ('unknown', '')

    # only consider known rows (not unknown) for stable/unstable denom
    known_rows = [r for r in rows_stab if not is_unknown(r)]
    known_count = len(known_rows)
    stable_count = sum(1 for r in known_rows if is_stable(r))
    unstable_series = [r.get('series_id') for r in known_rows if not is_stable(r)]
    unknown_count = total_stab_checked - known_count

report = {
    'total_series': total_filtered,
    'seasonal_count': seasonal_count,
    'seasonal_ratio': round(seasonal_ratio, 4),
    'missing_aic_count': missing_count,
    'missing_aic_ratio': round(missing_ratio, 4),
    'retry_candidates_total': len(retry_series),
    'retry_list_path': str(retry_out),
}
if total_stab_checked is not None:
    report.update({'stability_checked': total_stab_checked, 'stable_count': stable_count, 'unstable_count': total_stab_checked - stable_count})

report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open('w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# write retry list
retry_out.parent.mkdir(parents=True, exist_ok=True)
with retry_out.open('w', encoding='utf-8') as f:
    json.dump(retry_series, f, ensure_ascii=False, indent=2)

print('Gates report written:', report_path)
print('Retry list written:', retry_out)
print('Summary:')
print(json.dumps(report, ensure_ascii=False, indent=2))
