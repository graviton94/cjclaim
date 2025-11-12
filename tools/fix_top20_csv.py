#!/usr/bin/env python3
"""Fix top20 AR-root CSV encoding by parsing the Markdown summary and writing a UTF-8-sig CSV.

Usage:
  python tools/fix_top20_csv.py

This reads artifacts/top20_arroot_summary.md and writes artifacts/top20_arroot_summary_fixed.csv (UTF-8 with BOM).
"""
from pathlib import Path
import csv
import sys

md_path = Path('artifacts/top20_arroot_summary.md')
out_path = Path('artifacts/top20_arroot_summary_fixed.csv')

if not md_path.exists():
    print(f'MD file not found: {md_path}', file=sys.stderr)
    raise SystemExit(1)

text = md_path.read_text(encoding='utf-8')
lines = text.splitlines()

# find table header
start = None
for i, line in enumerate(lines):
    if line.strip().startswith('| rank '):
        start = i
        break
if start is None:
    print('Table header not found in md file', file=sys.stderr)
    raise SystemExit(1)

# header line is at start, separator at start+1, data from start+2 until a line that doesn't start with '|'
header_line = lines[start]
sep_line = lines[start+1] if start+1 < len(lines) else ''
rows = []
for line in lines[start+2:]:
    if not line.strip().startswith('|'):
        break
    # split on | and strip whitespace
    parts = [p.strip() for p in line.split('|')]
    # parts contains leading/trailing empty strings due to surrounding pipes
    # find non-empty columns between
    # We'll take columns 1..7 (common case). If more/less, adapt.
    if len(parts) < 8:
        # pad
        parts += ['']*(8-len(parts))
    # take columns 1..7
    row = parts[1:8]
    rows.append(row)

# header names mapping (from md header)
# | rank | series_id | max_abs_arroot | order | seasonal_order | converged | plot |
headers = ['rank','series_id','max_abs_arroot','order','seasonal_order','converged','plot']

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for r in rows:
        writer.writerow(r)

print(f'Wrote fixed CSV: {out_path} (UTF-8 with BOM) - rows={len(rows)}')
