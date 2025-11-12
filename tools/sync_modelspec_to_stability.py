"""Sync model_spec entries from artifacts/trained_models_v3.json into artifacts/stability_report.csv.

Behavior:
 - Backups the original CSV to artifacts/stability_report.csv.bak
 - For each series_id present in both the JSON snapshot and CSV, replaces the CSV `model_spec` field
   with the JSON `model_spec` (as a Python-style literal, matching existing CSV format).
 - Prints a short summary of how many rows were updated.

Usage: python tools/sync_modelspec_to_stability.py
"""
from pathlib import Path
import json, csv, ast


def main():
    base = Path(__file__).resolve().parents[1]
    trained_p = base / 'artifacts' / 'trained_models_v3.json'
    csv_p = base / 'artifacts' / 'stability_report.csv'
    bak_p = base / 'artifacts' / 'stability_report.csv.bak'

    if not trained_p.exists():
        print('Missing', trained_p); return
    if not csv_p.exists():
        print('Missing', csv_p); return

    print('Loading trained models...')
    trained = json.loads(trained_p.read_text(encoding='utf-8'))
    series = trained.get('series_models', {})

    # Read CSV rows
    rows = []
    with csv_p.open(encoding='utf-8', newline='') as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames
        for r in rdr:
            rows.append(r)

    # Backup
    if not bak_p.exists():
        csv_p.replace(bak_p)
        # restore the original to continue
        bak_p.replace(csv_p)

    updated = 0
    for r in rows:
        sid = r.get('series_id')
        if not sid:
            continue
        t = series.get(sid)
        if not t:
            continue
        ms = t.get('model_spec')
        if not ms:
            continue
        # Write python-literal-like representation (single quotes) to match existing CSV style
        try:
            rep = repr(ms)
        except Exception:
            rep = str(ms)
        # Only replace if different
        if (r.get('model_spec') or '').strip() != rep:
            r['model_spec'] = rep
            updated += 1

    if updated == 0:
        print('No updates required.')
        return

    # Write updated CSV (overwrite). Use utf-8-sig so Excel/PowerShell shows non-ASCII text correctly.
    with csv_p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Updated {updated} rows in {csv_p}')


if __name__ == '__main__':
    main()
