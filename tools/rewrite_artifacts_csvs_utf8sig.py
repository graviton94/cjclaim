"""Rewrite all CSV files in artifacts/ to UTF-8 with BOM (utf-8-sig).

This is a small maintenance utility you can run once to overwrite existing
CSV artifacts (for example `artifacts/arroot_filter_summary.csv`,
`artifacts/top20_arroot_summary.csv`, etc.) so they include a UTF-8 BOM.

Usage:
  python tools/rewrite_artifacts_csvs_utf8sig.py

Behavior:
 - For each .csv file in artifacts/, try to decode as UTF-8. If decoding fails,
   fall back to latin-1 to avoid crashing and preserve bytes.
 - Overwrite the file with encoding='utf-8-sig' (adds BOM) so Excel on Windows
   and PowerShell display non-ASCII text correctly.
"""
from pathlib import Path
import sys


def rewrite_file(p: Path):
    try:
        b = p.read_bytes()
        try:
            text = b.decode('utf-8')
            src_enc = 'utf-8'
        except Exception:
            # fallback to latin-1 to preserve bytes if file isn't valid utf-8
            text = b.decode('latin-1')
            src_enc = 'latin-1'
        # write back with BOM (utf-8-sig)
        p.write_text(text, encoding='utf-8-sig')
        print(f"Rewrote {p} (from {src_enc} -> utf-8-sig)")
    except Exception as e:
        print(f"Failed to rewrite {p}: {e}")


def main():
    root = Path(__file__).resolve().parents[1]
    art = root / 'artifacts'
    if not art.exists():
        print('artifacts/ directory not found; nothing to do')
        return 1
    csvs = sorted(art.glob('*.csv'))
    if not csvs:
        print('No CSV files found in artifacts/')
        return 0
    for p in csvs:
        rewrite_file(p)
    print('Done. Please verify files in artifacts/ (open in Excel/PowerShell).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
