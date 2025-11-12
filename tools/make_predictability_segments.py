"""Create predictability segments (predictable vs uncertain) from snapshot.

Writes `artifacts/predictability_segments.json` with keys `predictable` and `uncertain`.

Usage:
  python tools/make_predictability_segments.py \
      --snapshot artifacts/trained_models_v3.json \
      --exempt artifacts/exemptions.json \
      --stab artifacts/stability_report.csv \
      --out artifacts/predictability_segments.json
"""
import argparse
import json
from pathlib import Path
import sys
import math

import pandas as pd


def load_snapshot(p: Path) -> dict:
    with open(p, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def load_exemptions(p: Path) -> set:
    if not p.exists():
        return set()
    with open(p, 'r', encoding='utf-8-sig') as f:
        j = json.load(f)
    return set(j.get('exempt', []) or j.get('exemptions', []) or [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='artifacts/trained_models_v3.json')
    parser.add_argument('--exempt', default='artifacts/exemptions.json')
    parser.add_argument('--stab', default='artifacts/stability_report.csv')
    parser.add_argument('--out', default='artifacts/predictability_segments.json')
    parser.add_argument('--n-train', type=int, default=24, help='n_train_points threshold')
    parser.add_argument('--nonzero-pct', type=float, default=10.0, help='nonzero_pct threshold')
    parser.add_argument('--min-root', type=float, default=1.005, help='min abs AR-root threshold')
    parser.add_argument('--max-zero-run', type=int, default=12, help='max consecutive zeros allowed')
    args = parser.parse_args()

    snap_path = Path(args.snapshot)
    if not snap_path.exists():
        print('Snapshot not found:', snap_path, file=sys.stderr)
        raise SystemExit(2)

    snap = load_snapshot(snap_path)
    # support both top-level dict of series_models or dict mapping
    series_map = snap.get('series_models') if isinstance(snap, dict) and 'series_models' in snap else snap

    exempt = load_exemptions(Path(args.exempt))

    stab_df = None
    stab_path = Path(args.stab)
    if stab_path.exists():
        stab_df = pd.read_csv(stab_path, dtype={'series_id': str}).set_index('series_id')

    P = []
    U = []
    total = 0

    for sid, rec in series_map.items():
        total += 1
        if sid in exempt:
            U.append(sid)
            continue

        n = int(rec.get('n_train_points') or rec.get('n_train') or 0)
        nz = float(rec.get('nonzero_pct') or 0.0)
        train_var = float(rec.get('train_var') or rec.get('variance') or 0.0)
        max_zero_run = int(rec.get('max_zero_run') or rec.get('max_consecutive_zeros') or 0)

        # parse arroots from snapshot if present
        min_abs_root = None
        a = rec.get('arroots')
        if a:
            try:
                # arroots may be list of numbers or list of [real,imag]
                abs_vals = []
                for x in a:
                    if isinstance(x, (int, float)):
                        abs_vals.append(abs(float(x)))
                    elif isinstance(x, (list, tuple)) and len(x) >= 1:
                        r = float(x[0])
                        im = float(x[1]) if len(x) >= 2 else 0.0
                        abs_vals.append(abs(complex(r, im)))
                    elif isinstance(x, dict) and ('real' in x or 'r' in x):
                        r = float(x.get('real', x.get('r', 0.0)))
                        im = float(x.get('imag', x.get('i', 0.0)))
                        abs_vals.append(abs(complex(r, im)))
                if abs_vals:
                    min_abs_root = min(abs_vals)
            except Exception:
                min_abs_root = None

        # fallback to stability report status if needed
        stable_flag = False
        if stab_df is not None and sid in stab_df.index:
            st = stab_df.loc[sid].to_dict()
            status = str(st.get('status') or '').lower()
            # stability report may include max_abs_arroot (the largest absolute root)
            # we prefer min_abs_root computed above, but if not available, use status
            if min_abs_root is None:
                if status == 'stable':
                    stable_flag = True
            else:
                stable_flag = (min_abs_root > args.min_root)
        else:
            if min_abs_root is not None:
                stable_flag = (min_abs_root > args.min_root)

        aic = rec.get('aic')

        cond = (
            (n >= args.n_train)
            and (nz >= args.nonzero_pct)
            and (train_var > 0)
            and (max_zero_run < args.max_zero_run)
            and (aic not in (None, '', 'NA'))
            and stable_flag
        )

        if cond:
            P.append(sid)
        else:
            U.append(sid)

    out = {'predictable': P, 'uncertain': U}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print({'P': len(P), 'U': len(U), 'total': total})


if __name__ == '__main__':
    main()
