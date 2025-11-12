#!/usr/bin/env python3
"""
prefer-new merge: incoming이 aic/params를 가지면 덮어쓰기

Usage:
  python tools/merge_trained_snapshots.py --base artifacts/trained_models.json --add artifacts/trained_models_part_retry.json --out artifacts/trained_models.json

Result: writes merged snapshot (and does a backup of base)
"""
import json, argparse
from pathlib import Path
from datetime import datetime
import shutil


REQ_FIELDS = ('aic', 'params')


def loadm(p):
    d = json.load(open(p, 'r', encoding='utf-8-sig'))
    return d.get('series_models', d), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--add', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    base_series, base_full = loadm(args.base)
    add_series, add_full = loadm(args.add)

    for sid, nv in add_series.items():
        ov = base_series.get(sid)
        choose_new = False
        if ov is None:
            choose_new = True
        else:
            aic_new = nv.get('aic') not in (None, '', 'NA')
            aic_old = ov.get('aic') not in (None, '', 'NA')
            has_params_new = bool(nv.get('params'))
            has_params_old = bool(ov.get('params'))
            # 새 결과가 더 풍부하면 덮어쓰기
            choose_new = (aic_new and not aic_old) or (has_params_new and not has_params_old)
        if choose_new:
            base_series[sid] = nv

    out = args.out or args.base
    # ensure backup of base
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    basep = Path(args.base)
    bak = basep.with_suffix(basep.suffix + f'.bak.{ts}')
    shutil.copy2(basep, bak)
    print(f'Backup written: {bak}')

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'series_models': base_series}, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'Wrote merged snapshot: {outp} ({len(base_series)} series)')


if __name__ == '__main__':
    main()
