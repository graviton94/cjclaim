import json, argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default='artifacts/trained_models_v3.json')
    ap.add_argument('--exclude', default='artifacts/exemptions.json')
    ap.add_argument('--out', default='artifacts/retry_list_focus.json')
    args = ap.parse_args()

    p = Path(args.snapshot)
    if not p.exists():
        raise SystemExit(f"Snapshot not found: {p}")
    d = json.load(open(p, 'r', encoding='utf-8-sig'))
    m = d.get('series_models', d)

    ex = set()
    if args.exclude:
        e_p = Path(args.exclude)
        if e_p.exists():
            e = json.load(open(e_p, 'r', encoding='utf-8-sig'))
            if isinstance(e, dict) and 'exempt' in e:
                ex = set(e['exempt'])
            elif isinstance(e, list):
                ex = set(e)

    retry = [sid for sid, rec in m.items() if (sid not in ex) and (rec.get('aic') in (None, '', 'NA'))]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(retry, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote retry list: {outp} entries={len(retry)}')


if __name__ == '__main__':
    main()
