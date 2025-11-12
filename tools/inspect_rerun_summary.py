"""Summarize rerun outputs: trained_models_part.json, fit_summary_log.json, failed_series.json,
and selection_warnings_summary.csv.

Prints counts and basic diagnostics (AIC presence, seasonal vs non-seasonal specs).
"""
import json
from pathlib import Path
import csv


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    art = Path('artifacts')
    trained_p = art / 'trained_models_part.json'
    fit_log_p = art / 'fit_summary_log.json'
    failed_p = art / 'failed_series.json'
    sel_warn_p = art / 'selection_warnings_summary.csv'

    trained = load_json(trained_p) or {}
    series = trained.get('series_models', {})
    n_series = len(series)

    aic_missing = 0
    seasonal_count = 0
    nonseasonal_count = 0
    for sid, info in series.items():
        aic = info.get('aic')
        if aic is None:
            aic_missing += 1
        spec = info.get('model_spec') or {}
        seasonal = spec.get('seasonal_order')
        # seasonal may be list or tuple
        if seasonal and tuple(seasonal) != (0,0,0,0):
            seasonal_count += 1
        else:
            nonseasonal_count += 1

    fit_log = load_json(fit_log_p) or []
    fit_success = sum(1 for e in fit_log if e.get('success'))
    fit_total = len(fit_log)

    failed = load_json(failed_p) or []

    sel_warns = []
    if sel_warn_p.exists():
        try:
            with sel_warn_p.open('r', encoding='utf-8-sig', newline='') as f:
                rdr = csv.DictReader(f)
                sel_warns = list(rdr)
        except Exception:
            sel_warns = []

    print('Rerun summary')
    print('-------------')
    print('trained_models_part.json present:', trained_p.exists())
    print('series_models count:', n_series)
    print('AIC missing count:', aic_missing)
    print('seasonal_count:', seasonal_count, 'nonseasonal_count:', nonseasonal_count)
    print()
    print('fit_summary_log entries:', fit_total, 'successes:', fit_success)
    print('failed_series entries:', len(failed))
    print('selection_warnings_summary rows:', len(sel_warns))

    # show a few examples of entries with missing aic
    if aic_missing:
        print('\nExamples with missing AIC (up to 5):')
        c = 0
        for sid, info in series.items():
            if info.get('aic') is None:
                print('-', sid, info.get('model_spec'))
                c += 1
                if c >= 5:
                    break


if __name__ == '__main__':
    main()
