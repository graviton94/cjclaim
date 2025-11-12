import json
import os
import math
import csv
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FEATURES = ROOT / 'data' / 'features'


def safe_fname(s):
    return ''.join(c if c.isalnum() or c in ' ._-' else '_' for c in s)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def series_from_feature(feat_json, start=None, end=None):
    # expects 'data' list with year, month, claim_count
    rows = feat_json.get('data', []) if feat_json is not None else []
    if not rows:
        # if we don't know a range, return empty series
        if start is None or end is None:
            return pd.Series(dtype=float)
        idx = pd.date_range(start=start, end=end, freq='MS')
        return pd.Series(0.0, index=idx)

    idx = [pd.Timestamp(year=r['year'], month=r['month'], day=1) for r in rows]
    vals = [r.get('claim_count', 0.0) for r in rows]
    s = pd.Series(data=vals, index=idx).sort_index()

    # if start/end provided, reindex to full monthly range and fill zeros for missing months
    if start is not None and end is not None:
        full_idx = pd.date_range(start=start, end=end, freq='MS')
        s = s.reindex(full_idx, fill_value=0.0)
    else:
        # ensure monthly freq if possible
        try:
            s = s.asfreq('MS')
        except Exception:
            pass

    return s


def plot_series(series, orig_forecast, rerun_forecast, outpath, train_until=pd.Timestamp(2023,12,1)):
    plt.figure(figsize=(6,3))
    ax = plt.gca()
    if not series.empty:
        series.plot(ax=ax, label='actual', marker='o')
    # plot forecasts
    fh = len(rerun_forecast) if rerun_forecast is not None else (len(orig_forecast) if orig_forecast is not None else 0)
    if fh:
        last = train_until + pd.offsets.MonthBegin(1)
        fidx = pd.date_range(start=last, periods=fh, freq='MS')
        if orig_forecast is not None:
            ax.plot(fidx, orig_forecast, label='orig_forecast', linestyle='--', marker='x')
        if rerun_forecast is not None:
            ax.plot(fidx, rerun_forecast, label='rerun_forecast', linestyle='-.', marker='s')
    ax.axvline(train_until, color='gray', linestyle=':', linewidth=0.8)
    ax.legend(fontsize='small')
    ax.set_ylabel('count')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    csv_path = ART / 'seasonal_flag_changes.csv'
    if not csv_path.exists():
        print('missing', csv_path)
        return

    orig_snap = ART / 'trained_models_v3.json'
    rerun_snap = ART / 'trained_models_v3_rerun_500.json'
    if not orig_snap.exists() or not rerun_snap.exists():
        print('missing snapshots in artifacts')
        return

    orig = load_json(orig_snap)
    rerun = load_json(rerun_snap)

    df = pd.read_csv(csv_path)
    out_rows = []
    plots_dir = ART / 'plots' / 'seasonal_flag_changes'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # compute abs aic delta for ranking (treat NaN as 0)
    df['aic_delta_f'] = df['aic_delta'].fillna(0).abs()
    df = df.sort_values('aic_delta_f', ascending=False)

    # load rerun metadata to determine train_until and forecast horizon
    rerun_meta = rerun.get('metadata', {})
    ru_year = int(rerun_meta.get('train_until_year', 2023))
    ru_month = int(rerun_meta.get('train_until_month', 12))
    train_until = pd.Timestamp(year=ru_year, month=ru_month, day=1)
    forecast_horizon = int(rerun_meta.get('forecast_horizon', 6))

    for idx, row in df.iterrows():
        sid = row['series_id']
        feat_file = FEATURES / f"{sid}.json"
        feat = None
        if feat_file.exists():
            try:
                feat = load_json(feat_file)
            except Exception:
                feat = None

        # determine n_train (prefer rerun snapshot, fall back to orig, else infer)
        orig_entry = orig.get('series_models', {}).get(sid)
        rerun_entry = rerun.get('series_models', {}).get(sid)
        n_train = None
        if rerun_entry is not None:
            n_train = rerun_entry.get('n_train_points')
        if n_train is None and orig_entry is not None:
            n_train = orig_entry.get('n_train_points')

        # compute total months (n_total = n_train + forecast_horizon) if available
        start = None
        end = train_until
        if n_train is not None and not pd.isna(n_train):
            try:
                n_total = int(n_train) + int(forecast_horizon)
                # start is train_until minus (n_total - 1) months
                start = (train_until - pd.DateOffset(months=(n_total - 1))).replace(day=1)
            except Exception:
                start = None
        else:
            # fallback: if feature data exists, use its min date as start
            if feat is not None and feat.get('data'):
                rows = feat.get('data')
                min_row = min(rows, key=lambda r: (r.get('year', 9999), r.get('month', 99)))
                start = pd.Timestamp(year=int(min_row['year']), month=int(min_row['month']), day=1)

        series = series_from_feature(feat, start=start, end=end) if feat is not None or (start is not None and end is not None) else pd.Series(dtype=float)

        orig_entry = orig.get('series_models', {}).get(sid)
        rerun_entry = rerun.get('series_models', {}).get(sid)

        orig_aic = row.get('orig_aic')
        rerun_aic = row.get('rerun_aic')

        # if snapshot has forecasts, use them
        orig_fc = None
        rerun_fc = None
        if orig_entry is not None:
            orig_fc = orig_entry.get('forecast', {}).get('yhat')
            if orig_fc is not None:
                orig_fc = [float(x) for x in orig_fc]
        if rerun_entry is not None:
            rerun_fc = rerun_entry.get('forecast', {}).get('yhat')
            if rerun_fc is not None:
                rerun_fc = [float(x) for x in rerun_fc]

        # use hist stats from rerun if present
        n_train = None
        if rerun_entry is not None:
            n_train = rerun_entry.get('n_train_points')
        elif orig_entry is not None:
            n_train = orig_entry.get('n_train_points')

        # compute max abs arroot if present in CSV already
        max_abs_arroot = row.get('max_abs_arroot')

        # write plot for top 20 by aic delta; otherwise only for those with big change
        plot_path = plots_dir / f"{safe_fname(sid)}.png"
        try:
            plot_series(series, orig_fc, rerun_fc, plot_path)
        except Exception as e:
            print('plot failed', sid, e)

        out_rows.append({
            'series_id': sid,
            'n_train': n_train,
            'orig_seasonal': row.get('orig_seasonal'),
            'rerun_seasonal': row.get('rerun_seasonal'),
            'orig_aic': orig_aic,
            'rerun_aic': rerun_aic,
            'aic_delta': row.get('aic_delta'),
            'max_abs_arroot': max_abs_arroot,
            'plot': str(plot_path.relative_to(ROOT))
        })

    # write validation CSV (use utf-8-sig so Excel/PowerShell on Windows shows non-ASCII correctly)
    out_csv = ART / 'validation_seasonal_flag_changes.csv'
    keys = ['series_id','n_train','orig_seasonal','rerun_seasonal','orig_aic','rerun_aic','aic_delta','max_abs_arroot','plot']
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print('wrote', out_csv)


if __name__ == '__main__':
    main()
