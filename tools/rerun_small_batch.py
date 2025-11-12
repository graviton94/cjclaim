"""Rerun a small batch of untrained series with configurable retry strategy.

Usage example:
  python tools/rerun_small_batch.py \
    --input artifacts/untrained_series.json \
    --limit 600 \
    --retry-optimizers powell,bfgs,lbfgs \
    --maxiter 400

Behavior/policy:
 - If series has zero-ratio >= 0.9 -> skip and record reason
 - If n_train < 36 -> try ARIMA(1,0,0) non-seasonal with fit_with_retries
 - Else -> try seasonal candidates via fit_monthly_sarimax (which uses retries)
 - On success: add to new trained_models_v3.json (merge with existing trained_models.json)
 - Always append a fit entry into artifacts/fit_summary_log.json with keys: series_id, n_train, method, maxiter, success, error, aic, arroots
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import time
import traceback

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ensure local package imports work at runtime when run from repo root
import os
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in os.sys.path:
    os.sys.path.insert(0, str(repo_root))

from src.model_monthly import fit_monthly_sarimax, fit_with_retries, forecast_and_inverse

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--limit', type=int, default=100)
parser.add_argument('--retry-optimizers', default='powell,bfgs,lbfgs')
parser.add_argument('--maxiter', type=int, default=200)
parser.add_argument('--prefer-seasonal', action='store_true', help='Prefer seasonal fit even when n_train < 36')
parser.add_argument('--delta-aic', type=float, default=0.0, help='If >0, allow seasonal when seasonal AIC <= nonseasonal AIC + delta')
parser.add_argument('--output', default='artifacts/trained_models_v3.json')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

in_path = Path(args.input)
if not in_path.exists():
    raise SystemExit(f"Input not found: {in_path}")

with open(in_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

untrained = data.get('untrained_series', []) if isinstance(data, dict) else data
if not untrained:
    print('No untrained series found. Exiting.')
    raise SystemExit(0)

# load existing trained models to merge
trained_path = Path('artifacts/trained_models.json')
trained = {}
if trained_path.exists():
    with open(trained_path, 'r', encoding='utf-8') as f:
        trained = json.load(f)
else:
    trained = {'metadata': {}, 'series_models': {}}

trained_v3 = {'metadata': trained.get('metadata', {}).copy(), 'series_models': dict(trained.get('series_models', {}))}

# load parquet claims
p = Path('data/curated/claims_monthly.parquet')
if not p.exists():
    raise SystemExit('Parquet data missing')
df = pd.read_parquet(p)

optimizers = args.retry_optimizers.split(',') if args.retry_optimizers else ['powell','bfgs','lbfgs']
maxiter = args.maxiter
limit = min(args.limit, len(untrained))

fit_log = []
failed_list = []

start_all = time.time()
count_success = 0
count_skipped = 0

for i, sid in enumerate(untrained[:limit], 1):
    print(f"[{i}/{limit}] Processing {sid}...")
    try:
        df_s = df[df['series_id'] == sid][['year','month','claim_count']].copy()
        if df_s.empty:
            reason = 'no_data'
            failed_list.append({'series_id': sid, 'error': reason})
            fit_log.append({'series_id': sid, 'success': False, 'error': reason})
            count_skipped += 1
            continue
        df_s = df_s.groupby(['year','month'], as_index=False)['claim_count'].sum().sort_values(['year','month'])
        idx = pd.to_datetime(df_s[['year','month']].assign(day=1))
        y = pd.Series(df_s['claim_count'].values, index=idx).sort_index()
        # compute zero ratio
        zero_ratio = float(np.sum(y.values == 0) / len(y.values)) if len(y) > 0 else 1.0
        if zero_ratio >= 0.9:
            reason = 'high_zero_ratio'
            print(f"Skipping {sid}: zero_ratio={zero_ratio:.3f}")
            failed_list.append({'series_id': sid, 'error': reason})
            fit_log.append({'series_id': sid, 'success': False, 'error': reason, 'zero_ratio': zero_ratio})
            count_skipped += 1
            continue
        # prepare train horizon similar to pipeline: drop last 6 months
        if len(y) <= 6:
            reason = 'too_short_total'
            failed_list.append({'series_id': sid, 'error': reason})
            fit_log.append({'series_id': sid, 'success': False, 'error': reason})
            count_skipped += 1
            continue
        y_train = y[:-6]
        n_train = len(y_train)

        # choose model
        res = None
        used_method = None
        used_maxiter = None
        error = None
        aic = None
        arroots = None

        prefer_seasonal = args.prefer_seasonal
        delta_aic = float(args.delta_aic or 0.0)

        if n_train < 36 and not prefer_seasonal:
            # build ARIMA(1,0,0) non-seasonal and use fit_with_retries
            y_t = np.sqrt(y_train.astype(float))
            mod = SARIMAX(endog=y_t, order=(1,0,0), seasonal_order=(0,0,0,0), trend='n', enforce_stationarity=True, enforce_invertibility=True)
            try:
                res = fit_with_retries(mod, maxiter_list=(maxiter,), methods=optimizers)
                used_method = ','.join(optimizers)
                used_maxiter = maxiter
            except Exception as e:
                error = str(e)
        else:
            # try seasonal fits first (helper tries candidate seasonal orders)
            # If prefer_seasonal is False but n_train >=36, we also prefer seasonal.
            seasonal_res = None
            seasonal_aic = None
            try:
                seasonal_res = fit_monthly_sarimax(y_train, order=(2,0,0), seasonal_order=None)
                seasonal_aic = getattr(seasonal_res, 'aic', None)
            except Exception as e:
                seasonal_res = None
                seasonal_aic = None
                seasonal_err = str(e)

            # if seasonal fit succeeded, accept it
            if seasonal_res is not None:
                res = seasonal_res
                used_method = 'seasonal_candidates'
                used_maxiter = maxiter
            else:
                # fallback to non-seasonal ARIMA(1,0,0)
                try:
                    y_t = np.sqrt(y_train.astype(float))
                    mod = SARIMAX(endog=y_t, order=(1,0,0), seasonal_order=(0,0,0,0), trend='n', enforce_stationarity=True, enforce_invertibility=True)
                    res = fit_with_retries(mod, maxiter_list=(maxiter,), methods=optimizers)
                    used_method = ','.join(optimizers)
                    used_maxiter = maxiter
                except Exception as e:
                    error = str(e)
        if res is not None:
            # success: inverse forecast and store model summary
            aic = float(res.aic) if hasattr(res, 'aic') else None
            arroots = [float(x) for x in np.asarray(res.arroots)] if hasattr(res, 'arroots') else []
            # forecast to embed into artifact
            yhat = forecast_and_inverse(res, steps=6)
            model_info = {
                'model_type': 'sarimax_monthly',
                'model_spec': {'order': tuple(getattr(res.model, 'order', (2,0,0))), 'seasonal_order': tuple(getattr(res.model, 'seasonal_order', (1,1,1,12)))},
                'params': getattr(res, 'params', None).tolist() if hasattr(res, 'params') else None,
                'n_train_points': int(n_train),
                'forecast': {'yhat': list(map(float, yhat))},
                'aic': aic,
                'arroots': arroots
            }
            # require AIC to be present before committing to the snapshot
            if aic is None:
                reason = 'missing_aic_on_save'
                print(f"Not saving {sid}: {reason}")
                failed_list.append({'series_id': sid, 'error': reason})
                fit_log.append({'series_id': sid, 'success': False, 'error': reason, 'n_train': int(n_train)})
                count_skipped += 1
            else:
                trained_v3['series_models'][sid] = model_info
                fit_log.append({'series_id': sid, 'success': True, 'method': used_method, 'maxiter': used_maxiter, 'aic': aic, 'arroots': arroots, 'n_train': int(n_train)})
                count_success += 1
        else:
            fit_log.append({'series_id': sid, 'success': False, 'error': error, 'n_train': int(n_train) if 'n_train' in locals() else None})
            failed_list.append({'series_id': sid, 'error': error})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Unexpected error for {sid}: {e}\n{tb}")
        fit_log.append({'series_id': sid, 'success': False, 'error': str(e)})
        failed_list.append({'series_id': sid, 'error': str(e)})

    # brief sleep to avoid hammering resources
    time.sleep(0.05)

# write outputs
out_dir = Path('artifacts')
out_dir.mkdir(parents=True, exist_ok=True)
if not args.dry_run:
    out_file = Path(args.output)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(trained_v3, f, indent=2, ensure_ascii=False)
    print(f"Wrote trained models v3: {out_file}")

    # append/merge failed_series.json
    failed_path = out_dir / 'failed_series.json'
    existing_failed = []
    if failed_path.exists():
        try:
            existing_failed = json.load(open(failed_path, 'r', encoding='utf-8'))
        except Exception:
            existing_failed = []
    merged_failed = existing_failed + failed_list
    with open(failed_path, 'w', encoding='utf-8') as f:
        json.dump(merged_failed, f, indent=2, ensure_ascii=False)
    print(f"Updated failed_series.json ({len(merged_failed)} entries)")

    # write fit summary log
    fit_log_path = out_dir / 'fit_summary_log.json'
    existing_log = []
    if fit_log_path.exists():
        try:
            existing_log = json.load(open(fit_log_path, 'r', encoding='utf-8'))
        except Exception:
            existing_log = []
    all_log = existing_log + fit_log
    with open(fit_log_path, 'w', encoding='utf-8') as f:
        json.dump(all_log, f, indent=2, ensure_ascii=False)
    print(f"Wrote fit_summary_log.json ({len(all_log)} entries)")

print(f"Done. success={count_success}, skipped={count_skipped}, attempted={limit}")
print(f"Total time: {time.time()-start_all:.1f}s")
