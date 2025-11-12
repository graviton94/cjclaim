"""Diagnose SARIMAX model explosion
Loads a model pickle (dict), maps params to ar/ma/seasonal parts, computes characteristic roots,
and replays forecasting via statsmodels to show predicted growth.
Run with: python tools\diagnose_model_explosion.py <model_pkl_path> [horizon]
"""
import sys
import os
import pickle
import numpy as np
from pathlib import Path


def compute_ar_roots(arparams):
    # arparams is array [phi1, phi2, ..., phip]
    p = len(arparams)
    if p == 0:
        return np.array([])
    coeffs = np.concatenate((-np.array(arparams[::-1]), [1.0]))
    return np.roots(coeffs)


def compute_seasonal_roots(seasonal_ar_params, s):
    # seasonal_ar_params: [Phi1, Phi2, ...] for lags s,2s,...
    P = len(seasonal_ar_params)
    if P == 0:
        return np.array([])
    # Build polynomial degree P*s: 1 - Phi1 z^s - Phi2 z^{2s} - ...
    deg = P * s
    coeffs = np.zeros(deg + 1, dtype=float)
    # coefficient for z^deg (highest power) is -Phi_P ... but easier: place -Phi1 at index 0 (z^{deg}),
    # we'll position Phi_k at power deg - k*s
    for k, Phi in enumerate(seasonal_ar_params, start=1):
        power = deg - k * s
        if power < 0:
            raise ValueError('seasonal params exceed degree')
        coeffs[power] = -Phi
    coeffs[-1] = 1.0
    return np.roots(coeffs)


def main(argv):
    if len(argv) < 2:
        print('Usage: python tools\\diagnose_model_explosion.py <model_pkl> [horizon]')
        return 2
    pkl = argv[1]
    horizon = int(argv[2]) if len(argv) > 2 else 6
    if not os.path.exists(pkl):
        print('Model file not found:', pkl)
        return 3
    with open(pkl, 'rb') as f:
        mi = pickle.load(f)
    if not isinstance(mi, dict):
        print('Expected a dict pickle with model_info; got', type(mi))
        return 4
    print('Loaded model:', pkl)
    print('series_id:', mi.get('series_id'))
    params = np.array(mi.get('params', []), dtype=float)
    print('params shape:', params.shape)

    model_spec = mi.get('model_spec', {})
    order = tuple(model_spec.get('order', (0, 0, 0)))
    seasonal_order = tuple(model_spec.get('seasonal_order', (0, 0, 0, 0)))
    p, d, q = order
    P, D, Q, s = seasonal_order
    print('order:', order, 'seasonal_order:', seasonal_order)

    # Map params according to statsmodels SARIMAX convention:
    # [ar.L1..ar.Lp, ma.L1..ma.Lq, seasonal_ar.S.L1..S.LP, seasonal_ma.S.L1..S.LQ, sigma2]
    idx = 0
    arparams = params[idx: idx + p]
    idx += p
    maparams = params[idx: idx + q]
    idx += q
    seasonal_ar = params[idx: idx + P]
    idx += P
    seasonal_ma = params[idx: idx + Q]
    idx += Q
    # remaining maybe sigma2
    remainder = params[idx:]

    print('\nMapped parameters:')
    print('  ar params:', arparams)
    print('  ma params:', maparams)
    print('  seasonal_ar params:', seasonal_ar)
    print('  seasonal_ma params:', seasonal_ma)
    print('  remainder (likely sigma2):', remainder)

    # Compute roots
    ar_roots = compute_ar_roots(arparams)
    print('\nAR roots:')
    for r in ar_roots:
        print('  root:', r, 'abs:', abs(r))

    if len(seasonal_ar) > 0:
        s_roots = compute_seasonal_roots(seasonal_ar, s)
        print('\nSeasonal AR roots (poly degree {}):'.format(len(s) if isinstance(s, (list,tuple)) else s))
        for r in s_roots:
            print('  root:', r, 'abs:', abs(r))
    else:
        s_roots = np.array([])

    # Check stationarity: all AR roots must have abs > 1
    bad = []
    for r in np.concatenate((ar_roots, s_roots)):
        if abs(r) <= 1.0 + 1e-8:
            bad.append((r, abs(r)))
    if bad:
        print('\nStationarity check: FAIL - roots inside or on unit circle (magnitude <= 1):')
        for r, m in bad:
            print('  ', r, m)
    else:
        print('\nStationarity check: PASS - all roots outside unit circle')

    # Re-run smoothing and forecast to show behaviour
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # Load training data if available from data/features/<safe>.json
        safe_name = (mi.get('series_id') or '').replace('/', '_').replace('\\','_').replace(':','_').replace('|','_')
        json_path = Path('data/features') / f"{safe_name}.json"
        y = None
        if json_path.exists():
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                jd = json.load(f)
            recs = jd.get('data', [])
            import pandas as pd
            df = pd.DataFrame(recs)
            df_train = df[(df['year'] >= 2021) & (df['year'] <= 2023)].copy()
            y = df_train['claim_count'].values
            print('\nLoaded training series length', len(y), 'last values:', y[-10:])
        else:
            print('\nTraining JSON not found at', json_path)

        model = SARIMAX(
            endog=y if y is not None else np.zeros(24),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=model_spec.get('enforce_stationarity', False),
            enforce_invertibility=model_spec.get('enforce_invertibility', False)
        )
        # Try smoothing with params
        print('\nAttempting to smooth model with saved params (this replicates forecasting step)...')
        res = model.smooth(params)
        fc = res.forecast(steps=horizon)
        print('\nForecast sample (first {}):'.format(min(10, len(fc))))
        for i, v in enumerate(fc[:10], start=1):
            print(f'  step {i}: {v}')
    except Exception as e:
        print('\nError during smoothing/forecast:', e)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
