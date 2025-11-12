"""Test harness for src.model_monthly
- creates a small synthetic monthly series
- fits model_monthly.fit_monthly_sarimax
- forecasts and checks shape and non-negativity
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from src import model_monthly as mm


def make_sample_series():
    idx = pd.date_range('2021-01-01', periods=36, freq='MS')
    # Create low-count seasonal-ish data
    seasonal = (np.sin(np.arange(len(idx)) * 2 * np.pi / 12) + 1.2) * 2.0
    noise = np.random.RandomState(0).normal(scale=0.3, size=len(idx))
    y = np.maximum((seasonal + noise).round().astype(int), 0)
    return pd.Series(y, index=idx)


def main():
    s = make_sample_series()
    try:
        res = mm.fit_monthly_sarimax(s)
        fc = mm.forecast_and_inverse(res, steps=6)
        assert len(fc) == 6
        assert (fc >= 0).all()
        print('TC-model-monthly: PASS')
        pass
    except Exception as e:
        print('TC-model-monthly: FAIL', e)
        return 2

    # short-series guard: expect failure when data < min_months
    short_idx = pd.date_range('2024-01-01', periods=12, freq='MS')
    short_s = pd.Series(np.ones(len(short_idx)), index=short_idx)
    try:
        mm.fit_monthly_sarimax(short_s)
        print('TC-model-monthly-short: FAIL (expected error)')
        return 3
    except Exception:
        print('TC-model-monthly-short: PASS')
        return 0

if __name__ == '__main__':
    raise SystemExit(main())
