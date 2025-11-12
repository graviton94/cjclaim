"""Check AR-root stability and write artifacts/stability_report.csv.

This script prefers artifacts/trained_models_v3.json when present and will
reconstruct each series, attempt a fit (using the project's fit helpers), and
record AR roots and a stability flag. It's intended as an audit/checking tool.
"""

import argparse
import json
import os
from pathlib import Path
import csv

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _load_trained_snapshot(p: str | None = None) -> tuple[Path, dict]:
    """Return (path, loaded_json) for the chosen trained_models snapshot in artifacts/."""
    trained_dir = Path("artifacts")
    if p:
        cand = trained_dir / p
        if cand.exists():
            trained_path = cand
        else:
            raise SystemExit(f"Requested snapshot not found: {cand}")
    else:
        cand_v3 = trained_dir / "trained_models_v3.json"
        cand_default = trained_dir / "trained_models.json"
        if cand_v3.exists():
            trained_path = cand_v3
        elif cand_default.exists():
            trained_path = cand_default
        else:
            raise SystemExit("No trained_models snapshot found in artifacts/")

    with open(trained_path, "r", encoding="utf-8") as f:
        trained = json.load(f)
    return trained_path, trained


def _load_curated_parquet() -> pd.DataFrame:
    p = Path("data/curated/claims_monthly.parquet")
    if not p.exists():
        raise SystemExit("data/curated/claims_monthly.parquet not found")
    df = pd.read_parquet(p)
    required_cols = {"series_id", "year", "month", "claim_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Parquet missing required columns: {sorted(missing)}")
    return df


def _prepare_series(df: pd.DataFrame, sid: str) -> pd.Series:
    """Aggregate to monthly, return pandas Series indexed by month (DatetimeIndex)."""
    df_s = df[df["series_id"] == sid][["year", "month", "claim_count"]].copy()
    if df_s.empty:
        raise ValueError("no_data")
    df_s = (
        df_s.groupby(["year", "month"], as_index=False)["claim_count"].sum()
        .sort_values(["year", "month"])
    )
    idx = pd.to_datetime(df_s[["year", "month"]].assign(day=1))
    y = pd.Series(df_s["claim_count"].to_numpy(), index=idx).sort_index()
    # Best-effort frequency inference
    try:
        y.index = pd.DatetimeIndex(y.index.values, freq=pd.infer_freq(y.index))
    except Exception:
        pass
    return y


def _fit_single(y: pd.Series):
    """Fit model according to project logic. Returns (result, n_train).

    - Hold out the last 6 months (as in pipeline)
    - For short histories (<36), use sqrt-transform + AR(1)
    - Otherwise, call fit_monthly_sarimax with order=(2,0,0)
    """
    from src.model_monthly import fit_monthly_sarimax, fit_with_retries

    if len(y) <= 6:
        raise ValueError("too_short")

    y_train = y.iloc[:-6]
    n_train = int(len(y_train))

    if n_train < 36:
        y_t = np.sqrt(y_train.astype(float))
        mod = SARIMAX(
            endog=y_t,
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        # try a couple of optimizers/iteration budgets
        res = fit_with_retries(mod, maxiter_list=(50, 200), methods=("lbfgs", "bfgs", "powell"))
    else:
        res = fit_monthly_sarimax(y_train, order=(2, 0, 0))

    return res, n_train


def _check_series_quality(y_train: pd.Series) -> tuple[bool, str]:
    """Return (qualify_full_fit, reason)."""
    if y_train.isna().any() or not np.isfinite(y_train.to_numpy()).all():
        return False, "data_error"
    if np.allclose(y_train, 0):
        return False, "all_zeros"
    if float(np.nanstd(y_train)) < 1e-8:
        return False, "low_variance"
    if len(y_train) < 36:
        return False, "too_short_for_full"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=6012)
    parser.add_argument("--snapshot", default='trained_models_v3.json', help='권위 스냅샷 사용 (파일명, artifacts/에서 찾음)')
    parser.add_argument("--threshold", type=float, default=1.01,
                        help='|root| > threshold → stable')
    parser.add_argument("--exclude-empty", action='store_true',
                        help='arroots 비어있는 항목은 unknown으로 분리 집계(분모 제외)')
    args = parser.parse_args()
    trained_path, trained = _load_trained_snapshot(args.snapshot)
    series_info = trained.get("series_models", {})
    if not series_info:
        raise SystemExit("No series_models in trained_models snapshot")

    items = sorted(series_info.items(), key=lambda kv: kv[1].get("n_train_points", 0), reverse=True)
    items = items[: args.top]

    df = _load_curated_parquet()
    rows: list[dict] = []
    st_cnt = un_cnt = uk_cnt = 0

    for sid, info in items:
        try:
            ms_repr = info.get('model_spec') or {}
            n_train = info.get('n_train_points') or info.get('n_train') or 0

            # If snapshot already contains arroots, use them directly (avoid refit)
            snap_ar = info.get('arroots')
            if snap_ar:
                parsed = []
                try:
                    for a in snap_ar:
                        if isinstance(a, (int, float)):
                            parsed.append(complex(float(a), 0.0))
                        elif isinstance(a, (list, tuple)) and len(a) >= 2:
                            parsed.append(complex(float(a[0]), float(a[1])))
                        elif isinstance(a, dict) and ('real' in a or 'r' in a):
                            r = a.get('real', a.get('r'))
                            im = a.get('imag', a.get('i', 0.0))
                            parsed.append(complex(float(r), float(im)))
                        else:
                            parsed.append(complex(float(a), 0.0))
                except Exception:
                    parsed = []

                if not parsed:
                    rows.append({
                        'series_id': sid,
                        'hurdle': 'pass',
                        'fallback_forecast': '',
                        'n_train': n_train,
                        'model_spec': ms_repr,
                        'arroots': json.dumps([], ensure_ascii=False),
                        'max_abs_arroot': '',
                        'stable': False,
                        'status': 'unknown',
                        'error': 'invalid_arroots_in_snapshot',
                    })
                    uk_cnt += 1
                    continue

                max_abs = max(abs(x) for x in parsed) if parsed else ''
                stable = all(abs(x) > args.threshold for x in parsed)
                status = 'stable' if stable else 'unstable'
                if status == 'stable':
                    st_cnt += 1
                else:
                    un_cnt += 1

                rows.append({
                    'series_id': sid,
                    'hurdle': 'pass',
                    'fallback_forecast': '',
                    'n_train': n_train,
                    'model_spec': ms_repr,
                    'arroots': json.dumps([[x.real, x.imag] for x in parsed], ensure_ascii=False),
                    'max_abs_arroot': max_abs,
                    'stable': stable,
                    'status': status,
                    'error': '',
                })
                continue

            # fallback: reconstruct series and attempt a fit (legacy behavior)
            y_df = df[df['series_id'] == sid][['year', 'month', 'claim_count']].copy()
            if y_df.empty:
                rows.append({
                    "series_id": sid,
                    "hurdle": "no_data",
                    "fallback_forecast": "",
                    "n_train": n_train,
                    "model_spec": ms_repr,
                    "arroots": json.dumps([], ensure_ascii=False),
                    "max_abs_arroot": "",
                    "stable": False,
                    "status": 'unknown',
                    "error": "no_data",
                })
                uk_cnt += 1
                continue

            y_grp = y_df.groupby(['year','month'], as_index=False)['claim_count'].sum().sort_values(['year','month'])
            idx = pd.to_datetime(y_grp[['year','month']].assign(day=1))
            y = pd.Series(y_grp['claim_count'].values, index=idx).sort_index()

            if len(y) <= 6:
                rows.append({"series_id": sid, "error": "too_short", "n_train": n_train})
                continue

            y_train = y.iloc[:-6]
            n_train = int(len(y_train))
            qualify_full, reason = _check_series_quality(y_train)

            ms = info.get("model_spec") if isinstance(info, dict) else None
            ms_repr = repr(ms) if ms is not None else ms_repr

            if not qualify_full:
                fallback = [float(np.nanmean(y_train))] * 6 if len(y_train) > 0 else []
                rows.append(
                    {
                        "series_id": sid,
                        "n_train": n_train,
                        "model_spec": ms_repr,
                        "arroots": json.dumps([], ensure_ascii=False),
                        "max_abs_arroot": "",
                        "stable": False,
                        "error": reason,
                        "hurdle": "fail",
                        "fallback_forecast": json.dumps(fallback, ensure_ascii=False),
                    }
                )
                continue

            res, _n_train = _fit_single(y)
            arroots_raw = getattr(res, "arroots", [])
            arroots = []
            try:
                for x in np.asarray(arroots_raw):
                    if np.iscomplexobj(x):
                        arroots.append(complex(x))
                    else:
                        arroots.append(complex(float(x), 0.0))
            except Exception:
                arroots = []

            if not arroots:
                rows.append({
                    'series_id': sid,
                    'hurdle': 'pass',
                    'fallback_forecast': '',
                    'n_train': n_train,
                    'model_spec': ms_repr,
                    'arroots': json.dumps([], ensure_ascii=False),
                    'max_abs_arroot': '',
                    'stable': False,
                    'status': 'unknown',
                    'error': 'no_arroots_from_fit',
                })
                uk_cnt += 1
                continue

            max_abs = max(abs(x) for x in arroots) if arroots else ''
            stable = all(abs(x) > args.threshold for x in arroots)
            status = 'stable' if stable else 'unstable'
            if status == 'stable':
                st_cnt += 1
            else:
                un_cnt += 1

            rows.append({
                'series_id': sid,
                'hurdle': 'pass',
                'fallback_forecast': '',
                'n_train': n_train,
                'model_spec': ms_repr,
                'arroots': json.dumps([[x.real, x.imag] for x in arroots], ensure_ascii=False),
                'max_abs_arroot': max_abs,
                'stable': stable,
                'status': status,
                'error': '',
            })

        except ValueError as e:
            rows.append({"series_id": sid, "error": str(e)})
        except Exception as e:
            rows.append({"series_id": sid, "error": f"fit_failed:{e}"})

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stability_report.csv"

    keys = [
        "series_id",
        "status",
        "hurdle",
        "fallback_forecast",
        "n_train",
        "model_spec",
        "arroots",
        "max_abs_arroot",
        "stable",
        "error",
    ]
    # write CSV with BOM so Excel/PowerShell on Windows displays UTF-8 (Korean) correctly
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    denom = (st_cnt + un_cnt) if args.exclude_empty else (st_cnt + un_cnt + uk_cnt)
    print(f"Loaded snapshot: {trained_path}")
    print(f"Wrote stability report: {csv_path}")
    print({
        'checked': len(rows),
        'stable': st_cnt,
        'unstable': un_cnt,
        'unknown': uk_cnt,
        'threshold': args.threshold,
        'stable_ratio': round(st_cnt / max(1, denom), 4),
    })


if __name__ == "__main__":
    main()
