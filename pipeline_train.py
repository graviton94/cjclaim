"""
Train 파이프라인: features 데이터로 시계열 모델 학습 및 아티팩트 저장
"""
import pandas as pd
import numpy as np
import warnings
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from src.forecasting import safe_forecast
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.model_monthly import fit_monthly_sarimax, fit_with_retries, forecast_and_inverse, loss_with_bias


# Selection helper: prefer low loss, break ties by AIC, conditional seasonal promotion
def select_best(cands, eps_loss=0.05, delta_aic_seasonal=2.0, root_threshold=1.01):
    """
    cands: list of dicts with keys {'res','spec','yhat','loss','aic'}
    Policy:
      1) choose minimal validation loss
      2) if losses within eps_loss, choose by minimal AIC
      3) conditional seasonal promotion: only promote seasonal when its AIC
         disadvantage is <= delta_aic_seasonal
      4) exclude candidates with invalid AIC (NaN/inf) or unstable AR roots (min_abs <= root_threshold)
    """
    import math

    def ok(c):
        aic = c.get('aic', None)
        if aic is None or not np.isfinite(aic):
            return False
        # stability guard: reject candidates with AR roots magnitude <= threshold
        roots = getattr(c.get('res'), 'arroots', None)
        if roots is not None and len(roots) > 0:
            try:
                mins = np.min(np.abs(roots))
                if np.isfinite(mins) and mins <= root_threshold:
                    return False
            except Exception:
                # if any unexpected error, keep candidate (don't be overly strict)
                pass
        return True

    pool = [c for c in cands if ok(c)]
    if not pool:
        # fallback: choose smallest loss from original cands (preserve prior behavior)
        return min(cands, key=lambda x: x['loss'])

    # 1) loss primary
    pool.sort(key=lambda x: x['loss'])
    best = pool[0]
    peers = [c for c in pool if abs(c['loss'] - best['loss']) <= eps_loss]
    if len(peers) == 1:
        return best

    # 2) tie -> pick by aic
    peers.sort(key=lambda x: x['aic'])
    aic_best = peers[0]

    def is_seasonal(c):
        so = c.get('spec', {}).get('seasonal_order', (0, 0, 0, 0))
        try:
            return tuple(so) != (0, 0, 0, 0)
        except Exception:
            return False

    # conditional seasonal promotion: if AIC-best is non-seasonal but a seasonal
    # candidate is within delta_aic_seasonal, promote the seasonal candidate.
    if not is_seasonal(aic_best):
        for cand in peers[1:]:
            if is_seasonal(cand):
                if (cand['aic'] - aic_best['aic']) <= delta_aic_seasonal:
                    return cand
                break
    return aic_best


# 경로 설정
CURATED_PATH = "data/curated/claims_monthly.parquet"
FEATURES_PATH = "data/features/cycle_features.parquet"
ARTIFACTS_DIR = "artifacts"

# 파라미터 설정 (MONTHLY)
# If TRAIN_UNTIL_YEAR or TRAIN_UNTIL_MONTH is None, we will derive the cutoff from the data (latest yyyymm)
TRAIN_UNTIL_YEAR = None
TRAIN_UNTIL_MONTH = None
FORECAST_HORIZON = 6    # 6개월 예측
SEASONAL_ORDER = (1, 1, 1, 12)  # monthly seasonal order
CONFIDENCE_INTERVAL = 0.95
MIN_MONTHS_TO_FIT = 18

print("=" * 80)
print("Train Pipeline - 시계열(월간) 모델 학습")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 데이터 로드")
print("-" * 80)
df_curated = pd.read_parquet(CURATED_PATH)
df_features = pd.read_parquet(FEATURES_PATH)
print(f"Curated 데이터: {len(df_curated):,} 행, {df_curated['series_id'].nunique()} 시리즈")
print(f"Features 데이터: {len(df_features):,} 행, {df_features['series_id'].nunique()} 시리즈")

# 2. 학습 데이터 필터링 (월 기준)
print("\n[2] 학습 데이터 필터링")
print("-" * 80)
# ensure year/month exist in df_curated; if yyyymm present, derive year/month
if 'yyyymm' in df_curated.columns and not ({'year', 'month'} <= set(df_curated.columns)):
    df_curated['yyyymm'] = df_curated['yyyymm'].astype(str).str.replace(r'[^0-9]', '', regex=True).astype(int)
    df_curated['year'] = (df_curated['yyyymm'] // 100).astype(int)
    df_curated['month'] = (df_curated['yyyymm'] % 100).astype(int)

# derive train-until from constants if provided, otherwise from the latest data point
if TRAIN_UNTIL_YEAR is None or TRAIN_UNTIL_MONTH is None:
    # pick latest yyyymm from data
    if 'yyyymm' in df_curated.columns:
        latest = int(df_curated['yyyymm'].max())
        derived_year = latest // 100
        derived_month = latest % 100
    else:
        derived_year = int(df_curated['year'].max())
        # if month missing, default to 12
        derived_month = int(df_curated['month'].max()) if 'month' in df_curated.columns else 12
    train_until_year = TRAIN_UNTIL_YEAR if TRAIN_UNTIL_YEAR is not None else derived_year
    train_until_month = TRAIN_UNTIL_MONTH if TRAIN_UNTIL_MONTH is not None else derived_month
else:
    train_until_year = TRAIN_UNTIL_YEAR
    train_until_month = TRAIN_UNTIL_MONTH

# build mask up to train_until_year/train_until_month inclusive
train_mask = (
    (df_curated['year'] < int(train_until_year)) |
    ((df_curated['year'] == int(train_until_year)) & (df_curated['month'] <= int(train_until_month)))
)
df_train = df_curated[train_mask].copy()
print(f"학습 데이터: {len(df_train):,} 행 (train_until={train_until_year}-{int(train_until_month):02d})")

# Normalize year/month columns if yyyymm present and compute global start/stop dates
if {'yyyymm','year','month'} <= set(df_train.columns):
    # prefer yyyymm if present
    if 'yyyymm' in df_train.columns:
        df_train['yyyymm'] = df_train['yyyymm'].astype(str).str.replace(r'[^0-9]','',regex=True).astype(int)
        df_train['year'] = (df_train['yyyymm']//100).astype(int)
        df_train['month'] = (df_train['yyyymm']%100).astype(int)

# Establish a single fixed train-until date for the whole run and a global
# start-date fallback. This ensures consistent reindexing per-series and
# prevents mixed execution paths that previously produced uniform 24/18 windows.
train_until_date_fixed = pd.Timestamp(int(train_until_year), int(train_until_month), 1)
try:
    global_start_year = int(df_train['year'].min())
    global_start_month = int(df_train['month'].min())
    global_start_date = pd.Timestamp(global_start_year, global_start_month, 1)
except Exception:
    global_start_date = train_until_date_fixed

print(f"학습 전체 윈도우(글로벌 fallback): {global_start_date.date()} ~ {train_until_date_fixed.date()}")

# 3. 시리즈별 모델 학습
print("\n[3] 시리즈별 모델 학습")
print("-" * 80)
failed_series = []

artifacts = {
    'metadata': {
        'train_date': datetime.now().isoformat(),
        'train_until_year': train_until_year,
        'train_until_month': train_until_month,
        'forecast_horizon': FORECAST_HORIZON,
        'seasonal_order': SEASONAL_ORDER,
        'confidence_interval': CONFIDENCE_INTERVAL,
        'n_series': df_train['series_id'].nunique(),
        'n_samples': len(df_train)
    },
    'series_models': {}
}

series_list = df_train['series_id'].unique()
n_series = len(series_list)

# 시리즈별 학습 진행
print(f"총 {n_series}개 시리즈 학습 시작...")
print("=" * 80)

for i, series_id in enumerate(series_list, 1):
    # 매 시리즈마다 진행률 표시
    elapsed_pct = i / n_series * 100
    series_display = series_id[:40] if len(series_id) > 40 else series_id
    print(f"\r[{i:,}/{n_series:,}] ({elapsed_pct:.1f}%) {series_display}...", end='', flush=True)
    
    # 매 500개마다 줄바꿈
    if i % 500 == 0:
        print()  # 새 줄로 이동
    
    # 시리즈 데이터 추출 (monthly)
    series_data = df_train[df_train['series_id'] == series_id][['year', 'month', 'claim_count']]
    # aggregate in case multiple rows map to same year/month for this series
    series_data = series_data.groupby(['year', 'month'], as_index=False)['claim_count'].sum()
    series_data = series_data.sort_values(['year', 'month'])

    # Per-series reindex: build a monthly index from the series' first available month
    # up to the global train-until date. This avoids forcing every series to the
    # global `unique_months` range (which can add many leading zero months and
    # make all series lengths identical).
    # use the single fixed train-until date established above
    train_until_date = train_until_date_fixed

    if len(series_data) > 0:
        start_year = int(series_data['year'].min())
        start_month = int(series_data['month'].min())
        start_date = pd.Timestamp(start_year, start_month, 1)
        # ensure start_date is not after train_until_date
        if start_date > train_until_date:
            start_date = train_until_date
    else:
        # no data rows for this series (should be rare) - fall back to global start
        start_date = global_start_date

    # build monthly index for this series and merge to fill missing months with 0
    idx = pd.date_range(start=start_date, end=train_until_date, freq='MS')
    series_full = pd.DataFrame({'year': idx.year, 'month': idx.month})
    series_full = series_full.merge(series_data, on=['year', 'month'], how='left')
    series_full['claim_count'] = series_full['claim_count'].fillna(0)

    # construct DatetimeIndex (month-start) and series
    y_idx = pd.to_datetime(series_full[['year', 'month']].assign(day=1))
    y_idx = pd.DatetimeIndex(y_idx)
    y = pd.Series(series_full['claim_count'].astype(float).values, index=y_idx).sort_index()

    # ensure the series index has an explicit monthly freq (MS) so statsmodels
    # will not warn about missing frequency and will produce time-aware forecasts.
    # Try to set a concrete monthly frequency if the index is regular; if not,
    # avoid forcing an incorrect freq which raises. statsmodels will infer when
    # necessary but will warn; we prefer to set freq only when it can be inferred.
    inferred = None
    try:
        inferred = pd.infer_freq(y.index)
    except Exception:
        inferred = None

    if inferred is not None:
        try:
            y.index = pd.DatetimeIndex(y.index.values, freq=inferred)
        except Exception:
            # if setting freq fails, leave as-is and allow statsmodels to infer
            pass

    

    # dynamic holdout size depending on series length: at most FORECAST_HORIZON,
    # but scale down for short series to keep a reasonable train/val split.
    holdout = min(FORECAST_HORIZON, max(1, len(y) // 5))
    if len(y) <= holdout:
        artifacts.setdefault('failed_series', {})
        artifacts['failed_series'][series_id] = {
            'n_train_points': int(len(y)),
            'message': f'Not enough history to hold out validation (need > {holdout}), got {len(y)}'
        }
        failed_series.append({'series_id': series_id, 'error': f'Not enough history to hold out validation (need > {holdout}), got {len(y)}'})
        continue
    y_train = y[:-holdout]
    y_val = y[-holdout:]

    # enforce minimum training months based on the training portion (not raw total)
    if len(y_train) < MIN_MONTHS_TO_FIT:
        baseline_forecast = [float(np.mean(y_train)) if len(y_train) > 0 else 0.0 for _ in range(FORECAST_HORIZON)]
        artifacts['series_models'][series_id] = {
            'model_type': 'baseline_short',
            'n_total': int(len(y)),
            'n_train_points': int(len(y_train)),
            'forecast': {'yhat': baseline_forecast},
            'note': f'n_train<{MIN_MONTHS_TO_FIT} → baseline used'
        }
        failed_series.append({'series_id': series_id, 'error': f'n_train<{MIN_MONTHS_TO_FIT} - baseline used'})
        continue

    # ---------- 새로 추가: 시리즈 통계 계산 ----------
    # compute train-side statistics to persist with model metadata
    try:
        train_arr = np.asarray(y_train.values, dtype=float)
        n_train_points = int(len(train_arr)
                              )
        nonzero_pct_train = float(np.count_nonzero(train_arr) / max(1, n_train_points) * 100.0)
        train_var = float(np.var(train_arr, ddof=1)) if n_train_points > 1 else 0.0
        # max consecutive zero run in training series
        max_zero = 0
        run = 0
        for v in train_arr:
            if v == 0:
                run += 1
                if run > max_zero:
                    max_zero = run
            else:
                run = 0
    except Exception:
        n_train_points = len(y_train)
        nonzero_pct_train = float(np.sum(y_train.values > 0) / len(y_train.values) * 100) if len(y_train) > 0 else 0.0
        train_var = 0.0
        max_zero = 0

    # candidate specs for model selection: choose simpler (no-seasonal) specs for short histories
    n_train = len(y_train)
    n_total = len(y)
    # log selection branch for diagnostics
    print(f"[SELECT] {series_id} n_total={n_total} n_train={n_train} short_path={(n_total<24)}")
    # use total-length threshold (n_total) to decide whether to avoid seasonal models.
    # Treat series with total history < 24 months as short-path.
    if n_total < 24:
        # avoid seasonal SARIMAX for short histories
        candidate_specs = [
            {'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)},
            {'order': (0, 0, 0), 'seasonal_order': (0, 0, 0, 0)},
        ]
    else:
        candidate_specs = [
            {'order': (2, 0, 0), 'seasonal_order': SEASONAL_ORDER},
            {'order': (1, 0, 0), 'seasonal_order': SEASONAL_ORDER},
        ]
    # add a safe non-seasonal fallback candidate to avoid total failure if
    # seasonal candidate handling unexpectedly raises or returns bad results.
    # This candidate will be tried last and can produce a baseline AR fit.
    candidate_specs.append({'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)})
    # single pass: try each candidate and collect successful fits for post-selection
    best_loss = float('inf')
    best_info = None
    successful_candidates = []
    # capture warnings from statsmodels to keep logs cleaner; they are informative but not fatal
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        for spec in candidate_specs:
            try:
                # short histories: build SARIMAX directly and use fit_with_retries
                if spec.get('seasonal_order', None) == (0, 0, 0, 0):
                    # simple ARIMA without seasonal component
                    y_t = np.sqrt(y_train.astype(float))
                    mod = SARIMAX(endog=y_t, order=spec['order'], seasonal_order=spec['seasonal_order'],
                                  trend='n', enforce_stationarity=True, enforce_invertibility=True)
                    res = fit_with_retries(mod, maxiter_list=(50, 200), methods=('lbfgs', 'bfgs', 'powell'))
                else:
                    # seasonal candidates: use the helper that already tries seasonal candidates and retries
                    res = fit_monthly_sarimax(y_train, order=spec['order'], seasonal_order=spec['seasonal_order'])

                yhat = forecast_and_inverse(res, steps=FORECAST_HORIZON)
                loss = loss_with_bias(y_val.values, yhat)
                aic = getattr(res, 'aic', None)
                # exclude abnormal AIC candidates (NaN/inf) from consideration
                if aic is None or not np.isfinite(aic):
                    # record as failed candidate for visibility and skip
                    failed_series.append({'series_id': series_id, 'error': f'invalid_aic: {aic} spec={spec}'})
                    continue
                successful_candidates.append({
                    'res': res,
                    'spec': spec,
                    'yhat': yhat,
                    'loss': loss,
                    'aic': float(aic)
                })
                if loss < best_loss:
                    best_loss = loss
                    best_info = successful_candidates[-1]
            except Exception as e:
                # candidate failed to fit — record and try next candidate
                failed_series.append({'series_id': series_id, 'error': str(e)})
                continue

    if best_info is None:
        # no candidate succeeded for this series — continue (do not abort entire pipeline)
        artifacts.setdefault('failed_series', {})
        artifacts['failed_series'][series_id] = {
            'n_train_points': len(y),
            'message': 'all candidates failed to fit'
        }
        continue

    # Post-selection: prefer seasonal model when its AIC is within delta (2) of the chosen model.
    # Use helper from src.selection_utils for clarity and testability.
    # Use our stricter select_best policy to pick among successful candidates.
    try:
        if successful_candidates:
            best_info = select_best(successful_candidates, eps_loss=0.05, delta_aic_seasonal=2.0, root_threshold=1.01)
    except Exception:
        # on any failure in selection logic, keep previously chosen best_info
        pass

    # record artifact for best model
    res = best_info['res']
    spec = best_info['spec']
    yhat = best_info['yhat']

    hist_mean = float(np.mean(y.values))
    hist_std = float(np.std(y.values))
    hist_max = float(np.max(y.values))
    hist_min = float(np.min(y.values))
    nonzero_pct = float(np.sum(y.values > 0) / len(y.values) * 100) if len(y.values) > 0 else 0.0

    # prepare spec for saving and log for diagnostics
    spec_to_save = spec
    # diagnostic: print chosen spec and available successful candidates (aic/loss)
    try:
        cand_summaries = [
            f"order={c['spec'].get('order')}, so={c['spec'].get('seasonal_order')}, aic={c.get('aic')}, loss={c.get('loss'):.4f}"
            for c in successful_candidates
        ]
    except Exception:
        cand_summaries = []
    print(f"[SAVE] {series_id} selected_seasonal={spec_to_save.get('seasonal_order')} selected_loss={best_info.get('loss'):.4f} selected_aic={best_info.get('aic')}")
    if cand_summaries:
        print(f"[SAVE] {series_id} candidate_summaries={cand_summaries}")
    # warn if series is long enough to expect seasonality but final spec is non-seasonal
    if len(y) >= 36:
        if spec_to_save.get('seasonal_order') == (0, 0, 0, 0):
            # if there were seasonal candidates attempted, record a selection warning
            if any((c['spec'].get('seasonal_order') != (0, 0, 0, 0)) for c in successful_candidates):
                artifacts.setdefault('selection_warnings', {}).setdefault(series_id, {})
                artifacts['selection_warnings'][series_id] = {
                    'message': 'long-series selected non-seasonal despite seasonal candidates',
                    'candidates': cand_summaries
                }
    # serialize model params safely: if params contain complex values, coerce
    # to their real part to avoid ComplexWarning and JSON serialization errors.
    params_serial = None
    try:
        raw_params = getattr(res, 'params', None)
        if raw_params is not None:
            params_serial = []
            for p in raw_params:
                try:
                    # prefer the real part if complex; fall back to float coercion
                    if isinstance(p, complex) or (hasattr(p, 'imag') and getattr(p, 'imag', 0) != 0):
                        params_serial.append(float(np.real(p)))
                    else:
                        params_serial.append(float(p))
                except Exception:
                    params_serial.append(None)
    except Exception:
        params_serial = None

    artifacts['series_models'][series_id] = {
        'model_type': 'sarimax_monthly',
        'model_spec': spec_to_save,
        'params': params_serial,
        # save n_train_points as the number of points used for training (exclude holdout)
        'n_train_points': n_train_points,
        'hist_mean': hist_mean,
        'hist_std': hist_std,
        'hist_max': hist_max,
        'hist_min': hist_min,
        'nonzero_pct': nonzero_pct_train,
        'train_var': train_var,
        'max_zero_run': int(max_zero),
        'last_value': float(y.values[-1]) if len(y.values) > 0 else 0.0,
            'forecast': {
            # yhat may contain tiny complex rounding errors; coerce to real part
            # before converting to float to avoid ComplexWarning.
            'yhat': [float(np.real(x)) for x in np.asarray(yhat)],
        },
        'selection_loss': float(best_info['loss'])
    }
    # include AIC and AR roots when available (was missing in earlier saves)
    try:
        aic_val = float(getattr(res, 'aic', None)) if getattr(res, 'aic', None) is not None else None
    except Exception:
        aic_val = None
    # Safely handle AR roots: statsmodels may return complex roots. We store
    # the magnitude (abs) of each root as a float to avoid ComplexWarning when
    # coercing complex -> float and to keep a numeric stability metric for
    # downstream tools. Also log if complex parts are present for debugging.
    try:
        raw_arroots = getattr(res, 'arroots', None)
        arroots_val = []
        if raw_arroots is not None:
            for r in raw_arroots:
                try:
                    # capture magnitude as the canonical numeric representation
                    mag = float(abs(r))
                    arroots_val.append(mag)
                    # optional debug: if root has non-zero imaginary part, note it
                    if isinstance(r, complex) and (abs(r.imag) > 1e-12):
                        # attach a lightweight debug field in artifacts if needed
                        artifacts.setdefault('selection_warnings', {}).setdefault(series_id, {})
                        artifacts['selection_warnings'][series_id] = artifacts['selection_warnings'][series_id] or {}
                        artifacts['selection_warnings'][series_id].setdefault('complex_arroots', True)
                except Exception:
                    # fallback: try real coercion
                    try:
                        arroots_val.append(float(getattr(r, 'real', r)))
                    except Exception:
                        continue
        else:
            arroots_val = []
    except Exception:
        arroots_val = []
    # normalize model_spec to simple serializable types
    try:
        ms = spec_to_save.get('order') if isinstance(spec_to_save, dict) else None
    except Exception:
        ms = None
    # ensure model_spec.order and seasonal_order are lists (not tuples)
    if isinstance(spec_to_save, dict):
        spec_to_save['order'] = list(spec_to_save.get('order')) if spec_to_save.get('order') is not None else None
        spec_to_save['seasonal_order'] = list(spec_to_save.get('seasonal_order')) if spec_to_save.get('seasonal_order') is not None else None
    # ensure model_spec normalized
    if isinstance(spec_to_save, dict):
        # guarantee lists for orders
        spec_to_save['order'] = list(spec_to_save.get('order')) if spec_to_save.get('order') is not None else [0, 0, 0]
        spec_to_save['seasonal_order'] = list(spec_to_save.get('seasonal_order')) if spec_to_save.get('seasonal_order') is not None else [0, 0, 0, 0]

    # runtime asserts to prevent silent omission of critical fields
    # Require aic to be present (not None) so we do not write incomplete entries.
    if aic_val is None:
        raise AssertionError(f"aic missing for series {series_id} - aborting save to prevent silent omission")
    # Require seasonal_order key in model_spec
    if not isinstance(spec_to_save, dict) or 'seasonal_order' not in spec_to_save:
        raise AssertionError(f"model_spec.seasonal_order missing for series {series_id}")

    artifacts['series_models'][series_id]['model_spec'] = spec_to_save
    # Always write aic (now asserted non-None) and arroots (list)
    artifacts['series_models'][series_id]['aic'] = aic_val
    artifacts['series_models'][series_id]['arroots'] = arroots_val
    # computed stability metric: minimal absolute AR root (useful for quick checks)
    try:
        # arroots_val already contains magnitudes (floats), so min is direct
        min_abs_ar = float(min(arroots_val)) if arroots_val else None
    except Exception:
        min_abs_ar = None
    artifacts['series_models'][series_id]['min_abs_arroot'] = min_abs_ar
    # compatibility: also expose 'min_abs_root' key expected by some tools
    artifacts['series_models'][series_id]['min_abs_root'] = min_abs_ar

    # runtime asserts to ensure critical metadata persisted
    # Require aic to be present (not None) so we do not write incomplete entries.
    if artifacts['series_models'][series_id].get('aic', None) is None:
        raise AssertionError(f"aic missing for series {series_id} - aborting save to prevent silent omission")
    # Require seasonal_order key in model_spec
    if not isinstance(spec_to_save, dict) or 'seasonal_order' not in spec_to_save:
        raise AssertionError(f"model_spec.seasonal_order missing for series {series_id}")

print()  # 진행률 표시 후 줄바꿈
print("=" * 80)

# 4. 모델 통계
print("\n[4] 모델 학습 통계")
print("-" * 80)
model_types = {}
for series_id, model_info in artifacts['series_models'].items():
    model_type = model_info.get('model_type', 'unknown')
    model_types[model_type] = model_types.get(model_type, 0) + 1

print("모델 타입별 분포:")
for model_type, count in sorted(model_types.items(), key=lambda x: -x[1]):
    pct = count / n_series * 100
    print(f"  {model_type}: {count} ({pct:.1f}%)")

# 5. 아티팩트 저장
print("\n[5] 아티팩트 저장")
print("-" * 80)
Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
artifact_path = Path(ARTIFACTS_DIR) / "trained_models.json"

# --- 안전한(atomic) 저장 및 빈 덮어쓰기 보호 ---
new_series_count = len(artifacts.get('series_models', {}))

# If an existing snapshot is present, check its series count to avoid overwriting
if artifact_path.exists():
    try:
        with open(artifact_path, 'r', encoding='utf-8') as _f:
            existing = json.load(_f)
        existing_count = len(existing.get('series_models', {}) if isinstance(existing, dict) else {})
    except Exception:
        existing_count = 0
else:
    existing_count = 0

if existing_count > 0 and new_series_count == 0:
    # Protect against accidental overwrite: do not replace a populated snapshot with an empty one
    warn_msg = (
        f"Refusing to overwrite non-empty snapshot ({existing_count} series) with empty snapshot (0 series). "
        f"Promote or inspect artifacts manually. Backup saved as .skipped_empty.<ts>."
    )
    print("[WARN] " + warn_msg)
    # write a diagnostic backup so we don't lose the attempted empty save
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    backup_path = artifact_path.with_name(artifact_path.name + f".skipped_empty.{ts}.json")
    try:
        with open(backup_path, 'w', encoding='utf-8') as bf:
            json.dump(artifacts, bf, indent=2, ensure_ascii=False)
        print(f"Wrote diagnostic backup of attempted empty snapshot to: {backup_path}")
    except Exception as e:
        print(f"Failed to write diagnostic backup: {e}")
else:
    # perform atomic write: write to a temp file and fsync before replace
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=artifact_path.name + '.', dir=artifact_path.parent)
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as tf:
            json.dump(artifacts, tf, indent=2, ensure_ascii=False)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, artifact_path)
        print(f"아티팩트 저장 완료: {artifact_path}")
        try:
            print(f"파일 크기: {artifact_path.stat().st_size / 1024 / 1024:.2f} MB")
        except Exception:
            pass
    finally:
        # ensure tmp file does not linger
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# save failed series list separately for inspection
failed_path = Path(ARTIFACTS_DIR) / "failed_series.json"
if len(failed_series) > 0:
    with open(failed_path, 'w', encoding='utf-8') as f:
        json.dump(failed_series, f, indent=2, ensure_ascii=False)
    print(f"Failed series written: {failed_path} ({len(failed_series)} entries)")
else:
    # ensure an empty file exists for downstream checks
    with open(failed_path, 'w', encoding='utf-8') as f:
        json.dump([], f)

# 6. 학습 요약 저장
print("\n[6] 학습 요약 저장")
print("-" * 80)
summary = {
    'train_date': artifacts['metadata']['train_date'],
    'n_series': n_series,
    'n_samples': len(df_train),
    'train_period': f"{df_train['year'].min()}-{df_train['month'].min():02d} ~ {df_train['year'].max()}-{df_train['month'].max():02d}",
    'forecast_horizon': FORECAST_HORIZON,
    'model_distribution': model_types,
}

# safe computation of average non-zero percentage across saved series
nonzero_vals = [
    float(m.get('nonzero_pct', 0.0))
    for m in artifacts.get('series_models', {}).values()
]
avg_nonzero = float(np.mean(nonzero_vals)) if len(nonzero_vals) > 0 else 0.0

summary['avg_nonzero_pct'] = avg_nonzero

summary_path = Path(ARTIFACTS_DIR) / "training_summary.json"
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"학습 요약 저장 완료: {summary_path}")

print("\n" + "=" * 80)
print("✓ Train Pipeline 완료!")
print("=" * 80)
print(f"✓ {n_series} 개 시리즈 학습 완료")
print(f"✓ 예측 기간: {FORECAST_HORIZON} 개월")
print(f"✓ 아티팩트: {artifact_path}")
print("=" * 80)
