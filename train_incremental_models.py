"""
증분학습 파이프라인 (월별 데이터)
- Changepoint detection으로 구조 변화 감지
- 변화 있으면 Optuna로 재최적화
- 변화 없으면 기존 파라미터로 빠른 업데이트
- 신규 시리즈는 자동 학습 (12개월 이상 데이터)
"""
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def detect_recent_changepoint(y, window=6, recent_months=6):
    """
    최근 N개월 내 변곡점 감지
    Returns: (has_recent_change, changepoint_indices)
    """
    if len(y) < window * 2:
        return False, []
    
    # 이동 평균 계산
    rolling_mean = pd.Series(y).rolling(window=window, center=True).mean()
    
    # 변화율 계산
    changes = rolling_mean.diff().abs()
    
    # 상위 변화점 탐지
    threshold = changes.quantile(0.75)
    changepoints = np.where(changes > threshold)[0].tolist()
    
    # 최근 N개월 내 변화 있는지
    recent_threshold = len(y) - recent_months
    has_recent = any(cp >= recent_threshold for cp in changepoints)
    
    return has_recent, changepoints


def test_stationarity(y):
    """정상성 검정 (ADF Test)"""
    try:
        result = adfuller(y, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        needs_differencing = 0 if is_stationary else 1
        return is_stationary, p_value, needs_differencing
    except:
        return False, 1.0, 0


def auto_arima_optuna(y, seasonal=True, n_trials=50, max_p=2, max_q=2, max_P=1, max_Q=1):
    """
    Optuna 기반 베이지안 최적화
    """
    if not OPTUNA_AVAILABLE:
        # Fallback to grid search
        return auto_arima_grid(y, seasonal, max_p, max_q, max_P, max_Q)
    
    _, _, d = test_stationarity(y)
    best_result = {'aic': np.inf, 'model': None, 'order': None, 'seasonal': None}
    
    def objective(trial):
        try:
            p = trial.suggest_int('p', 0, max_p)
            q = trial.suggest_int('q', 0, max_q)
            
            if p == 0 and q == 0:
                return float('inf')
            
            if seasonal:
                P = trial.suggest_int('P', 0, max_P)
                D = trial.suggest_int('D', 0, 1) if len(y) >= 24 else 0
                Q = trial.suggest_int('Q', 0, max_Q)
                seasonal_order = (P, D, Q, 12)
            else:
                seasonal_order = (0, 0, 0, 0)
            
            order = (p, d, q)
            
            model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False, maxiter=50, method='lbfgs')
            
            if result.aic < best_result['aic']:
                best_result['aic'] = result.aic
                best_result['model'] = result
                best_result['order'] = order
                best_result['seasonal'] = seasonal_order
            
            return result.aic
        except:
            return float('inf')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    if best_result['model'] is None:
        return (1, 0, 1), (1, 0, 1, 12) if seasonal else (0, 0, 0, 0), np.inf, None
    
    return best_result['order'], best_result['seasonal'], best_result['aic'], best_result['model']


def auto_arima_grid(y, seasonal=True, max_p=2, max_q=2, max_P=1, max_Q=1):
    """Grid search fallback"""
    _, _, d = test_stationarity(y)
    
    best_aic = np.inf
    best_order = (1, 0, 1)
    best_seasonal = (1, 0, 1, 12) if seasonal else (0, 0, 0, 0)
    best_model = None
    
    D_values = [0, 1] if seasonal and len(y) >= 24 else [0]
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            
            for P in range(0, max_P + 1):
                for Q in range(0, max_Q + 1):
                    for D in D_values:
                        if not seasonal and (P > 0 or Q > 0 or D > 0):
                            continue
                        
                        try:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, 12) if seasonal else (0, 0, 0, 0)
                            
                            model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                                          enforce_stationarity=False, enforce_invertibility=False)
                            result = model.fit(disp=False, maxiter=50, method='lbfgs')
                            
                            if result.aic < best_aic:
                                best_aic = result.aic
                                best_order = order
                                best_seasonal = seasonal_order
                                best_model = result
                        except:
                            continue
    
    return best_order, best_seasonal, best_aic, best_model


def cross_validate_forecast(y, order, seasonal_order, horizon=6):
    """교차 검증"""
    if len(y) < horizon + 12:
        return None
    
    train_size = len(y) - horizon
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    try:
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=50, method='lbfgs')
        forecast = result.forecast(steps=horizon)
        forecast = np.maximum(forecast, 0)
        
        mae = np.mean(np.abs(y_test - forecast))
        
        non_zero_mask = y_test > 0.1
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - forecast[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = mae if mae < 1.0 else 100.0
        
        return {'mape': float(mape), 'mae': float(mae), 'predictions': forecast.tolist()}
    except:
        return None


def train_single_series_incremental(json_path, model_dir, output_dir, train_until_year=2024, 
                                    force_reoptimize=False, changepoint_threshold=6):
    """
    단일 시리즈 증분학습
    
    Parameters:
    -----------
    json_path: JSON 파일 경로
    model_dir: 기존 모델 디렉토리
    output_dir: 출력 디렉토리
    train_until_year: 학습 종료 연도
    force_reoptimize: 강제 재최적화
    changepoint_threshold: Changepoint 감지 기준 (최근 N개월)
    """
    try:
        # JSON 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        series_id = data['series_id']
        df = pd.DataFrame(data['data'])
        
        # 학습 데이터 필터링
        df_train = df[(df['year'] >= 2021) & (df['year'] <= train_until_year)].copy()
        
        if len(df_train) < 12:
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': f'insufficient_data ({len(df_train)} months)',
                'path': None
            }
        
        y = df_train['claim_count'].values
        
        # 희소 시리즈 필터링: 월 평균 0.5건 이하는 스킵
        avg_claims_per_month = y.mean()
        if avg_claims_per_month < 0.5:
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': f'sparse_series (avg={avg_claims_per_month:.2f}/month)',
                'path': None
            }
        
        if y.std() < 0.01:
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': 'zero_variance',
                'path': None
            }
        
        # 기존 모델 확인
        safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                       .replace('|', '_').replace('?', '_').replace('*', '_')
                       .replace('<', '_').replace('>', '_').replace('"', '_'))
        
        model_path = Path(model_dir) / f"{safe_filename}.pkl"
        existing_model = None
        existing_order = None
        existing_seasonal = None
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    existing_model = pickle.load(f)
                existing_order = existing_model['model_spec']['order']
                existing_seasonal = existing_model['model_spec']['seasonal_order']
            except:
                pass
        
        # Changepoint 감지 (기존 모델이 있을 때만)
        needs_reoptimization = force_reoptimize
        strategy = 'new_series'
        
        if existing_model is not None:
            has_recent_change, changepoints = detect_recent_changepoint(
                y, window=6, recent_months=changepoint_threshold
            )
            
            if has_recent_change:
                needs_reoptimization = True
                strategy = 'changepoint_detected'
            else:
                strategy = 'fast_update'
        
        # 모델 학습
        if needs_reoptimization or existing_model is None:
            # Optuna 재최적화
            best_order, best_seasonal, best_aic, result = auto_arima_optuna(
                y, seasonal=True, n_trials=50, max_p=2, max_q=2, max_P=1, max_Q=1
            )
            
            if result is None:
                best_order = (1, 0, 1)
                best_seasonal = (1, 0, 1, 12)
                model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal,
                              enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=100, method='lbfgs')
                best_aic = result.aic
        else:
            # 기존 파라미터로 빠른 재학습
            best_order = existing_order
            best_seasonal = existing_seasonal
            model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal,
                          enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False, maxiter=100, method='lbfgs')
            best_aic = result.aic
        
        # 교차 검증
        cv_results = cross_validate_forecast(y, best_order, best_seasonal, horizon=6)
        
        # 변곡점 감지 (전체)
        _, all_changepoints = detect_recent_changepoint(y, window=6, recent_months=len(y))
        
        # 모델 저장
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{safe_filename}.pkl"
        
        model_info = {
            'series_id': series_id,
            'params': result.params.tolist(),
            'model_spec': {
                'order': best_order,
                'seasonal_order': best_seasonal,
                'enforce_stationarity': False,
                'enforce_invertibility': False
            },
            'metadata': {
                'start_year': int(df_train['year'].min()),
                'end_year': int(df_train['year'].max()),
                'n_obs': len(y),
                'mean_claims': float(y.mean()),
                'std_claims': float(y.std()),
                'changepoints': all_changepoints,
                'training_strategy': strategy
            },
            'performance': {
                'aic': float(result.aic),
                'bic': float(result.bic),
                'cv_mape': cv_results['mape'] if cv_results else None,
                'cv_mae': cv_results['mae'] if cv_results else None
            },
            'trained_at': datetime.now().isoformat(),
            'converged': result.mle_retvals['converged'] if hasattr(result, 'mle_retvals') else True
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        return {
            'series_id': series_id,
            'status': 'success',
            'path': str(output_path),
            'order': best_order,
            'seasonal_order': best_seasonal,
            'aic': float(result.aic),
            'mape': cv_results['mape'] if cv_results else None,
            'n_obs': len(y),
            'strategy': strategy
        }
        
    except Exception as e:
        return {
            'series_id': json_path.stem,
            'status': 'error',
            'reason': str(e),
            'path': None
        }


def main():
    parser = argparse.ArgumentParser(description="Incremental training with smart re-optimization")
    parser.add_argument("--json-dir", type=str, default="data/features",
                        help="Directory containing updated JSON files")
    parser.add_argument("--model-dir", type=str, default="artifacts/models/base_monthly",
                        help="Directory with existing models")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/base_monthly",
                        help="Output directory (usually same as model-dir)")
    parser.add_argument("--train-until", type=int, default=2024,
                        help="Train data until this year (inclusive)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--force-reoptimize", action="store_true",
                        help="Force re-optimization for all series")
    parser.add_argument("--changepoint-months", type=int, default=6,
                        help="Changepoint detection window (recent N months)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of series (for testing)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Incremental Training: SARIMAX Models (2021-{args.train_until})")
    print(f"Strategy: Smart Re-optimization")
    if OPTUNA_AVAILABLE:
        print(f"Optimization: Optuna (Bayesian, 50 trials when needed)")
    else:
        print(f"Optimization: Grid Search (Optuna not available)")
    print(f"Changepoint Detection: Last {args.changepoint_months} months")
    print("=" * 80)
    
    # JSON 파일 목록
    json_dir = Path(args.json_dir)
    json_files = list(json_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != '_summary.json']
    
    if args.limit:
        json_files = json_files[:args.limit]
    
    print(f"\n[INFO] Found {len(json_files)} series JSON files")
    print(f"[INFO] Training until: {args.train_until}")
    print(f"[INFO] Model directory: {args.model_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Parallel workers: {args.max_workers}")
    
    # 병렬 학습
    results = []
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    strategy_counts = {
        'new_series': 0,
        'changepoint_detected': 0,
        'fast_update': 0
    }
    
    print(f"\n[INFO] Starting incremental training...")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                train_single_series_incremental,
                json_file,
                args.model_dir,
                args.output_dir,
                args.train_until,
                args.force_reoptimize,
                args.changepoint_months
            ): json_file
            for json_file in json_files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
                strategy = result.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            elif result['status'] == 'skipped':
                skipped_count += 1
            else:
                error_count += 1
            
            if i % 50 == 0 or i == len(json_files):
                print(f"  Progress: {i}/{len(json_files)} | "
                      f"Success: {success_count} | "
                      f"Skipped: {skipped_count} | "
                      f"Errors: {error_count}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("Incremental Training Summary")
    print("=" * 80)
    print(f"Total series:   {len(json_files)}")
    print(f"Success:        {success_count}")
    print(f"Skipped:        {skipped_count}")
    print(f"Errors:         {error_count}")
    
    if success_count > 0:
        print(f"\nTraining Strategy Distribution:")
        print(f"  New series (full optimization):    {strategy_counts.get('new_series', 0)}")
        print(f"  Changepoint detected (re-optimize): {strategy_counts.get('changepoint_detected', 0)}")
        print(f"  Fast update (existing params):      {strategy_counts.get('fast_update', 0)}")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_path = Path(args.output_dir) / "incremental_training_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] Results saved to: {results_path}")
    
    # 상위 모델 표시
    if success_count > 0:
        success_results = results_df[results_df['status'] == 'success'].copy()
        success_results = success_results.sort_values('aic')
        
        print(f"\n[INFO] Top 10 models (by AIC):")
        for idx, row in success_results.head(10).iterrows():
            strategy_icon = {
                'new_series': '🆕',
                'changepoint_detected': '🔄',
                'fast_update': '⚡'
            }.get(row.get('strategy', ''), '❓')
            
            mape_str = f"{row['mape']:.1f}%" if pd.notna(row.get('mape')) else "N/A"
            print(f"  {strategy_icon} {row['series_id'][:45]:45s} | {row['order']} {row['seasonal_order']} | "
                  f"AIC: {row['aic']:,.1f} | MAPE: {mape_str}")
    
    print("\n[SUCCESS] Incremental training completed!")
    print(f"💡 Tip: Use --force-reoptimize to re-optimize all series")


if __name__ == '__main__':
    main()
