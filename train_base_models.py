"""
Base Training v2: 2021-2023 월별 데이터로 전체 시리즈 SARIMAX 모델 학습
- 자동 파라미터 최적화: 시리즈별 최적 ARIMA order 탐색
- 계절성 SARIMAX: (p,d,q)(P,D,Q,12) - 12개월 계절성
- AIC 기반 모델 선택
- 변곡점 감지 및 트렌드 분석
- 3-Metric KPI: WMAPE, SMAPE, Bias
- Reproducibility Manifest: run_id, git_commit, data_hash
- 희소 시리즈 필터링 (avg<0.5 or nonzero<30%)
- 병렬 처리로 속도 최적화
- artifacts/models/base_monthly/ 저장
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
import sys

# Import new modules
from src.metrics_v2 import calculate_all_metrics, get_performance_level
from src.manifest import ManifestBuilder, generate_run_id

warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def detect_changepoints(y, window=6):
    """
    변곡점 감지: 이동 평균의 변화율 분석
    Returns: 주요 변곡점 인덱스 리스트
    """
    if len(y) < window * 2:
        return []
    
    # 이동 평균 계산
    rolling_mean = pd.Series(y).rolling(window=window, center=True).mean()
    
    # 변화율 계산
    changes = rolling_mean.diff().abs()
    
    # 상위 변화점 탐지
    threshold = changes.quantile(0.75)
    changepoints = np.where(changes > threshold)[0].tolist()
    
    return changepoints


def test_stationarity(y):
    """
    정상성 검정 (ADF Test)
    Returns: (is_stationary, p_value, needs_differencing)
    """
    try:
        result = adfuller(y, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        
        # d 파라미터 결정
        needs_differencing = 0 if is_stationary else 1
        
        return is_stationary, p_value, needs_differencing
    except:
        return False, 1.0, 0


def auto_arima_selection(y, seasonal=True, max_p=3, max_q=3, max_P=2, max_Q=2):
    """
    자동 ARIMA 파라미터 선택
    - Grid search로 최적 (p,d,q)(P,D,Q,s) 탐색
    - AIC 최소화
    
    Returns: best_order, best_seasonal_order, best_aic, best_model
    """
    # 정상성 검정으로 d 결정
    _, _, d = test_stationarity(y)
    
    best_aic = np.inf
    best_order = (1, 0, 1)
    best_seasonal = (1, 0, 1, 12) if seasonal else (0, 0, 0, 0)
    best_model = None
    
    # 계절성 D 파라미터
    D_values = [0, 1] if seasonal and len(y) >= 24 else [0]
    
    # Grid search
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue  # (0,d,0) 제외
            
            for P in range(0, max_P + 1):
                for Q in range(0, max_Q + 1):
                    for D in D_values:
                        if not seasonal and (P > 0 or Q > 0 or D > 0):
                            continue
                        
                        try:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, 12) if seasonal else (0, 0, 0, 0)
                            
                            model = SARIMAX(
                                y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            
                            result = model.fit(disp=False, maxiter=50, method='lbfgs')
                            
                            if result.aic < best_aic:
                                best_aic = result.aic
                                best_order = order
                                best_seasonal = seasonal_order
                                best_model = result
                                
                        except:
                            continue
    
    return best_order, best_seasonal, best_aic, best_model


def auto_arima_selection_optuna(y, seasonal=True, n_trials=50, max_p=3, max_q=3, max_P=2, max_Q=2):
    """
    Optuna 기반 베이지안 최적화로 ARIMA 파라미터 선택
    - Grid search보다 10-20배 빠름
    - 더 효율적인 탐색
    
    Returns: best_order, best_seasonal_order, best_aic, best_model
    """
    if not OPTUNA_AVAILABLE:
        # Optuna 없으면 기본 Grid search 사용
        return auto_arima_selection(y, seasonal, max_p, max_q, max_P, max_Q)
    
    # 정상성 검정으로 d 결정
    _, _, d = test_stationarity(y)
    
    best_result = {'aic': np.inf, 'model': None, 'order': None, 'seasonal': None}
    
    def objective(trial):
        try:
            # 파라미터 제안
            p = trial.suggest_int('p', 0, max_p)
            q = trial.suggest_int('q', 0, max_q)
            
            # (0,d,0) 제외
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
            
            # 모델 학습
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            result = model.fit(disp=False, maxiter=50, method='lbfgs')
            
            # 최고 결과 저장
            if result.aic < best_result['aic']:
                best_result['aic'] = result.aic
                best_result['model'] = result
                best_result['order'] = order
                best_result['seasonal'] = seasonal_order
            
            return result.aic
            
        except Exception as e:
            return float('inf')
    
    # Optuna 최적화 실행
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    if best_result['model'] is None:
        # 실패 시 기본값
        return (1, 0, 1), (1, 0, 1, 12) if seasonal else (0, 0, 0, 0), np.inf, None
    
    return best_result['order'], best_result['seasonal'], best_result['aic'], best_result['model']


def cross_validate_forecast(y, order, seasonal_order, horizon=6):
    """
    교차 검증으로 예측 성능 평가 - 3-Metric KPI 계산
    - 마지막 horizon 개월을 테스트 셋으로 사용
    - WMAPE, SMAPE, Bias, MAE 계산
    
    Returns: {'wmape': float, 'smape': float, 'bias': float, 'mae': float, 'predictions': array}
    """
    if len(y) < horizon + 12:
        return None
    
    train_size = len(y) - horizon
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    try:
        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False, maxiter=50, method='lbfgs')
        forecast = result.forecast(steps=horizon)
        forecast = np.maximum(forecast, 0)
        
        # 3-Metric KPI 계산
        metrics = calculate_all_metrics(y_test, forecast)
        metrics['predictions'] = forecast.tolist()
        
        return metrics
    except:
        return None


def train_single_series(json_path, train_until_year=2023, output_dir=None, auto_optimize=True):
    """
    단일 시리즈 학습 with 자동 파라미터 최적화
    """
    try:
        # JSON 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        series_id = data['series_id']
        df = pd.DataFrame(data['data'])
        
        # 학습 기간 필터링 (2021-2023)
        df_train = df[(df['year'] >= 2021) & (df['year'] <= train_until_year)].copy()
        
        if len(df_train) < 12:  # 최소 12개월 데이터 필요
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': f'insufficient_data (only {len(df_train)} months)',
                'path': None
            }
        
        # 시계열 데이터 준비
        y = df_train['claim_count'].values
        
        # 희소 시리즈 필터링: 월 평균 0.5건 이하 또는 비제로 비율 30% 미만
        avg_claims_per_month = y.mean()
        nonzero_ratio = np.count_nonzero(y) / len(y)
        
        sparse_flag = (avg_claims_per_month < 0.5) or (nonzero_ratio < 0.3)
        
        if sparse_flag:
            reason_parts = []
            if avg_claims_per_month < 0.5:
                reason_parts.append(f'avg={avg_claims_per_month:.2f}<0.5')
            if nonzero_ratio < 0.3:
                reason_parts.append(f'nonzero={nonzero_ratio:.1%}<30%')
            sparse_reason = '; '.join(reason_parts)
            
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': f'sparse_series ({sparse_reason})',
                'sparse_flag': True,
                'sparse_reason': sparse_reason,
                'path': None
            }
        
        # 0 분산 체크
        if y.std() < 0.01:
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': 'zero_variance',
                'path': None
            }
        
        # 변곡점 감지
        changepoints = detect_changepoints(y)
        
        # 자동 파라미터 최적화
        if auto_optimize:
            # Optuna 사용 (더 빠르고 효율적)
            best_order, best_seasonal, best_aic, result = auto_arima_selection_optuna(
                y, seasonal=True, n_trials=50, max_p=2, max_q=2, max_P=1, max_Q=1
            )
            
            if result is None:
                # Optuna 실패 시 기본 파라미터
                best_order = (1, 0, 1)
                best_seasonal = (1, 0, 1, 12)
                model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal,
                              enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=100, method='lbfgs')
                best_aic = result.aic
        else:
            # 고정 파라미터
            best_order = (1, 0, 1)
            best_seasonal = (1, 0, 1, 12)
            model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal,
                          enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False, maxiter=100, method='lbfgs')
            best_aic = result.aic
        
        # 교차 검증
        cv_results = cross_validate_forecast(y, best_order, best_seasonal, horizon=6)
        
        # 모델 저장
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                           .replace('|', '_').replace('?', '_').replace('*', '_')
                           .replace('<', '_').replace('>', '_').replace('"', '_'))
            
            model_path = output_dir / f"{safe_filename}.pkl"
            
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
                    'nonzero_ratio': float(nonzero_ratio),
                    'changepoints': changepoints,
                    'auto_optimized': auto_optimize
                },
                'performance': {
                    'aic': float(result.aic),
                    'bic': float(result.bic),
                    'cv_wmape': cv_results['wmape'] if cv_results else None,
                    'cv_smape': cv_results['smape'] if cv_results else None,
                    'cv_bias': cv_results['bias'] if cv_results else None,
                    'cv_mae': cv_results['mae'] if cv_results else None,
                    'wmape_level': get_performance_level('wmape', cv_results['wmape']) if cv_results else None,
                    'smape_level': get_performance_level('smape', cv_results['smape']) if cv_results else None,
                    'bias_level': get_performance_level('bias', cv_results['bias']) if cv_results else None
                },
                'trained_at': datetime.now().isoformat(),
                'aic': result.aic,
                'bic': result.bic,
                'converged': result.mle_retvals['converged'] if hasattr(result, 'mle_retvals') else True
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            return {
                'series_id': series_id,
                'status': 'success',
                'path': str(model_path),
                'order': best_order,
                'seasonal_order': best_seasonal,
                'aic': float(result.aic),
                'wmape': cv_results['wmape'] if cv_results else None,
                'smape': cv_results['smape'] if cv_results else None,
                'bias': cv_results['bias'] if cv_results else None,
                'mae': cv_results['mae'] if cv_results else None,
                'wmape_level': get_performance_level('wmape', cv_results['wmape']) if cv_results else None,
                'sparse_flag': False,
                'sparse_reason': '',
                'n_obs': len(y),
                'nonzero_ratio': float(nonzero_ratio)
            }
        
        return {'series_id': series_id, 'status': 'error', 'reason': 'no_output_dir', 'path': None}
        
    except Exception as e:
        return {'series_id': json_path.stem, 'status': 'error', 'reason': str(e), 'path': None}
def main():
    parser = argparse.ArgumentParser(description="Train base SARIMAX models with auto-optimization v2")
    parser.add_argument("--json-dir", type=str, default="data/features",
                        help="Directory containing series JSON files")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/base_monthly",
                        help="Output directory for trained models")
    parser.add_argument("--train-until", type=int, default=2023,
                        help="Train data until this year (inclusive)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--auto-optimize", action="store_true", default=True,
                        help="Enable auto ARIMA parameter optimization (default: True)")
    parser.add_argument("--no-auto-optimize", action="store_false", dest="auto_optimize",
                        help="Disable auto optimization, use fixed params (1,0,1)(1,0,1,12)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of series to train (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Initialize reproducibility manifest
    manifest = ManifestBuilder(run_id=generate_run_id('P'))
    manifest.set_args(args) \
            .set_git_info() \
            .set_cv_scheme('holdout_6m', val_window=6) \
            .set_seed(args.seed) \
            .set_optuna_config(n_trials=50, timeout=None) \
            .set_sparse_config(threshold=0.5, nonzero_min=0.3) \
            .start()
    
    print("=" * 80)
    print(f"Base Training v2: SARIMAX Models (2021-{args.train_until})")
    print(f"Run ID: {manifest.manifest['run_id']}")
    print(f"Git Commit: {manifest.manifest['git_commit']}")
    print(f"Auto-Optimization: {'ENABLED' if args.auto_optimize else 'DISABLED (fixed params)'}")
    if args.auto_optimize and OPTUNA_AVAILABLE:
        print(f"Optimization Method: Optuna (Bayesian, 50 trials per series)")
    elif args.auto_optimize:
        print(f"Optimization Method: Grid Search (Optuna not available)")
    print("=" * 80)
    
    # JSON 파일 목록
    json_dir = Path(args.json_dir)
    json_files = list(json_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != '_summary.json']
    
    if args.limit:
        json_files = json_files[:args.limit]
    
    print(f"\n[INFO] Found {len(json_files)} series JSON files")
    print(f"[INFO] Training until: {args.train_until}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Parallel workers: {args.max_workers}")
    
    # 병렬 학습
    results = []
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"\n[INFO] Starting training...")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(train_single_series, json_file, args.train_until, args.output_dir, args.auto_optimize): json_file
            for json_file in json_files
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skipped':
                skipped_count += 1
            else:
                error_count += 1
            
            # Progress update every 50 series
            if i % 50 == 0 or i == len(json_files):
                print(f"  Progress: {i}/{len(json_files)} | "
                      f"Success: {success_count} | "
                      f"Skipped: {skipped_count} | "
                      f"Errors: {error_count}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total series:   {len(json_files)}")
    print(f"Success:        {success_count}")
    print(f"Skipped:        {skipped_count}")
    print(f"Errors:         {error_count}")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_path = Path(args.output_dir) / "training_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] Results saved to: {results_path}")
    
    # KPI Summary 생성
    if success_count > 0:
        success_results = results_df[results_df['status'] == 'success'].copy()
        
        kpi_summary = {
            'total_series': len(json_files),
            'successful': success_count,
            'skipped': skipped_count,
            'errors': error_count,
            'metrics': {
                'wmape_mean': float(success_results['wmape'].mean()),
                'wmape_median': float(success_results['wmape'].median()),
                'smape_mean': float(success_results['smape'].mean()),
                'smape_median': float(success_results['smape'].median()),
                'bias_mean': float(success_results['bias'].mean()),
                'bias_median': float(success_results['bias'].median()),
                'mae_mean': float(success_results['mae'].mean())
            },
            'performance_distribution': {
                'wmape_excellent': int((success_results['wmape'] < 20).sum()),
                'wmape_good': int(((success_results['wmape'] >= 20) & (success_results['wmape'] < 50)).sum()),
                'wmape_fair': int(((success_results['wmape'] >= 50) & (success_results['wmape'] < 100)).sum()),
                'wmape_poor': int((success_results['wmape'] >= 100).sum())
            }
        }
        
        kpi_path = Path(args.output_dir) / "kpi_summary.json"
        with open(kpi_path, 'w') as f:
            json.dump(kpi_summary, f, indent=2)
        print(f"[INFO] KPI summary saved to: {kpi_path}")
        
        print(f"\n{'='*80}")
        print("3-Metric KPI Performance")
        print(f"{'='*80}")
        print(f"WMAPE: {kpi_summary['metrics']['wmape_mean']:.2f}% (median: {kpi_summary['metrics']['wmape_median']:.2f}%)")
        print(f"SMAPE: {kpi_summary['metrics']['smape_mean']:.2f}% (median: {kpi_summary['metrics']['smape_median']:.2f}%)")
        print(f"Bias:  {kpi_summary['metrics']['bias_mean']:+.2f}% (median: {kpi_summary['metrics']['bias_median']:+.2f}%)")
        print(f"\nWMAPE Distribution:")
        print(f"  Excellent (<20%):  {kpi_summary['performance_distribution']['wmape_excellent']}")
        print(f"  Good (20-50%):     {kpi_summary['performance_distribution']['wmape_good']}")
        print(f"  Fair (50-100%):    {kpi_summary['performance_distribution']['wmape_fair']}")
        print(f"  Poor (>100%):      {kpi_summary['performance_distribution']['wmape_poor']}")
    
    # 상위 모델 표시
    if success_count > 0:
        success_results = success_results.sort_values('wmape')
        
        print(f"\n[INFO] Top 10 models (by WMAPE):")
        for idx, row in success_results.head(10).iterrows():
            print(f"  {row['series_id'][:50]:50s} | WMAPE: {row['wmape']:5.1f}% | SMAPE: {row['smape']:5.1f}% | Bias: {row['bias']:+5.1f}%")
    
    # 오류 상세
    if error_count > 0:
        print(f"\n[WARNING] {error_count} errors occurred:")
        error_results = results_df[results_df['status'] == 'error']
        for idx, row in error_results.head(5).iterrows():
            print(f"  {row['series_id']}: {row['reason']}")
    
    # Complete manifest
    manifest.add_metadata('total_series', len(json_files)) \
            .add_metadata('successful_series', success_count) \
            .add_metadata('skipped_series', skipped_count) \
            .add_metadata('error_series', error_count)
    
    exit_code = 0 if error_count == 0 else 1
    manifest.finish(exit_code) \
            .save(Path(args.output_dir) / 'manifest.json')
    
    print(f"\n[SUCCESS] Base training v2 completed! (Duration: {manifest.manifest.get('duration_human', 'N/A')})")


if __name__ == '__main__':
    main()
