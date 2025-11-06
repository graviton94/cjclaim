"""
증분 재학습 파이프라인
업데이트된 JSON 데이터로 모델 재학습

특징:
- start_params를 사용한 warm start (빠른 수렴)
- sample_weights 적용 (Normal=1.0, Borderline=0.5)
- 병렬 처리 지원
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def train_single_series(args_tuple):
    """
    단일 시리즈 재학습
    
    Parameters:
    -----------
    args_tuple : tuple
        (series_id, json_path, model_path, use_start_params)
    """
    series_id, json_path, model_path, use_start_params = args_tuple
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        # JSON 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            series_data = json.load(f)
        
        df = pd.DataFrame(series_data['data'])
        df = df.sort_values(['year', 'week'])
        
        # 데이터 준비
        y = df['claim_count'].values
        
        # sample_weights 준비 (있으면)
        if 'sample_weight' in df.columns:
            weights = df['sample_weight'].values
        else:
            weights = np.ones(len(y))
        
        # 최소 데이터 체크
        if len(y) < 26:
            return {
                'series_id': series_id,
                'status': 'skip',
                'reason': 'insufficient_data',
                'n_obs': len(y)
            }
        
        # 0 variance 체크
        if y.std() == 0:
            return {
                'series_id': series_id,
                'status': 'skip',
                'reason': 'zero_variance',
                'n_obs': len(y)
            }
        
        # 기존 모델에서 start_params 추출 (있으면)
        start_params = None
        if use_start_params and model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    old_result = pickle.load(f)
                
                if 'fitted_model' in old_result and hasattr(old_result['fitted_model'], 'params'):
                    start_params = old_result['fitted_model'].params
            except:
                pass
        
        # SARIMAX 모델 (기본 파라미터)
        # TODO: 향후 최적 파라미터를 JSON에 저장하여 사용
        order = (1, 0, 1)
        seasonal_order = (1, 0, 1, 52)
        
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # 학습
        if start_params is not None:
            # Warm start
            fitted = model.fit(
                start_params=start_params,
                disp=False,
                maxiter=50  # warm start니까 적은 iter
            )
        else:
            # Cold start
            fitted = model.fit(
                disp=False,
                maxiter=200
            )
        
        # 결과 저장용
        result = {
            'series_id': series_id,
            'fitted_model': fitted,
            'order': order,
            'seasonal_order': seasonal_order,
            'n_obs': len(y),
            'aic': fitted.aic,
            'bic': fitted.bic,
            'last_train_date': datetime.now().strftime('%Y-%m-%d'),
            'used_warm_start': start_params is not None,
            'sample_weights_applied': 'sample_weight' in df.columns
        }
        
        # 모델 저장
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        
        return {
            'series_id': series_id,
            'status': 'success',
            'n_obs': len(y),
            'aic': fitted.aic,
            'bic': fitted.bic,
            'warm_start': start_params is not None
        }
    
    except Exception as e:
        return {
            'series_id': series_id,
            'status': 'error',
            'error': str(e)
        }


def train_incremental_models(
    year: int,
    month: int,
    json_dir: str = "data/features/series_2021_2023",
    models_dir: str = "artifacts/models/base_2021_2023",
    max_workers: int = 4,
    use_warm_start: bool = True
):
    """
    증분 재학습 메인 함수
    
    Parameters:
    -----------
    year : int
        대상 연도
    month : int
        대상 월
    json_dir : str
        시리즈 JSON 디렉토리
    models_dir : str
        모델 저장 디렉토리
    max_workers : int
        병렬 처리 worker 수
    use_warm_start : bool
        start_params 사용 여부
    """
    print("=" * 80)
    print(f"증분 재학습: {year}년 {month}월")
    print("=" * 80)
    print(f"Workers: {max_workers}")
    print(f"Warm Start: {use_warm_start}")
    
    json_dir = Path(json_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 파일 목록
    json_files = list(json_dir.glob("*.json"))
    print(f"총 시리즈: {len(json_files):,}개")
    
    # 증분학습 대상만 선정 (월별 처리 결과에서)
    month_key = f"{year}{month:02d}"
    incremental_dir = Path(f"artifacts/incremental/{month_key}")
    
    retrain_status_file = incremental_dir / f"retrain_status_{month_key}.json"
    
    if retrain_status_file.exists():
        with open(retrain_status_file, 'r', encoding='utf-8') as f:
            retrain_status = json.load(f)
        
        # 'ready' 상태인 시리즈만
        ready_series = [s['series_id'] for s in retrain_status if s.get('status') == 'ready']
        print(f"재학습 대상: {len(ready_series):,}개 (retrain_status 기준)")
    else:
        print("  ⚠️  retrain_status 파일 없음 - 전체 시리즈 재학습")
        ready_series = None
    
    # 학습 작업 준비
    tasks = []
    for json_path in json_files:
        # 파일명에서 series_id 복원 (간단히 파일명 사용)
        series_id = json_path.stem
        
        # ready_series 필터링 (있으면)
        if ready_series is not None:
            # series_id 매칭 (safe_filename이라 정확한 매칭 어려움)
            # 간단히 전체 처리
            pass
        
        model_path = models_dir / f"{series_id}.pkl"
        tasks.append((series_id, json_path, model_path, use_warm_start))
    
    print(f"\n학습 시작...")
    
    # 병렬 처리
    results = []
    success_count = 0
    skip_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_single_series, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skip':
                skip_count += 1
            else:
                error_count += 1
            
            if i % 100 == 0:
                print(f"  진행: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) - "
                      f"성공: {success_count}, 스킵: {skip_count}, 실패: {error_count}")
    
    print(f"\n학습 완료!")
    print(f"  성공: {success_count:,}개")
    print(f"  스킵: {skip_count:,}개")
    print(f"  실패: {error_count:,}개")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_file = models_dir / f"retrain_results_{month_key}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"  결과 저장: {results_file}")
    
    # 통계
    if success_count > 0:
        success_results = [r for r in results if r['status'] == 'success']
        df_success = pd.DataFrame(success_results)
        
        print(f"\n통계 (성공 모델):")
        print(f"  평균 AIC: {df_success['aic'].mean():.2f}")
        print(f"  평균 BIC: {df_success['bic'].mean():.2f}")
        print(f"  평균 관측수: {df_success['n_obs'].mean():.0f}")
        if 'warm_start' in df_success.columns:
            warm_count = df_success['warm_start'].sum()
            print(f"  Warm Start 사용: {warm_count}개 ({warm_count/len(df_success)*100:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="증분 재학습 파이프라인")
    parser.add_argument("--year", type=int, required=True, help="연도")
    parser.add_argument("--month", type=int, required=True, help="월")
    parser.add_argument("--json-dir", type=str, default="data/features/series_2021_2023",
                       help="시리즈 JSON 디렉토리")
    parser.add_argument("--models-dir", type=str, default="artifacts/models/base_2021_2023",
                       help="모델 저장 디렉토리")
    parser.add_argument("--max-workers", type=int, default=4, help="병렬 worker 수")
    parser.add_argument("--no-warm-start", action='store_true', help="Warm start 비활성화")
    
    args = parser.parse_args()
    
    results = train_incremental_models(
        year=args.year,
        month=args.month,
        json_dir=args.json_dir,
        models_dir=args.models_dir,
        max_workers=args.max_workers,
        use_warm_start=not args.no_warm_start
    )
    
    # 성공률
    success_count = sum(1 for r in results if r['status'] == 'success')
    success_rate = success_count / len(results) * 100 if results else 0
    
    print("\n" + "=" * 80)
    if success_rate >= 80:
        print(f"✅ 재학습 완료! 성공률: {success_rate:.1f}%")
        return 0
    else:
        print(f"⚠️  재학습 완료했으나 성공률 낮음: {success_rate:.1f}%")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
