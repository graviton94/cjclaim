"""
월별 예측 생성 파이프라인
학습된 모델로 다음 4-13주 예측 생성

출력: artifacts/forecasts/YYYY/forecast_YYYY_MM.parquet
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


def forecast_single_series(args_tuple):
    """
    단일 시리즈 예측
    
    Parameters:
    -----------
    args_tuple : tuple
        (series_id, model_path, horizon, year, month)
    """
    series_id, model_path, horizon, year, month = args_tuple
    
    try:
        # 모델 로드
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # JSON에서 학습 데이터 로드
        json_path = Path("data/features") / f"{series_id.replace('/', '_').replace('\\', '_').replace(':', '_').replace('|', '_').replace('?', '_').replace('*', '_').replace('<', '_').replace('>', '_').replace('\"', '_')}.json"
        
        if not json_path.exists():
            return {'status': 'error', 'reason': f'JSON not found: {json_path}'}
        
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            series_data = json.load(f)
        
        # 학습 데이터 추출
        data_records = series_data.get('data', [])
        df = pd.DataFrame(data_records)
        
        # 2021-2023 데이터만 (베이스 모델 학습 범위)
        df_train = df[(df['year'] >= 2021) & (df['year'] <= 2023)].copy()
        
        if len(df_train) < 52:
            return {'status': 'error', 'reason': 'insufficient training data'}
        
        y = df_train['claim_count'].values
        
        # SARIMAX 모델 재구성
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        model_spec = model_info.get('model_spec', {})
        order = model_spec.get('order', (1, 0, 1))
        seasonal_order = model_spec.get('seasonal_order', (1, 0, 1, 52))
        
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=model_spec.get('enforce_stationarity', False),
            enforce_invertibility=model_spec.get('enforce_invertibility', False)
        )
        
        # 저장된 파라미터로 모델 초기화
        params = np.array(model_info.get('params', []))
        
        # smoothed_state를 사용하여 예측
        fitted_model = model.smooth(params)
        
        # 예측
        forecast = fitted_model.forecast(steps=horizon)
        forecast = np.maximum(forecast, 0)  # 음수 방지
        
        # 신뢰구간
        forecast_result = fitted_model.get_forecast(steps=horizon)
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI
        
        # 주차 계산 (대상 월의 첫 주부터)
        year_int = int(year)
        month_int = int(month)
        
        # 월의 첫 날
        first_day = datetime(year_int, month_int, 1)
        
        # ISO week 계산
        weeks = []
        for i in range(horizon):
            date = first_day + timedelta(weeks=i)
            iso_year, iso_week, _ = date.isocalendar()
            weeks.append({'year': iso_year, 'week': iso_week})
        
        # 결과 DataFrame
        forecasts = []
        for i in range(horizon):
            forecasts.append({
                'series_id': series_id,
                'year': weeks[i]['year'],
                'week': weeks[i]['week'],
                'y_pred': float(forecast[i]),
                'y_pred_lower': float(conf_int.iloc[i, 0]),
                'y_pred_upper': float(conf_int.iloc[i, 1]),
                'forecast_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return {
            'status': 'success',
            'forecasts': forecasts
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'series_id': series_id,
            'error': str(e),
            'forecasts': []
        }


def generate_monthly_forecast(
    year: int,
    month: int,
    models_dir: str = "artifacts/models/base_2021_2023",
    output_dir: str = None,
    horizon: int = 8,  # 다음 8주 (약 2개월)
    max_workers: int = 4
):
    """
    월별 예측 생성
    
    Parameters:
    -----------
    year : int
        대상 연도
    month : int
        대상 월
    models_dir : str
        모델 디렉토리
    output_dir : str
        출력 디렉토리 (기본: artifacts/forecasts/YYYY/)
    horizon : int
        예측 주차 수
    max_workers : int
        병렬 worker 수
    """
    print("=" * 80)
    print(f"월별 예측 생성: {year}년 {month}월")
    print("=" * 80)
    print(f"Horizon: {horizon}주")
    print(f"Workers: {max_workers}")
    
    models_dir = Path(models_dir)
    
    if output_dir is None:
        output_dir = Path(f"artifacts/forecasts/{year}")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 파일 목록
    model_files = list(models_dir.glob("*.pkl"))
    print(f"총 모델: {len(model_files):,}개")
    
    # 예측 작업 준비
    tasks = []
    for model_path in model_files:
        series_id = model_path.stem
        tasks.append((series_id, model_path, horizon, year, month))
    
    print(f"\n예측 시작...")
    
    # 병렬 처리
    all_forecasts = []
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(forecast_single_series, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            
            if result['status'] == 'success':
                all_forecasts.extend(result['forecasts'])
                success_count += 1
            else:
                error_count += 1
            
            if i % 100 == 0:
                print(f"  진행: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) - "
                      f"성공: {success_count}, 실패: {error_count}")
    
    print(f"\n예측 완료!")
    print(f"  성공: {success_count:,}개")
    print(f"  실패: {error_count:,}개")
    print(f"  총 예측 레코드: {len(all_forecasts):,}개")
    
    if not all_forecasts:
        print("  ⚠️  예측 결과 없음")
        return None
    
    # DataFrame 변환
    df_forecast = pd.DataFrame(all_forecasts)
    
    # 저장
    output_file = output_dir / f"forecast_{year}_{month:02d}.parquet"
    df_forecast.to_parquet(output_file, index=False)
    print(f"  ✅ 저장: {output_file}")
    
    # CSV도 저장 (확인용)
    csv_file = output_dir / f"forecast_{year}_{month:02d}.csv"
    df_forecast.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ CSV 저장: {csv_file}")
    
    # 통계
    print(f"\n예측 통계:")
    print(f"  시리즈 수: {df_forecast['series_id'].nunique():,}개")
    print(f"  주차 범위: {df_forecast['week'].min()}-{df_forecast['week'].max()}")
    print(f"  평균 예측값: {df_forecast['y_pred'].mean():.2f}")
    print(f"  예측값 합계: {df_forecast['y_pred'].sum():.0f}")
    
    return df_forecast


def main():
    parser = argparse.ArgumentParser(description="월별 예측 생성 파이프라인")
    parser.add_argument("--year", type=int, required=True, help="연도")
    parser.add_argument("--month", type=int, required=True, help="월")
    parser.add_argument("--models-dir", type=str, default="artifacts/models/base_2021_2023",
                       help="모델 디렉토리")
    parser.add_argument("--output-dir", type=str, help="출력 디렉토리 (기본: artifacts/forecasts/YYYY/)")
    parser.add_argument("--horizon", type=int, default=8, help="예측 주차 수")
    parser.add_argument("--max-workers", type=int, default=4, help="병렬 worker 수")
    
    args = parser.parse_args()
    
    df_forecast = generate_monthly_forecast(
        year=args.year,
        month=args.month,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        horizon=args.horizon,
        max_workers=args.max_workers
    )
    
    if df_forecast is not None and len(df_forecast) > 0:
        print("\n" + "=" * 80)
        print("✅ 예측 생성 완료!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ 예측 생성 실패")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
