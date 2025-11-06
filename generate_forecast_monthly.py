"""
월별 예측 생성 스크립트
- 훈련된 base_monthly 모델 로드
- 6개월 예측 생성 (horizon=6 months)
- forecast parquet 저장
"""
import argparse
from pathlib import Path
import pickle
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


def forecast_single_series(
    model_path: Path,
    json_dir: Path,
    forecast_year: int,
    forecast_month: int,
    horizon: int = 6
) -> dict:
    """
    단일 시리즈에 대해 예측 수행
    
    Args:
        model_path: 모델 pkl 파일 경로
        json_dir: JSON 데이터 디렉토리
        forecast_year: 예측 시작 연도
        forecast_month: 예측 시작 월
        horizon: 예측 개월 수 (default=6)
    
    Returns:
        dict with status, series_id, forecasts
    """
    try:
        # 모델 정보 로드
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        series_id = model_info['series_id']
        
        # JSON 데이터 로드
        safe_name = series_id.replace('/', '_').replace('|', '_').replace('\\', '_')
        json_path = json_dir / f"{safe_name}.json"
        
        if not json_path.exists():
            return {
                'status': 'error',
                'series_id': series_id,
                'reason': 'json_not_found'
            }
        
        with open(json_path, 'r', encoding='utf-8') as f:
            series_data = json.load(f)
        
        # 훈련 데이터 필터링 (2021-2023)
        df = pd.DataFrame(series_data['data'])
        df_train = df[(df['year'] >= 2021) & (df['year'] <= 2023)].copy()
        
        if len(df_train) < 12:
            return {
                'status': 'error',
                'series_id': series_id,
                'reason': 'insufficient_data',
                'data_points': len(df_train)
            }
        
        # 예측 수행
        y = df_train['claim_count'].values
        
        # 희소 시리즈 필터링: 월 평균 0.5건 이하는 스킵
        avg_claims_per_month = y.mean()
        if avg_claims_per_month < 0.5:
            return {
                'status': 'skipped',
                'series_id': series_id,
                'reason': f'sparse_series (avg={avg_claims_per_month:.2f}/month)',
                'forecast': None
            }
        
        model = SARIMAX(
            y,
            order=model_info['model_spec']['order'],
            seasonal_order=model_info['model_spec']['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        params = np.array(model_info['params'])
        fitted = model.smooth(params)
        
        # horizon개월 예측
        forecast = fitted.forecast(steps=horizon)
        
        # 음수 클리핑
        forecast = np.maximum(forecast, 0)
        
        # 예측 월 계산
        forecast_dates = []
        current_year = forecast_year
        current_month = forecast_month
        
        for _ in range(horizon):
            forecast_dates.append({
                'year': current_year,
                'month': current_month
            })
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        # 결과 포맷팅
        forecasts = []
        for date_info, pred_value in zip(forecast_dates, forecast):
            forecasts.append({
                'year': date_info['year'],
                'month': date_info['month'],
                'forecast': float(pred_value)
            })
        
        return {
            'status': 'success',
            'series_id': series_id,
            'plant': series_data.get('plant'),
            'product_cat2': series_data.get('product_cat2'),
            'mid_category': series_data.get('mid_category'),
            'forecasts': forecasts
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'series_id': model_path.stem,
            'reason': f'{type(e).__name__}: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(description='Generate monthly forecasts from trained models')
    parser.add_argument('--year', type=int, required=True, help='Forecast start year')
    parser.add_argument('--month', type=int, required=True, help='Forecast start month (1-12)')
    parser.add_argument('--horizon', type=int, default=6, help='Forecast horizon in months (default=6)')
    parser.add_argument('--models-dir', type=str, default='artifacts/models/base_monthly',
                        help='Directory containing trained model pkl files')
    parser.add_argument('--json-dir', type=str, default='data/features',
                        help='Directory containing series JSON files')
    parser.add_argument('--output-dir', type=str, default='artifacts/forecasts',
                        help='Output directory for forecast parquet files')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum parallel workers')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    
    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        return
    
    if not json_dir.exists():
        print(f"[ERROR] JSON directory not found: {json_dir}")
        return
    
    # 출력 디렉토리 생성
    output_year_dir = output_dir / str(args.year)
    output_year_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 파일 검색
    model_files = list(models_dir.glob("*.pkl"))
    
    print(f"\n=== Monthly Forecast Generation ===")
    print(f"Models directory: {models_dir}")
    print(f"JSON directory: {json_dir}")
    print(f"Output directory: {output_year_dir}")
    print(f"Forecast start: {args.year}-{args.month:02d}")
    print(f"Horizon: {args.horizon} months")
    print(f"Total models: {len(model_files)}")
    print(f"Max workers: {args.max_workers}")
    print()
    
    # 병렬 예측 실행
    results = []
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                forecast_single_series,
                model_path,
                json_dir,
                args.year,
                args.month,
                args.horizon
            ): model_path for model_path in model_files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            else:
                error_count += 1
            
            if i % 50 == 0:
                print(f"Progress: {i}/{len(model_files)} | Success: {success_count} | Failed: {error_count}")
    
    print(f"\n=== Forecast Results ===")
    print(f"Success: {success_count}")
    print(f"Failed: {error_count}")
    
    # 실패 샘플 출력
    errors = [r for r in results if r['status'] == 'error']
    if errors:
        print(f"\n[ERROR] Failed samples (first 5):")
        for err in errors[:5]:
            print(f"  - {err.get('series_id', 'unknown')}: {err.get('reason', 'unknown error')}")
    
    # 성공한 예측만 parquet 저장
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("\n[WARNING] No successful forecasts to save")
        return
    
    # DataFrame 변환
    rows = []
    for result in successful:
        for forecast_item in result['forecasts']:
            rows.append({
                'series_id': result['series_id'],
                'plant': result['plant'],
                'product_cat2': result['product_cat2'],
                'mid_category': result['mid_category'],
                'forecast_year': forecast_item['year'],
                'forecast_month': forecast_item['month'],
                'forecast_value': forecast_item['forecast']
            })
    
    df_forecast = pd.DataFrame(rows)
    
    # 출력 파일명
    output_file = output_year_dir / f"forecast_{args.year}_{args.month:02d}.parquet"
    df_forecast.to_parquet(output_file, index=False, engine='pyarrow')
    
    print(f"\n=== Output ===")
    print(f"File: {output_file}")
    print(f"Total forecast records: {len(df_forecast)}")
    print(f"Unique series: {df_forecast['series_id'].nunique()}")
    print(f"Forecast months: {df_forecast['forecast_month'].min()}-{df_forecast['forecast_month'].max()}")
    print("\nSample forecasts:")
    print(df_forecast.head(12))
    print("\n[SUCCESS] Forecast generation completed!")


if __name__ == "__main__":
    main()
