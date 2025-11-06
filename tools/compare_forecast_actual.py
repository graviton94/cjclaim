"""
예측-실측 비교 및 KPI 계산
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json


def calculate_metrics(actual, predicted):
    """시계열 정확도 메트릭 계산"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 0이 아닌 실측값만 사용 (MAPE 계산)
    mask = actual != 0
    
    metrics = {}
    
    # MAPE (Mean Absolute Percentage Error)
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        metrics['MAPE'] = float(mape)
    else:
        metrics['MAPE'] = None
    
    # Bias (평균 오차율)
    if actual.sum() != 0:
        bias = (predicted.sum() - actual.sum()) / actual.sum()
        metrics['Bias'] = float(bias)
    else:
        metrics['Bias'] = None
    
    # MAE (Mean Absolute Error)
    metrics['MAE'] = float(np.mean(np.abs(actual - predicted)))
    
    # RMSE (Root Mean Squared Error)
    metrics['RMSE'] = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    
    # R² (결정계수)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot != 0:
        metrics['R2'] = float(1 - (ss_res / ss_tot))
    else:
        metrics['R2'] = None
    
    return metrics


def compare_forecast_vs_actual(
    actual_file: str,
    forecast_file: str,
    year: int,
    month: int,
    output_dir: str = "logs"
):
    """
    월별 예측-실측 비교
    
    Parameters:
    -----------
    actual_file : str
        실제 데이터 파일 (filtered CSV 또는 candidates CSV)
    forecast_file : str
        예측 데이터 파일 (parquet)
    year : int
        대상 연도
    month : int
        대상 월
    
    Returns:
    --------
    dict: 시리즈별 메트릭
    """
    
    print(f"  예측-실측 비교 시작...")
    
    # 실제 데이터 로드 및 주간 집계
    df_actual = pd.read_csv(actual_file, encoding='utf-8-sig')
    df_actual['발생일자'] = pd.to_datetime(df_actual['발생일자'])
    df_actual['year'] = df_actual['발생일자'].dt.isocalendar().year
    df_actual['week'] = df_actual['발생일자'].dt.isocalendar().week
    
    # 주간 집계
    actual_weekly = df_actual.groupby(['플랜트', '제품범주2', '중분류(보정)', 'year', 'week']).agg({
        'count': 'sum'
    }).reset_index()
    actual_weekly['series_id'] = (actual_weekly['플랜트'] + '|' + 
                                   actual_weekly['제품범주2'] + '|' + 
                                   actual_weekly['중분류(보정)'])
    
    # 예측 데이터 로드
    forecast_path = Path(forecast_file)
    if not forecast_path.exists():
        print(f"  ⚠️ 예측 파일 없음: {forecast_file}")
        return {}
    
    df_forecast = pd.read_parquet(forecast_file)
    df_forecast = df_forecast[(df_forecast['year'] == year) & 
                              (df_forecast['week'] >= 1) & 
                              (df_forecast['week'] <= 53)]
    
    # 시리즈별 비교
    series_metrics = {}
    
    for series_id in actual_weekly['series_id'].unique():
        actual_series = actual_weekly[actual_weekly['series_id'] == series_id]
        forecast_series = df_forecast[df_forecast['series_id'] == series_id]
        
        if len(forecast_series) == 0:
            continue
        
        # 주차별 매칭
        merged = actual_series.merge(
            forecast_series[['year', 'week', 'y_pred']],
            on=['year', 'week'],
            how='outer'
        )
        merged['count'] = merged['count'].fillna(0)
        merged['y_pred'] = merged['y_pred'].fillna(0)
        
        # 메트릭 계산
        metrics = calculate_metrics(
            merged['count'].values,
            merged['y_pred'].values
        )
        
        metrics['n_weeks'] = len(merged)
        metrics['total_actual'] = float(merged['count'].sum())
        metrics['total_predicted'] = float(merged['y_pred'].sum())
        
        series_metrics[series_id] = metrics
    
    # 로그 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / f"predict_vs_actual_{year}_{month:02d}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            'year': year,
            'month': month,
            'series_count': len(series_metrics),
            'series_metrics': series_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 비교 완료: {len(series_metrics)}개 시리즈")
    print(f"     로그 저장: {log_file}")
    
    # 전체 KPI 계산
    if series_metrics:
        valid_mapes = [m['MAPE'] for m in series_metrics.values() if m['MAPE'] is not None]
        valid_bias = [m['Bias'] for m in series_metrics.values() if m['Bias'] is not None]
        
        if valid_mapes:
            avg_mape = np.mean(valid_mapes)
            print(f"     평균 MAPE: {avg_mape:.2f}%")
        
        if valid_bias:
            avg_bias = np.mean(valid_bias)
            print(f"     평균 Bias: {avg_bias:.4f}")
    
    return series_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True)
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()
    
    metrics = compare_forecast_vs_actual(
        args.actual, args.forecast, args.year, args.month
    )
