"""
EWS (Early Warning System) 스코어링 모듈
- 시리즈별 상대적 위험도 점수 계산 (0-100)
- Top 5 위험 시리즈 랭킹
- 다차원 평가: 증가율, 변동성, 계절성, 트렌드 가속도
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import json


def calculate_growth_rate(forecast_mean: float, historical_mean: float) -> float:
    """
    성장률 점수 (0-100)
    - 과거 평균 대비 예측 평균의 증가율
    - 100% 이상 증가 시 만점
    """
    if historical_mean < 0.01:
        return 0.0
    
    growth = (forecast_mean - historical_mean) / historical_mean * 100
    
    # 음수는 0점, 100% 이상은 100점
    score = np.clip(growth, 0, 100)
    return score


def calculate_volatility_score(forecast_values: np.ndarray, historical_std: float, historical_mean: float) -> float:
    """
    변동성 점수 (0-100)
    - 예측값들이 과거 표준편차 범위를 얼마나 초과하는지
    """
    if historical_std < 0.01 or historical_mean < 0.01:
        return 0.0
    
    # 예측값 중 historical_mean + 2*std를 초과하는 비율
    threshold = historical_mean + 2 * historical_std
    exceed_ratio = (forecast_values > threshold).sum() / len(forecast_values)
    
    # 50% 이상 초과 시 만점
    score = np.clip(exceed_ratio * 200, 0, 100)
    return score


def calculate_seasonality_score(forecast_values: np.ndarray, 
                                 historical_same_months: np.ndarray) -> float:
    """
    계절성 점수 (0-100)
    - 동일 월(전년도) 대비 예측값의 증가율
    """
    if len(historical_same_months) == 0 or historical_same_months.mean() < 0.01:
        return 0.0
    
    historical_seasonal_mean = historical_same_months.mean()
    forecast_mean = forecast_values.mean()
    
    seasonal_growth = (forecast_mean - historical_seasonal_mean) / historical_seasonal_mean * 100
    
    score = np.clip(seasonal_growth, 0, 100)
    return score


def calculate_acceleration_score(recent_slope: float, forecast_slope: float) -> float:
    """
    가속도 점수 (0-100)
    - 최근 트렌드 대비 예측 트렌드의 가속도
    - 기울기가 급격히 증가하면 높은 점수
    """
    if abs(recent_slope) < 0.001:
        return 0.0
    
    # 예측 기울기가 최근 기울기보다 얼마나 큰지
    acceleration = (forecast_slope - recent_slope) / abs(recent_slope) * 100
    
    score = np.clip(acceleration, 0, 100)
    return score


def calculate_ews_score(
    series_id: str,
    forecast_values: np.ndarray,
    historical_data: np.ndarray,
    weights: Dict[str, float] = None
) -> Dict:
    """
    종합 EWS 점수 계산
    
    Args:
        series_id: 시리즈 ID
        forecast_values: 예측값 배열 (6개월)
        historical_data: 과거 데이터 배열 (2021-2023, 36개월)
        weights: 점수 가중치 {'growth': 0.3, 'volatility': 0.25, ...}
    
    Returns:
        {
            'series_id': str,
            'total_score': float (0-100),
            'growth_score': float,
            'volatility_score': float,
            'seasonality_score': float,
            'acceleration_score': float,
            'forecast_mean': float,
            'historical_mean': float,
            'details': dict
        }
    """
    if weights is None:
        weights = {
            'growth': 0.35,      # 증가율 - 가장 중요
            'volatility': 0.25,  # 변동성
            'seasonality': 0.20, # 계절성
            'acceleration': 0.20 # 가속도
        }
    
    # 기본 통계
    historical_mean = historical_data.mean()
    historical_std = historical_data.std()
    forecast_mean = forecast_values.mean()
    
    # 1. 성장률 점수
    growth_score = calculate_growth_rate(forecast_mean, historical_mean)
    
    # 2. 변동성 점수
    volatility_score = calculate_volatility_score(forecast_values, historical_std, historical_mean)
    
    # 3. 계절성 점수 (예측 6개월과 같은 월의 과거 데이터)
    # 2021-2023 데이터에서 1-6월 추출
    historical_same_months = historical_data[:6]  # 간단히 처음 6개월 사용
    seasonality_score = calculate_seasonality_score(forecast_values, historical_same_months)
    
    # 4. 가속도 점수
    # 최근 6개월 vs 예측 6개월의 선형 트렌드
    recent_6months = historical_data[-6:]
    recent_slope = np.polyfit(range(6), recent_6months, 1)[0]
    forecast_slope = np.polyfit(range(6), forecast_values, 1)[0]
    acceleration_score = calculate_acceleration_score(recent_slope, forecast_slope)
    
    # 종합 점수
    total_score = (
        growth_score * weights['growth'] +
        volatility_score * weights['volatility'] +
        seasonality_score * weights['seasonality'] +
        acceleration_score * weights['acceleration']
    )
    
    return {
        'series_id': series_id,
        'total_score': float(total_score),
        'growth_score': float(growth_score),
        'volatility_score': float(volatility_score),
        'seasonality_score': float(seasonality_score),
        'acceleration_score': float(acceleration_score),
        'forecast_mean': float(forecast_mean),
        'historical_mean': float(historical_mean),
        'forecast_max': float(forecast_values.max()),
        'historical_max': float(historical_data.max()),
        'growth_rate_pct': float((forecast_mean - historical_mean) / historical_mean * 100) if historical_mean > 0 else 0,
        'details': {
            'weights': weights,
            'recent_slope': float(recent_slope),
            'forecast_slope': float(forecast_slope)
        }
    }


def rank_ews_scores(scores: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    EWS 점수 기준 Top N 랭킹
    
    Args:
        scores: calculate_ews_score 결과 리스트
        top_n: 상위 N개
    
    Returns:
        Top N 시리즈 (total_score 내림차순, rank 추가)
    """
    df = pd.DataFrame(scores)
    df_sorted = df.sort_values('total_score', ascending=False).head(top_n)
    
    # 랭킹 추가
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    
    return df_sorted.to_dict('records')


def generate_ews_report(
    forecast_parquet_path: str,
    json_dir: str,
    metadata_path: str = "artifacts/models/base_monthly/training_results.csv",
    output_path: str = None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    전체 시리즈 EWS 리포트 생성
    
    Args:
        forecast_parquet_path: 예측 parquet 파일 경로
        json_dir: JSON 데이터 디렉토리
        metadata_path: 모델 metadata CSV 경로 (신뢰도 계산용)
        output_path: 결과 저장 경로 (optional)
        top_n: Top N 랭킹
    
    Returns:
        전체 EWS 점수 DataFrame
    """
    # 예측 데이터 로드
    df_forecast = pd.read_parquet(forecast_parquet_path)
    
    # 모델 metadata 로드 (신뢰도 계산용)
    try:
        df_metadata = pd.read_csv(metadata_path)
    except:
        df_metadata = None
    
    # 시리즈별 EWS 점수 계산
    ews_scores = []
    
    for series_id in df_forecast['series_id'].unique():
        # 예측값 및 시점
        series_forecast = df_forecast[df_forecast['series_id'] == series_id].copy()
        forecast_values = series_forecast['forecast_value'].values
        
        # 최대값이 나올 시점 찾기
        max_idx = forecast_values.argmax()
        max_row = series_forecast.iloc[max_idx]
        expected_time = f"{int(max_row['forecast_year'])}-{int(max_row['forecast_month']):02d}"
        expected_count = float(forecast_values[max_idx])
        
        # 신뢰도 계산 (모델 metadata에서)
        if df_metadata is not None:
            model_info = df_metadata[df_metadata['series_id'] == series_id]
            if len(model_info) > 0 and 'mape' in model_info.columns:
                mape = model_info.iloc[0]['mape']
                # MAPE가 비정상적으로 크면 0으로 처리
                if pd.isna(mape) or mape > 1000:
                    confidence = 0.0
                else:
                    confidence = max(0, 100 - mape)
            else:
                confidence = 0.0
        else:
            confidence = 0.0
        
        # 과거 데이터 로드
        safe_name = series_id.replace('/', '_').replace('|', '_').replace('\\', '_')
        json_path = Path(json_dir) / f"{safe_name}.json"
        
        if not json_path.exists():
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df_hist = pd.DataFrame(data['data'])
        df_train = df_hist[(df_hist['year'] >= 2021) & (df_hist['year'] <= 2023)]
        historical_data = df_train['claim_count'].values
        
        if len(historical_data) < 12:
            continue
        
        # 희소 시리즈 필터링: 월 평균 0.5건 이하는 제외
        avg_claims_per_month = historical_data.mean()
        if avg_claims_per_month < 0.5:
            continue
        
        # EWS 점수 계산
        score = calculate_ews_score(series_id, forecast_values, historical_data)
        
        # 신뢰도, 예상시점, 예상건수 추가
        score['confidence'] = round(confidence, 1)
        score['expected_time'] = expected_time
        score['expected_count'] = round(expected_count, 2)
        
        ews_scores.append(score)
    
    # DataFrame 변환
    df_ews = pd.DataFrame(ews_scores)
    df_ews = df_ews.sort_values('total_score', ascending=False)
    df_ews['rank'] = range(1, len(df_ews) + 1)
    
    # 저장
    if output_path:
        df_ews.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[SUCCESS] EWS report saved: {output_path}")
    
    # Top N 출력
    print(f"\n{'='*80}")
    print(f"EWS TOP {top_n} 위험 시리즈")
    print(f"{'='*80}")
    
    top_series = df_ews.head(top_n)
    for _, row in top_series.iterrows():
        print(f"\n[{int(row['rank'])}위] {row['series_id']}")
        print(f"  종합 점수: {row['total_score']:.1f}/100")
        print(f"  - 증가율: {row['growth_score']:.1f} (평균 {row['historical_mean']:.1f} → {row['forecast_mean']:.1f}, {row['growth_rate_pct']:+.1f}%)")
        print(f"  - 변동성: {row['volatility_score']:.1f}")
        print(f"  - 계절성: {row['seasonality_score']:.1f}")
        print(f"  - 가속도: {row['acceleration_score']:.1f}")
    
    return df_ews


if __name__ == "__main__":
    # 테스트 실행
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate EWS scores and rankings")
    parser.add_argument("--forecast", type=str, required=True,
                        help="Forecast parquet file path")
    parser.add_argument("--json-dir", type=str, default="data/features",
                        help="JSON data directory")
    parser.add_argument("--metadata", type=str, 
                        default="artifacts/models/base_monthly/training_results.csv",
                        help="Model metadata CSV path")
    parser.add_argument("--output", type=str, default="artifacts/metrics/ews_scores.csv",
                        help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top series to display")
    
    args = parser.parse_args()
    
    df_ews = generate_ews_report(
        forecast_parquet_path=args.forecast,
        json_dir=args.json_dir,
        metadata_path=args.metadata,
        output_path=args.output,
        top_n=args.top_n
    )
