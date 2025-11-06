"""
Train 파이프라인: features 데이터로 시계열 모델 학습 및 아티팩트 저장
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from src.forecasting import safe_forecast

# 경로 설정
CURATED_PATH = "data/curated/claims_base_2021_2023.parquet"
FEATURES_PATH = "data/features/cycle_features.parquet"
ARTIFACTS_DIR = "artifacts"

# 파라미터 설정
TRAIN_UNTIL_YEAR = 2023  # 2023년까지 학습
TRAIN_UNTIL_WEEK = 52    # 52주까지 학습
FORECAST_HORIZON = 26    # 26주 예측 (6개월)
SEASONAL_ORDER = (0, 1, 1, 52)  # SARIMA 파라미터
CONFIDENCE_INTERVAL = 0.95

print("=" * 80)
print("Train Pipeline - 시계열 모델 학습")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 데이터 로드")
print("-" * 80)
df_curated = pd.read_parquet(CURATED_PATH)
df_features = pd.read_parquet(FEATURES_PATH)
print(f"Curated 데이터: {len(df_curated):,} 행, {df_curated['series_id'].nunique()} 시리즈")
print(f"Features 데이터: {len(df_features):,} 행, {df_features['series_id'].nunique()} 시리즈")

# 2. 학습 데이터 필터링
print("\n[2] 학습 데이터 필터링")
print("-" * 80)
# 기본 상수로 상한을 정하되, 실제 입력된 데이터의 최대 주차를 사용해 과도한 패딩(W53 등)을 피합니다.
effective_max_week_for_year = df_curated.loc[df_curated['year'] == TRAIN_UNTIL_YEAR, 'week'].max()
if pd.isna(effective_max_week_for_year):
    # 해당 연도 데이터가 없으면 상수값 사용
    effective_max_week_for_year = TRAIN_UNTIL_WEEK
else:
    # 정수로 변환
    effective_max_week_for_year = int(effective_max_week_for_year)

train_mask = (
    (df_curated['year'] < TRAIN_UNTIL_YEAR) |
    ((df_curated['year'] == TRAIN_UNTIL_YEAR) & (df_curated['week'] <= effective_max_week_for_year))
)
df_train = df_curated[train_mask].copy()
print(f"학습 데이터: {len(df_train):,} 행")

# 전역 학습 주(week) 시퀀스 생성 - 이후 각 시리즈는 이 시퀀스를 기준으로 재색인(reindex)하여
# 누락된 주는 0으로 채웁니다. (요청하신대로 공백 시리즈는 건너뛰지 않고 0으로 카운트됩니다.)
unique_weeks = (
    df_train[['year', 'week']]
    .drop_duplicates()
    .sort_values(['year', 'week'])
    .reset_index(drop=True)
)

print(f"학습 기간: {unique_weeks.iloc[0]['year']}-W{int(unique_weeks.iloc[0]['week'])} ~ {unique_weeks.iloc[-1]['year']}-W{int(unique_weeks.iloc[-1]['week'])}")

# 3. 시리즈별 모델 학습
print("\n[3] 시리즈별 모델 학습")
print("-" * 80)
artifacts = {
    'metadata': {
        'train_date': datetime.now().isoformat(),
        'train_until_year': TRAIN_UNTIL_YEAR,
        'train_until_week': TRAIN_UNTIL_WEEK,
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
    
    # 시리즈 데이터 추출
    series_data = df_train[df_train['series_id'] == series_id][['year', 'week', 'claim_count']]
    series_data = series_data.sort_values(['year', 'week'])

    # 전역 주(week) 시퀀스에 맞춰 재색인하여 누락된 주는 0으로 채웁니다.
    series_full = unique_weeks.merge(series_data, on=['year', 'week'], how='left')
    series_full['claim_count'] = series_full['claim_count'].fillna(0)

    # 시계열 생성 (float 타입 보장)
    y = series_full['claim_count'].astype(float).values
    
    # 모델 학습 및 예측
    try:
        forecast_result = safe_forecast(
            pd.Series(y),
            horizon=FORECAST_HORIZON,
            seasonal_order=SEASONAL_ORDER,
            ci=CONFIDENCE_INTERVAL
        )
        
        # 통계 계산
        hist_mean = float(np.mean(y))
        hist_std = float(np.std(y))
        hist_max = float(np.max(y))
        hist_min = float(np.min(y))
        nonzero_pct = float(np.sum(y > 0) / len(y) * 100) if len(y) > 0 else 0.0
        
        # 아티팩트 저장
        artifacts['series_models'][series_id] = {
            'model_type': forecast_result['model'],
            'n_train_points': len(y),
            'hist_mean': hist_mean,
            'hist_std': hist_std,
            'hist_max': hist_max,
            'hist_min': hist_min,
            'nonzero_pct': nonzero_pct,
            'last_value': float(y[-1]) if len(y) > 0 else 0.0,
            'forecast': {
                'yhat': forecast_result['yhat'].tolist(),
                'yhat_lower': forecast_result['yhat_lower'].tolist(),
                'yhat_upper': forecast_result['yhat_upper'].tolist()
            }
        }
        
    except Exception as e:
        print(f"\n경고: 시리즈 {series_id} 학습 실패: {e}")
        # 실패 시 naive 예측
        artifacts['series_models'][series_id] = {
            'model_type': 'failed_naive',
            'n_train_points': len(y),
            'hist_mean': float(np.mean(y)) if len(y) > 0 else 0.0,
            'hist_std': float(np.std(y)) if len(y) > 0 else 0.0,
            'error': str(e)
        }

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

with open(artifact_path, 'w', encoding='utf-8') as f:
    json.dump(artifacts, f, indent=2, ensure_ascii=False)

print(f"아티팩트 저장 완료: {artifact_path}")
print(f"파일 크기: {artifact_path.stat().st_size / 1024 / 1024:.2f} MB")

# 6. 학습 요약 저장
print("\n[6] 학습 요약 저장")
print("-" * 80)
summary = {
    'train_date': artifacts['metadata']['train_date'],
    'n_series': n_series,
    'n_samples': len(df_train),
    'train_period': f"{df_train['year'].min()}-W{df_train['week'].min()} ~ {df_train['year'].max()}-W{df_train['week'].max()}",
    'forecast_horizon': FORECAST_HORIZON,
    'model_distribution': model_types,
    'avg_nonzero_pct': np.mean([
        m.get('nonzero_pct', 0) 
        for m in artifacts['series_models'].values()
    ])
}

summary_path = Path(ARTIFACTS_DIR) / "training_summary.json"
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"학습 요약 저장 완료: {summary_path}")

print("\n" + "=" * 80)
print("✓ Train Pipeline 완료!")
print("=" * 80)
print(f"✓ {n_series} 개 시리즈 학습 완료")
print(f"✓ 예측 기간: {FORECAST_HORIZON} 주")
print(f"✓ 아티팩트: {artifact_path}")
print("=" * 80)
