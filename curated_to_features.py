"""
curated parquet → 사이클 특성 추출 → features parquet 자동화 스크립트
"""
import pandas as pd
import numpy as np
from src.cycle_features import compute_psi, detect_peaks, detect_changepoints, compute_amplitude
from contracts import validate_features

CURATED_PATH = "data/curated/claims.parquet"
FEATURES_PATH = "data/features/cycle_features.parquet"

print(f"[INFO] Loading curated data: {CURATED_PATH}")
df_curated = pd.read_parquet(CURATED_PATH)

print(f"[INFO] Computing cycle features for {df_curated['series_id'].nunique()} series...")

# 시리즈별로 사이클 특성 계산
feature_list = []

for series_id, group in df_curated.groupby('series_id'):
    # 시계열 데이터 정렬
    group = group.sort_values(['year', 'week']).reset_index(drop=True)
    y = group['claim_count'].values
    
    # 1. PSI (Periodic Seasonality Index) 계산
    psi = compute_psi(y)
    
    # 2. 피크 탐지
    peak_flags = detect_peaks(y)
    
    # 3. 변화점 탐지
    cp_flags = detect_changepoints(y)
    
    # 4. 진폭 계산 (이동 표준편차)
    amplitude = compute_amplitude(y)
    
    # 결과 조합
    features = group[['series_id', 'year', 'week']].copy()
    features['psi'] = psi
    features['peak_flag'] = peak_flags
    features['cp_flag'] = cp_flags
    features['amplitude'] = amplitude
    
    feature_list.append(features)

# 모든 시리즈 결합
df_features = pd.concat(feature_list, ignore_index=True)

print("[INFO] Validating features contract...")
df_features = validate_features(df_features)

print(f"[INFO] Saving features parquet: {FEATURES_PATH}")
df_features.to_parquet(FEATURES_PATH)
print(f"[SUCCESS] Features extracted for {df_features['series_id'].nunique()} series, {len(df_features):,} rows")
