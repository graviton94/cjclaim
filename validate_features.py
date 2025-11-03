"""
features parquet 파일의 데이터 품질 검증 스크립트
"""
import pandas as pd
import numpy as np

FEATURES_PATH = "data/features/cycle_features.parquet"

print("=" * 80)
print("Features 데이터 품질 검증")
print("=" * 80)

# Features 데이터 로드
print("\n[1] Features 데이터 로드 및 기본 통계")
print("-" * 80)
df = pd.read_parquet(FEATURES_PATH)
print(f"총 행 수: {len(df):,}")
print(f"컬럼: {df.columns.tolist()}")
print(f"유니크 시리즈 수: {df['series_id'].nunique()}")
print(f"연도 범위: {df['year'].min()} ~ {df['year'].max()}")

# 데이터 타입 검증
print("\n[2] 데이터 타입 검증")
print("-" * 80)
print(df.dtypes)

# 결측값 검증
print("\n[3] 결측값 검증")
print("-" * 80)
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("✓ 결측값 없음")
else:
    print("결측값 발견:")
    print(null_counts[null_counts > 0])

# 각 특성 통계
print("\n[4] PSI (Periodic Seasonality Index) 통계")
print("-" * 80)
print(f"최소값: {df['psi'].min():.4f}")
print(f"최대값: {df['psi'].max():.4f}")
print(f"평균값: {df['psi'].mean():.4f}")
print(f"중앙값: {df['psi'].median():.4f}")
print(f"표준편차: {df['psi'].std():.4f}")

print("\n[5] Peak Flag 통계")
print("-" * 80)
peak_counts = df['peak_flag'].value_counts()
print(peak_counts)
if len(peak_counts) > 0:
    peak_pct = peak_counts.get(1, 0) / len(df) * 100
    print(f"피크 비율: {peak_pct:.2f}%")

print("\n[6] Changepoint Flag 통계")
print("-" * 80)
cp_counts = df['cp_flag'].value_counts()
print(cp_counts)
if len(cp_counts) > 0:
    cp_pct = cp_counts.get(1, 0) / len(df) * 100
    print(f"변화점 비율: {cp_pct:.2f}%")

print("\n[7] Amplitude 통계")
print("-" * 80)
print(f"최소값: {df['amplitude'].min():.4f}")
print(f"최대값: {df['amplitude'].max():.4f}")
print(f"평균값: {df['amplitude'].mean():.4f}")
print(f"중앙값: {df['amplitude'].median():.4f}")
print(f"표준편차: {df['amplitude'].std():.4f}")

# 플래그 값 검증
print("\n[8] Flag 값 검증 (0 또는 1만 허용)")
print("-" * 80)
peak_values = df['peak_flag'].unique()
cp_values = df['cp_flag'].unique()
print(f"Peak flag 유니크 값: {sorted(peak_values)}")
print(f"CP flag 유니크 값: {sorted(cp_values)}")

peak_valid = all(v in [0, 1] for v in peak_values)
cp_valid = all(v in [0, 1] for v in cp_values)

if peak_valid and cp_valid:
    print("✓ 모든 플래그 값이 0 또는 1")
else:
    print("✗ 유효하지 않은 플래그 값 발견")

# 시리즈별 샘플
print("\n[9] 시리즈별 샘플 데이터 (첫 번째 시리즈)")
print("-" * 80)
first_series = df['series_id'].iloc[0]
sample = df[df['series_id'] == first_series].head(20)
print(f"\n시리즈: {first_series}")
print(sample[['year', 'week', 'psi', 'peak_flag', 'cp_flag', 'amplitude']])

# 피크가 있는 샘플
print("\n[10] 피크가 감지된 샘플")
print("-" * 80)
peaks = df[df['peak_flag'] == 1].head(10)
if len(peaks) > 0:
    print(peaks[['series_id', 'year', 'week', 'psi', 'peak_flag', 'amplitude']])
else:
    print("피크가 감지된 데이터 없음")

# 변화점이 있는 샘플
print("\n[11] 변화점이 감지된 샘플")
print("-" * 80)
cps = df[df['cp_flag'] == 1].head(10)
if len(cps) > 0:
    print(cps[['series_id', 'year', 'week', 'psi', 'cp_flag', 'amplitude']])
else:
    print("변화점이 감지된 데이터 없음")

# 최종 요약
print("\n" + "=" * 80)
print("검증 결과 요약")
print("=" * 80)

issues = []
if null_counts.sum() > 0:
    issues.append("결측값 존재")
if not (peak_valid and cp_valid):
    issues.append("유효하지 않은 플래그 값")
if df['psi'].isna().all():
    issues.append("모든 PSI 값이 NaN")
if df['amplitude'].isna().all():
    issues.append("모든 Amplitude 값이 NaN")

if len(issues) == 0:
    print("✓ 모든 검증 통과!")
    print(f"✓ 총 {df['series_id'].nunique()} 개 시리즈")
    print(f"✓ 총 {len(df):,} 행")
    print(f"✓ 피크 감지: {df['peak_flag'].sum():,} 건")
    print(f"✓ 변화점 감지: {df['cp_flag'].sum():,} 건")
    print(f"✓ Feature 추출 완료")
else:
    print("✗ 발견된 문제:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

print("=" * 80)
