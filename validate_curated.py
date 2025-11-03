"""
curated parquet 파일의 데이터 품질 검증 스크립트
"""
import pandas as pd
import numpy as np

# 원본 데이터와 curated 데이터 로드
RAW_PATH = "c:/cjclaim/quality-cycles/data/claims(2020_2024).csv"
CURATED_PATH = "data/curated/claims.parquet"

print("=" * 80)
print("데이터 품질 검증 시작")
print("=" * 80)

# 1. 원본 데이터 로드 및 기본 통계
print("\n[1] 원본 데이터 로드 및 분석")
print("-" * 80)
df_raw = pd.read_csv(RAW_PATH, encoding="euc-kr")
print(f"원본 데이터 행 수: {len(df_raw):,}")
print(f"원본 데이터 컬럼: {df_raw.columns.tolist()}")
print(f"날짜 범위: {df_raw['제조일자'].min()} ~ {df_raw['제조일자'].max()}")
print(f"총 클레임 수: {df_raw['count'].sum():,.0f}")

# 원본 데이터의 시리즈 수 계산
series_cols = ['플랜트', '제품범주2', '중분류(보정)']
raw_series = df_raw[series_cols].drop_duplicates()
print(f"유니크 시리즈 수: {len(raw_series)}")

# 2. Curated 데이터 로드 및 기본 통계
print("\n[2] Curated 데이터 로드 및 분석")
print("-" * 80)
df_curated = pd.read_parquet(CURATED_PATH)
print(f"Curated 데이터 행 수: {len(df_curated):,}")
print(f"Curated 데이터 컬럼: {df_curated.columns.tolist()}")
print(f"연도 범위: {df_curated['year'].min()} ~ {df_curated['year'].max()}")
print(f"총 클레임 수: {df_curated['claim_count'].sum():,.0f}")

curated_series = df_curated[['plant', 'product_cat2', 'mid_category']].drop_duplicates()
print(f"유니크 시리즈 수: {len(curated_series)}")

# 3. 데이터 타입 검증
print("\n[3] 데이터 타입 검증")
print("-" * 80)
print(df_curated.dtypes)

# 4. 주차 범위 검증
print("\n[4] 주차 범위 검증")
print("-" * 80)
week_min = df_curated['week'].min()
week_max = df_curated['week'].max()
print(f"주차 범위: {week_min} ~ {week_max}")
if week_min >= 1 and week_max <= 53:
    print("✓ 주차 범위 정상 (1-53)")
else:
    print("✗ 주차 범위 오류!")

# 5. 시리즈별 주차 완성도 검증
print("\n[5] 시리즈별 주차 완성도 검증")
print("-" * 80)
completeness = df_curated.groupby(['series_id', 'year'])['week'].apply(
    lambda s: len(s) == 53 and set(s) == set(range(1, 54))
)
incomplete_series = completeness[~completeness]
if len(incomplete_series) == 0:
    print(f"✓ 모든 시리즈-연도 조합이 53주 완전함 (총 {len(completeness)} 개)")
else:
    print(f"✗ 불완전한 시리즈-연도 조합: {len(incomplete_series)} 개")
    print("\n불완전한 조합 샘플:")
    print(incomplete_series.head(10))

# 6. 결측값 검증
print("\n[6] 결측값 검증")
print("-" * 80)
null_counts = df_curated.isnull().sum()
if null_counts.sum() == 0:
    print("✓ 결측값 없음")
else:
    print("✗ 결측값 발견:")
    print(null_counts[null_counts > 0])

# 7. claim_count 값 검증
print("\n[7] claim_count 값 검증")
print("-" * 80)
print(f"최소값: {df_curated['claim_count'].min()}")
print(f"최대값: {df_curated['claim_count'].max()}")
print(f"평균값: {df_curated['claim_count'].mean():.2f}")
print(f"중앙값: {df_curated['claim_count'].median():.2f}")
negative_counts = df_curated[df_curated['claim_count'] < 0]
if len(negative_counts) == 0:
    print("✓ 모든 claim_count >= 0")
else:
    print(f"✗ 음수 claim_count 발견: {len(negative_counts)} 개")

# 8. 0이 아닌 값의 비율
print("\n[8] 데이터 밀도 분석")
print("-" * 80)
zero_count = (df_curated['claim_count'] == 0).sum()
nonzero_count = (df_curated['claim_count'] > 0).sum()
total_count = len(df_curated)
print(f"0인 행: {zero_count:,} ({zero_count/total_count*100:.2f}%)")
print(f"0이 아닌 행: {nonzero_count:,} ({nonzero_count/total_count*100:.2f}%)")

# 9. 원본 데이터와 총합 비교
print("\n[9] 원본 데이터와 총합 비교")
print("-" * 80)
raw_total = df_raw['count'].sum()
curated_total = df_curated['claim_count'].sum()
diff = curated_total - raw_total
diff_pct = (diff / raw_total * 100) if raw_total > 0 else 0
print(f"원본 총합: {raw_total:,.0f}")
print(f"Curated 총합: {curated_total:,.0f}")
print(f"차이: {diff:,.0f} ({diff_pct:+.4f}%)")
if abs(diff_pct) < 0.01:  # 0.01% 이내 허용
    print("✓ 총합 일치")
else:
    print(f"⚠ 총합 차이 발견 (허용범위: ±0.01%)")

# 10. 시리즈별 샘플 데이터 확인
print("\n[10] 시리즈별 샘플 데이터 (첫 번째 시리즈)")
print("-" * 80)
first_series = df_curated['series_id'].iloc[0]
sample = df_curated[df_curated['series_id'] == first_series].head(10)
print(f"\n시리즈: {first_series}")
print(sample[['year', 'week', 'claim_count']])

# 11. 연도별 주차 수 검증
print("\n[11] 연도별 시리즈별 주차 수 검증")
print("-" * 80)
weeks_per_year_series = df_curated.groupby(['series_id', 'year']).size()
print(f"연도별 시리즈별 주차 수 통계:")
print(weeks_per_year_series.describe())
wrong_week_count = weeks_per_year_series[weeks_per_year_series != 53]
if len(wrong_week_count) == 0:
    print("✓ 모든 시리즈-연도 조합이 정확히 53주")
else:
    print(f"✗ 53주가 아닌 조합: {len(wrong_week_count)} 개")
    print(wrong_week_count.head(10))

# 12. 최종 요약
print("\n" + "=" * 80)
print("검증 결과 요약")
print("=" * 80)
issues = []
if week_min < 1 or week_max > 53:
    issues.append("주차 범위 오류")
if len(incomplete_series) > 0:
    issues.append(f"불완전한 시리즈-연도 조합 {len(incomplete_series)}개")
if null_counts.sum() > 0:
    issues.append("결측값 존재")
if len(negative_counts) > 0:
    issues.append("음수 claim_count 존재")
if abs(diff_pct) >= 0.01:
    issues.append(f"총합 차이 {diff_pct:+.4f}%")
if len(wrong_week_count) > 0:
    issues.append(f"53주가 아닌 조합 {len(wrong_week_count)}개")

if len(issues) == 0:
    print("✓ 모든 검증 통과!")
    print(f"✓ 총 {len(curated_series)} 개 시리즈")
    print(f"✓ 총 {len(df_curated):,} 행 (시리즈 × 연도 × 주차)")
    print(f"✓ 데이터 무결성 확인 완료")
else:
    print("✗ 발견된 문제:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

print("=" * 80)
