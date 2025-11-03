"""
원본 데이터와 curated 데이터 간 총합 차이 원인 조사
"""
import pandas as pd
import numpy as np

RAW_PATH = "c:/cjclaim/quality-cycles/data/claims(2020_2024).csv"
CURATED_PATH = "data/curated/claims.parquet"

print("=" * 80)
print("총합 차이 원인 분석")
print("=" * 80)

# 원본 데이터 로드
df_raw = pd.read_csv(RAW_PATH, encoding="euc-kr")

# ISO 주차 변환
dates = pd.to_datetime(df_raw['제조일자'])
df_raw['year'] = dates.dt.isocalendar().year
df_raw['week'] = dates.dt.isocalendar().week

# Curated 데이터 로드
df_curated = pd.read_parquet(CURATED_PATH)

print("\n[1] 원본 데이터의 연도 분포")
print("-" * 80)
print(df_raw['year'].value_counts().sort_index())

print("\n[2] Curated 데이터의 연도별 총합")
print("-" * 80)
curated_by_year = df_curated.groupby('year')['claim_count'].sum().sort_index()
print(curated_by_year)

print("\n[3] 원본 데이터의 연도별 총합")
print("-" * 80)
raw_by_year = df_raw.groupby('year')['count'].sum().sort_index()
print(raw_by_year)

print("\n[4] 연도별 차이")
print("-" * 80)
diff_by_year = curated_by_year - raw_by_year
print(diff_by_year)

# 2025년 데이터 확인
print("\n[5] 2025년 데이터 확인")
print("-" * 80)
raw_2025 = df_raw[df_raw['year'] == 2025]
print(f"원본 데이터 중 2025년 행 수: {len(raw_2025)}")
if len(raw_2025) > 0:
    print("2025년 데이터 샘플:")
    print(raw_2025[['제조일자', 'year', 'week', 'count']].head(10))
    print(f"2025년 총합: {raw_2025['count'].sum()}")

curated_2025 = df_curated[df_curated['year'] == 2025]
print(f"\nCurated 데이터 중 2025년 행 수: {len(curated_2025)}")
curated_2025_nonzero = curated_2025[curated_2025['claim_count'] > 0]
print(f"2025년 중 0이 아닌 행 수: {len(curated_2025_nonzero)}")
if len(curated_2025_nonzero) > 0:
    print("2025년 0이 아닌 데이터 샘플:")
    print(curated_2025_nonzero[['year', 'week', 'claim_count']].head(10))
    print(f"2025년 총합: {curated_2025['claim_count'].sum()}")

# ISO 주차가 다음 해로 넘어가는 경우 확인
print("\n[6] 연말/연초 주차 이슈 확인")
print("-" * 80)
print("2024년 12월 마지막 날짜들:")
dec_2024 = df_raw[df_raw['제조일자'].str.startswith('2024-12')]
if len(dec_2024) > 0:
    sample = dec_2024.tail(10)
    print(sample[['제조일자', 'year', 'week', 'count']])
    print(f"\n2024년 12월의 ISO year 분포:")
    print(sample['year'].value_counts())

print("\n2020년 1월 첫 날짜들:")
jan_2020 = df_raw[df_raw['제조일자'].str.startswith('2020-01')]
if len(jan_2020) > 0:
    sample = jan_2020.head(10)
    print(sample[['제조일자', 'year', 'week', 'count']])
    print(f"\n2020년 1월의 ISO year 분포:")
    print(sample['year'].value_counts())

# 시리즈별 집계 비교
print("\n[7] 시리즈별 집계 비교 (차이가 있는 시리즈)")
print("-" * 80)
# 원본 데이터 시리즈별 집계
raw_agg = df_raw.groupby(['플랜트', '제품범주2', '중분류(보정)'])['count'].sum()
raw_agg = raw_agg.reset_index()
raw_agg['series_id'] = raw_agg[['플랜트', '제품범주2', '중분류(보정)']].astype(str).agg('|'.join, axis=1)

# Curated 데이터 시리즈별 집계
curated_agg = df_curated.groupby('series_id')['claim_count'].sum().reset_index()

# 병합 및 비교
comparison = raw_agg[['series_id', 'count']].merge(
    curated_agg, 
    on='series_id', 
    how='outer',
    suffixes=('_raw', '_curated')
)
comparison['diff'] = comparison['claim_count'] - comparison['count']
diff_series = comparison[abs(comparison['diff']) > 0.001]
if len(diff_series) > 0:
    print(f"차이가 있는 시리즈 수: {len(diff_series)}")
    print(diff_series.sort_values('diff', ascending=False).head(10))
else:
    print("모든 시리즈에서 총합 일치")

print("\n[8] 총 차이 요약")
print("-" * 80)
print(f"원본 총합: {df_raw['count'].sum():,.0f}")
print(f"Curated 총합: {df_curated['claim_count'].sum():,.0f}")
print(f"차이: {df_curated['claim_count'].sum() - df_raw['count'].sum():,.0f}")
print(f"차이 비율: {(df_curated['claim_count'].sum() - df_raw['count'].sum()) / df_raw['count'].sum() * 100:.4f}%")
