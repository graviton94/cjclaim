"""
원본 클레임 데이터(2021-2023)에서 제조일자 기준 주별 시계열 생성
- 발생일자 2021-2023 데이터를 제조일자 기준으로 주차 변환
- lag_class, sample_weight는 labeled 데이터에서 가져옴
- 모든 시리즈에 대해 전체 주차를 생성 (0 포함)
"""
import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("제조일자 기준 주별 시계열 데이터 생성 (0 포함)")
print("=" * 80)

# 1. Labeled 데이터 로드 (제품범주별 lag 정책 적용됨)
labeled_path = "artifacts/metrics/candidates_claims_2021_2023_with_lag_policy.csv"
print(f"\n[1] Labeled 데이터 로드: {labeled_path}")
df = pd.read_csv(labeled_path)
print(f"총 레코드: {len(df):,}건")
print(f"컬럼: {df.columns.tolist()}")

# 2. 날짜 처리 (제조일자 기준으로 year/week 계산)
df['발생일자'] = pd.to_datetime(df['발생일자'])
df['제조일자'] = pd.to_datetime(df['제조일자'])

print(f"\n발생일자 범위: {df['발생일자'].min()} ~ {df['발생일자'].max()}")
print(f"제조일자 범위: {df['제조일자'].min()} ~ {df['제조일자'].max()}")

df['year'] = df['제조일자'].dt.isocalendar().year
df['week'] = df['제조일자'].dt.isocalendar().week

print(f"제조일자 기준 연도 범위: {df['year'].min()} ~ {df['year'].max()}")

print(f"제조일자 기준 연도 범위: {df['year'].min()} ~ {df['year'].max()}")

# 3. series_id 생성
df['series_id'] = (df['플랜트'].astype(str) + '|' + 
                   df['제품범주2'].astype(str) + '|' + 
                   df['중분류(보정)'].astype(str))

# 4. 제조일자 기준 주별 집계
print("\n[2] 제조일자 기준 주별 집계 중...")
df_weekly = df.groupby(['series_id', 'year', 'week', '플랜트', '제품범주2', '중분류(보정)']).agg(
    claim_count=('클레임건수', 'sum'),
    # lag_class와 sample_weight는 weighted average 또는 majority로 처리
    lag_class_mode=('lag_class', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'normal'),
    sample_weight_mean=('sample_weight', 'mean')
).reset_index()

print(f"집계 후 레코드: {len(df_weekly):,}건")
print(f"시리즈 개수: {df_weekly['series_id'].nunique():,}개")
print(f"제조년도 범위: {df_weekly['year'].min()} ~ {df_weekly['year'].max()}")

# 5. 전체 주차 범위 생성 (제조일자 기준)
print("\n[3] 전체 주차 프레임 생성 (0 포함)...")
all_series = df_weekly[['series_id', '플랜트', '제품범주2', '중분류(보정)']].drop_duplicates()

# 제조년도 범위 확인
min_year = int(df_weekly['year'].min())
max_year = int(df_weekly['year'].max())
years = list(range(min_year, max_year + 1))
weeks = list(range(1, 54))  # 1-53주

print(f"제조년도 범위: {min_year} ~ {max_year}")

# 연도-주차 조합 생성
year_week_df = pd.DataFrame([(y, w) for y in years for w in weeks], 
                            columns=['year', 'week'])

# 모든 시리즈 × 모든 주차 조합
full_frame = all_series.merge(year_week_df, how='cross')
print(f"전체 프레임: {len(full_frame):,}건 ({len(all_series):,} 시리즈 × {len(year_week_df):,} 주)")

# 6. 집계 데이터와 병합 (없으면 0)
df_complete = full_frame.merge(
    df_weekly[['series_id', 'year', 'week', 'claim_count', 'lag_class_mode', 'sample_weight_mean']], 
    on=['series_id', 'year', 'week'], 
    how='left'
)
df_complete['claim_count'] = df_complete['claim_count'].fillna(0).astype(int)
df_complete['lag_class'] = df_complete['lag_class_mode'].fillna('normal')
df_complete['sample_weight'] = df_complete['sample_weight_mean'].fillna(1.0)

# 임시 컬럼 제거
df_complete = df_complete.drop(columns=['lag_class_mode', 'sample_weight_mean'])

print(f"\n완전한 시계열: {len(df_complete):,}건")
print(f"Non-zero: {(df_complete['claim_count'] > 0).sum():,}건")
print(f"Zero: {(df_complete['claim_count'] == 0).sum():,}건")

# 7. 컬럼명 정리
df_complete = df_complete.rename(columns={
    '플랜트': 'plant',
    '제품범주2': 'product_cat2',
    '중분류(보정)': 'mid_category'
})

# 8. 정렬
df_complete = df_complete.sort_values(['series_id', 'year', 'week']).reset_index(drop=True)

# 9. Parquet 저장
output_path = Path("data/curated/weekly_timeseries_2021_2023.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)

df_complete.to_parquet(output_path, index=False, engine='pyarrow')

print(f"\n[4] 저장 완료: {output_path}")
print(f"파일 크기: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# 통계 출력
print("\n" + "=" * 80)
print("데이터 통계")
print("=" * 80)
print(f"총 레코드: {len(df_complete):,}건")
print(f"시리즈 개수: {df_complete['series_id'].nunique():,}개")
print(f"연도 범위: {df_complete['year'].min()}-{df_complete['year'].max()}")
print(f"\n시리즈당 평균 레코드: {len(df_complete) / df_complete['series_id'].nunique():.1f}건")
print(f"Non-zero 비율: {(df_complete['claim_count'] > 0).sum() / len(df_complete) * 100:.2f}%")
print(f"\nQuality 메타데이터:")
print(df_complete['lag_class'].value_counts())
print(f"\n컬럼: {df_complete.columns.tolist()}")
print("\n✅ 완료!")
