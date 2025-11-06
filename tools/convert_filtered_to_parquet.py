"""
필터링된 학습 후보 CSV를 parquet으로 변환
"""
import pandas as pd
from pathlib import Path

print("=" * 80)
print("필터링된 학습 데이터 → Parquet 변환")
print("=" * 80)

# 1. 필터링된 후보 데이터 로드
input_csv = Path("artifacts/metrics/candidates_claims_2021_2023_labeled.csv")
print(f"\n[1] 입력 파일: {input_csv}")

df = pd.read_csv(input_csv)
print(f"총 레코드: {len(df):,}건")
print(f"원본 컬럼: {df.columns.tolist()}")

# 2. 날짜 처리 및 year/week 추출
df['발생일자'] = pd.to_datetime(df['발생일자'])
df['year'] = df['발생일자'].dt.year
df['week'] = df['발생일자'].dt.isocalendar().week

print(f"\n연도 범위: {df['year'].min()} ~ {df['year'].max()}")
year_dist = df['year'].value_counts().sort_index()
print("\n연도별 분포:")
for year, count in year_dist.items():
    print(f"  {year}: {count:,}건")

# 3. series_id 생성 (플랜트|제품범주2|중분류)
df['series_id'] = df['플랜트'].astype(str) + '|' + df['제품범주2'].astype(str) + '|' + df['중분류(보정)'].astype(str)

# 4. 컬럼명 정리
df = df.rename(columns={
    '플랜트': 'plant',
    '제품범주2': 'product_cat2',
    '중분류(보정)': 'mid_category',
    '클레임건수': 'claim_count'
})

# 5. lag_class 분포
if 'lag_class' in df.columns:
    print("\nlag_class 분포:")
    print(df['lag_class'].value_counts())

# 6. sample_weight 분포
if 'sample_weight' in df.columns:
    print(f"\nsample_weight 평균: {df['sample_weight'].mean():.3f}")
    print(f"sample_weight 분포:")
    print(df['sample_weight'].value_counts())

# 7. 필요한 컬럼만 선택하여 Parquet 저장
final_columns = ['series_id', 'plant', 'product_cat2', 'mid_category', 
                 'year', 'week', 'claim_count', 'lag_class', 'sample_weight']
df_final = df[final_columns].copy()

print(f"\n최종 컬럼: {df_final.columns.tolist()}")
print(f"시리즈 개수: {df_final['series_id'].nunique():,}개")

output_parquet = Path("data/curated/claims_base_2021_2023.parquet")
output_parquet.parent.mkdir(parents=True, exist_ok=True)

df_final.to_parquet(output_parquet, index=False, engine='pyarrow')

print(f"\n[2] 출력 파일: {output_parquet}")
print(f"파일 크기: {output_parquet.stat().st_size / 1024:.1f} KB")

print("\n" + "=" * 80)
print("✅ 변환 완료!")
print("=" * 80)
