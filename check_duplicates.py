import pandas as pd

# 2021_raw.csv 확인
print("=" * 80)
print("Checking 2021_raw.csv for duplicate keys")
print("=" * 80)

df = pd.read_csv('C:/cjclaim/data/2021_raw.csv', encoding='utf-8-sig')
print(f"Total rows: {len(df):,}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# count 제외한 모든 컬럼으로 중복 확인
dup_cols = [c for c in df.columns if c not in ['count']]
print(f"\nGrouping by: {dup_cols}")

# 같은 키가 여러 행에 나타나는 경우 찾기
dups = df[df.duplicated(subset=dup_cols, keep=False)].sort_values(by=dup_cols)

if len(dups) > 0:
    print(f"\n⚠️  Found {len(dups):,} rows with duplicate keys (same series+date)")
    print("\nExample duplicates (first 20):")
    print(dups.head(20).to_string())
    
    # 그룹화 후 count 합산 확인
    grouped = df.groupby(dup_cols, as_index=False)['count'].sum()
    print(f"\nAfter groupby aggregation: {len(grouped):,} unique rows")
    print(f"Reduction: {len(df) - len(grouped):,} rows")
else:
    print("\n✅ No duplicate keys found (all rows are unique)")

# 전체 3개 파일 확인
print("\n" + "=" * 80)
print("Checking all 3 files")
print("=" * 80)

total_original = 0
total_aggregated = 0

for year in [2021, 2022, 2023]:
    df = pd.read_csv(f'C:/cjclaim/data/{year}_raw.csv', encoding='utf-8-sig')
    original = len(df)
    
    dup_cols = [c for c in df.columns if c not in ['count']]
    grouped = df.groupby(dup_cols, as_index=False)['count'].sum()
    aggregated = len(grouped)
    
    reduction = original - aggregated
    
    print(f"{year}_raw.csv: {original:,} rows → {aggregated:,} rows (reduced: {reduction:,})")
    
    total_original += original
    total_aggregated += aggregated

print(f"\nTotal: {total_original:,} rows → {total_aggregated:,} rows")
print(f"Total reduction: {total_original - total_aggregated:,} rows")
