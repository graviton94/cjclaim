# 월별 집계 스크립트: claims_2021_2023_merged.csv → monthly_ts_2021_2023.csv
import pandas as pd
src = r"c:\cjclaim\quality-cycles\artifacts\metrics\claims_2021_2023_merged.csv"
out = r"c:\cjclaim\quality-cycles\artifacts\metrics\monthly_ts_2021_2023.csv"
df = pd.read_csv(src)
df['발생일자'] = pd.to_datetime(df['발생일자'], errors='coerce')
df = df.dropna(subset=['발생일자'])
df['year'] = df['발생일자'].dt.year
df['month'] = df['발생일자'].dt.month
df['series_id'] = df[['플랜트','제품범주2','중분류']].astype(str).agg('_'.join, axis=1)
grouped = df.groupby(['series_id','year','month'], as_index=False)['count'].sum()
grouped.to_csv(out, index=False, encoding='utf-8-sig')
print(f"Saved: {out}")
