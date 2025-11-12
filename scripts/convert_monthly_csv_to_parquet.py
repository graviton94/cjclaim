# 변환 스크립트: monthly_ts_2021_2023.csv → monthly_ts_2021_2023.parquet
import pandas as pd
csv_path = r"c:\cjclaim\quality-cycles\artifacts\metrics\monthly_ts_2021_2023.csv"
parquet_path = r"c:\cjclaim\quality-cycles\artifacts\metrics\monthly_ts_2021_2023.parquet"
df = pd.read_csv(csv_path)
df.to_parquet(parquet_path, index=False)
print(f"Saved: {parquet_path}")
