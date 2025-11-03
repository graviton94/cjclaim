"""
원본 claims(2020_2024).csv → 주간 집계/패딩 → curated parquet 자동화 스크립트 (동적 컬럼명 매핑)
"""
import pandas as pd
from src.preprocess import weekly_agg_from_counts
from contracts import validate_curated

# 글로벌 컬럼명 매핑
COLUMN_MAP = {
    "plant": ["플랜트", "plant"],
    "product_cat2": ["제품범주2", "product_cat2"],
    "mid_category": ["중분류(보정)", "mid_category", "중분류"],
    "claim_count": ["count", "claim_count", "y"],
    "date": ["제조일자", "date"],
}

def get_column(df, logical_name):
    for col in COLUMN_MAP[logical_name]:
        if col in df.columns:
            return col
    raise KeyError(f"Column for {logical_name} not found in DataFrame: {df.columns.tolist()}")

RAW_PATH = "c:/cjclaim/quality-cycles/data/claims(2020_2024).csv"
CURATED_PATH = "data/curated/claims.parquet"

print(f"[INFO] Loading raw data: {RAW_PATH}")
df_raw = pd.read_csv(RAW_PATH, encoding="euc-kr")

# ISO 주차 체계를 고려하여 실제 데이터의 최대 연도 확인
dates = pd.to_datetime(df_raw[get_column(df_raw, "date")])
max_iso_year = dates.dt.isocalendar().year.max()
PAD_TO = pd.Timestamp(f"{max_iso_year}-12-31")
print(f"[INFO] Detected maximum ISO year: {max_iso_year}, padding to: {PAD_TO}")

# 동적 그룹 컬럼 추출
group_cols = [get_column(df_raw, "plant"), get_column(df_raw, "product_cat2"), get_column(df_raw, "mid_category")]
date_col = get_column(df_raw, "date")
value_col = get_column(df_raw, "claim_count")

print("[INFO] Aggregating and padding weekly counts...")
df_curated = weekly_agg_from_counts(df_raw, date_col=date_col, value_col=value_col, group_cols=group_cols, pad_to_date=PAD_TO)

# 결과 컬럼명도 동적으로 계약에 맞게 변환
rename_map = {
    group_cols[0]: "plant",
    group_cols[1]: "product_cat2",
    group_cols[2]: "mid_category",
    "y": "claim_count"
}
df_curated = df_curated.rename(columns=rename_map)

# year와 week는 이미 ISO 형식으로 되어 있음

print("[INFO] Validating curated contract...")
df_curated = validate_curated(df_curated)

print(f"[INFO] Saving curated parquet: {CURATED_PATH}")
df_curated.to_parquet(CURATED_PATH)
print("[SUCCESS] Curated data ready for pipeline.")