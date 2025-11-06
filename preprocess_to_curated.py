"""
원본 claims(2020_2024).csv → 월별 집계/패딩 → curated parquet 자동화 스크립트
증분학습 모드: 필터링된 훈련 데이터 (최소 클레임 10건 이상)
"""
import pandas as pd
from src.preprocess import monthly_agg_from_counts
from pathlib import Path
import argparse

# 글로벌 컬럼명 매핑
COLUMN_MAP = {
    "plant": ["플랜트", "plant"],
    "product_cat2": ["제품범주2", "product_cat2"],
    "mid_category": ["중분류(보정)", "mid_category", "중분류"],
    "claim_count": ["count", "claim_count", "y"],
    "date": ["제조일자", "date"],
    "lag_class": ["lag_class"],
    "sample_weight": ["sample_weight"],
}

def get_column(df, logical_name):
    for col in COLUMN_MAP[logical_name]:
        if col in df.columns:
            return col
    raise KeyError(f"Column for {logical_name} not found in DataFrame: {df.columns.tolist()}")

# CLI 인자 파싱
parser = argparse.ArgumentParser(description="Preprocess claims data to curated parquet (MONTHLY)")
parser.add_argument("--mode", choices=["legacy", "incremental"], default="incremental",
                    help="legacy: raw data, incremental: filtered candidates (default)")
parser.add_argument("--input", type=str, help="Custom input CSV path (overrides mode defaults)")
parser.add_argument("--output", type=str, default="data/curated/claims_monthly.parquet", help="Output parquet path")
parser.add_argument("--min-claims", type=int, default=10, help="Minimum total claims per series (default: 10)")
args = parser.parse_args()

# 입력 파일 결정
if args.input:
    RAW_PATH = args.input
elif args.mode == "incremental":
    RAW_PATH = "artifacts/metrics/candidates_filtered_train_2021_2023.csv"
else:
    RAW_PATH = "c:/cjclaim/quality-cycles/data/claims(2020_2024).csv"

CURATED_PATH = args.output

print(f"[INFO] Mode: {args.mode} (MONTHLY)")
print(f"[INFO] Minimum claims per series: {args.min_claims}")
print(f"[INFO] Loading data: {RAW_PATH}")

# 인코딩 자동 탐지
for enc in ['utf-8-sig', 'utf-8', 'euc-kr', 'cp949']:
    try:
        df_raw = pd.read_csv(RAW_PATH, encoding=enc)
        print(f"[INFO] Encoding: {enc}")
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError(f"Failed to decode {RAW_PATH} with any encoding")

# ISO 주차 체계를 고려하여 실제 데이터의 최대 연도/주차 확인
dates = pd.to_datetime(df_raw[get_column(df_raw, "date")])
iso_calendar = dates.dt.isocalendar()

# 기본 동작: 입력 데이터에 존재하는 최대 연도/주차까지만 패딩합니다.
# (이전 구현은 계약상 W1~W53 고정으로 패딩했으나, 증분 업로드 시 불필요한 W53 확장이 발생하여
#  업로드 월/범위에 따라 실제 최대 주차까지만 패딩하도록 변경합니다.)
max_iso_year = int(iso_calendar['year'].max())
max_iso_week = int(iso_calendar['week'].max())
PAD_TO = (max_iso_year, max_iso_week)
print(f"[INFO] Padding to detected max in input: {max_iso_year}-W{max_iso_week:02d}")

# 참고: 레거시 전체 데이터(계약상 연단위 W1~W53 패딩 필요)로 강제하려면
# --mode legacy 또는 별도 --pad53 옵션을 추가하여 이전 동작을 재현할 수 있습니다.

# 동적 그룹 컬럼 추출
group_cols = [get_column(df_raw, "plant"), get_column(df_raw, "product_cat2"), get_column(df_raw, "mid_category")]
date_col = get_column(df_raw, "date")
value_col = get_column(df_raw, "claim_count")

# 증분학습 모드: lag_class, sample_weight 보존
preserve_cols = []
if args.mode == "incremental":
    if "lag_class" in df_raw.columns:
        preserve_cols.append("lag_class")
    if "sample_weight" in df_raw.columns:
        preserve_cols.append("sample_weight")
    print(f"[INFO] Incremental mode - preserving quality metadata: {preserve_cols}")

print("[INFO] Aggregating and padding monthly counts...")
df_curated = monthly_agg_from_counts(
    df_raw, 
    date_col=date_col, 
    value_col=value_col,
    group_cols=group_cols, 
    pad_to_date=(2023, 12),
    min_claims=args.min_claims
)

# 결과 컬럼명도 동적으로 계약에 맞게 변환
rename_map = {
    group_cols[0]: "plant",
    group_cols[1]: "product_cat2",
    group_cols[2]: "mid_category",
    "y": "claim_count"
}
df_curated = df_curated.rename(columns=rename_map)

# year와 month는 이미 정수형으로 되어 있음

# Quality metadata 병합 (증분학습용)
if preserve_cols:
    # 월별 집계된 데이터에 대표 lag_class 계산
    df_raw_copy = df_raw.copy()
    df_raw_copy['year'] = dates.dt.year
    df_raw_copy['month'] = dates.dt.month
    
    # lag_class만 있는 경우
    agg_dict = {'lag_class': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'normal'}
    
    # sample_weight가 있으면 추가
    if 'sample_weight' in df_raw_copy.columns:
        agg_dict['sample_weight'] = 'mean'
    
    quality_meta = df_raw_copy.groupby(group_cols + ['year', 'month']).agg(agg_dict).reset_index()
    
    # 컬럼명 변환
    quality_meta = quality_meta.rename(columns={
        group_cols[0]: "plant",
        group_cols[1]: "product_cat2",
        group_cols[2]: "mid_category"
    })
    
    # 병합할 컬럼 리스트
    merge_cols = ['plant', 'product_cat2', 'mid_category', 'year', 'month', 'lag_class']
    if 'sample_weight' in quality_meta.columns:
        merge_cols.append('sample_weight')
    
    df_curated = df_curated.merge(quality_meta[merge_cols], 
                                  on=['plant', 'product_cat2', 'mid_category', 'year', 'month'], 
                                  how='left')
    # 패딩된 행(merge 후 NaN)은 normal/1.0으로 기본값
    df_curated['lag_class'] = df_curated['lag_class'].fillna('normal')
    if 'sample_weight' in df_curated.columns:
        df_curated['sample_weight'] = df_curated['sample_weight'].fillna(1.0)
    print(f"[INFO] Quality metadata merged - lag_class distribution:")
    print(df_curated['lag_class'].value_counts())

print(f"[INFO] Saving curated parquet: {CURATED_PATH}")
df_curated.to_parquet(CURATED_PATH)
print("[SUCCESS] Curated data ready for pipeline.")