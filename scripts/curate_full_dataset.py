from pathlib import Path
import pandas as pd

def curate_full_dataset(raw_csv_path: str, curated_parquet_path: str):
    """
    2021~2023 전체 데이터에서 월별 집계 및 표준 스키마로 Parquet 저장
    Args:
        raw_csv_path: 원본 CSV 경로 (claims_2021_2023_merged.csv)
        curated_parquet_path: 저장할 Parquet 경로
    Returns:
        dict: {'curated_parquet': path, 'rows': row_count}
    """
    df = pd.read_csv(raw_csv_path)
    # 발생건수 컬럼명 통일
    if 'count' in df.columns:
        df = df.rename(columns={'count': '발생건수'})
    # 날짜 컬럼 변환
    for c in ['발생일자', '제조일자']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    # series_id 생성
    df['series_id'] = df[['플랜트', '제품범주2', '중분류']].astype(str).agg('_'.join, axis=1)
    # 월 키 생성
    df['year'] = df['발생일자'].dt.year
    df['month'] = df['발생일자'].dt.month
    # 월별 집계
    monthly = (df.groupby(['series_id', 'year', 'month'], as_index=False)['발생건수'].sum()
                 .rename(columns={'발생건수': 'claim_count'}))
    Path(curated_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(curated_parquet_path, index=False)
    return {'curated_parquet': curated_parquet_path, 'rows': len(monthly)}

if __name__ == "__main__":
    raw_csv = "artifacts/metrics/claims_2021_2023_merged.csv"
    out_parquet = "artifacts/metrics/claims_monthly_2021_2023.parquet"
    result = curate_full_dataset(raw_csv, out_parquet)
    print(result)